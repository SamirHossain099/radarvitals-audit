"""Blood pressure from radar under the published protocols, against the per-subject-median floor.

Labels: window-median systolic (beat maxima) and diastolic (beat minima) finger pressure from tfm_bp (200 Hz),
on the gate's 30-s windows (stride 5 s), joined to the radar window cache by subject and start time.
Protocols: random windows 80/20 (segment-level shuffle, as Liu et al. 2026 and Dual-Stage 2025); within-subject
temporal 8:2 (TriDP-PTM); subject-disjoint 5-fold (MARU-MTL, Wang et al. 2025). Predictors: the gate's 1-D CNN with
two outputs (3 seeds); per-subject median of the training part; population median.
Writes results/gate_bp.json.
"""
import json
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

import bp_floor
import gate_vitals as gv

OUT = gv.OUT
SEEDS = [0, 1, 2]


def bp_labels():
    rows = {}
    for d in sorted(p for p in gv.RAW.iterdir() if p.is_dir()):
        s = d.name
        m = sio.loadmat(next(d.glob(f"{s}_*_Resting.mat")))
        bp = m["tfm_bp"].ravel()
        if not np.isfinite(bp).all() or bp.std() < 1:
            continue
        ts, sv, td, dv = bp_floor.beat_pressures(bp, bp_floor.FS_BP)
        dur = len(bp) / bp_floor.FS_BP
        for w in np.arange(0, dur - gv.WIN, gv.STRIDE):
            ms = (ts >= w) & (ts < w + gv.WIN); md = (td >= w) & (td < w + gv.WIN)
            if ms.sum() < 10 or md.sum() < 10:
                continue
            sbp, dbp = np.median(sv[ms]), np.median(dv[md])
            if 70 < sbp < 220 and 30 < dbp < 130 and sbp > dbp + 15:
                rows[(s, int(round(w)))] = (sbp, dbp)
    return rows


def cnn2(Xtr, Ytr, Xte, seed):
    torch.manual_seed(seed)
    m = gv.Net().to(gv.DEV)
    m.fc = torch.nn.Linear(256, 2).to(gv.DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, gv.EPOCHS)
    mu, sd = Ytr.mean(0), Ytr.std(0)
    xt = torch.as_tensor(Xtr, device=gv.DEV); yt = torch.as_tensor((Ytr - mu) / sd, device=gv.DEV, dtype=torch.float32)
    xe = torch.as_tensor(Xte, device=gv.DEV)
    n = len(xt)
    for _ in range(gv.EPOCHS):
        m.train()
        perm = torch.randperm(n, device=gv.DEV)
        for s in range(0, n, 64):
            b = perm[s:s + 64]
            if len(b) < 2:
                continue
            loss = F.smooth_l1_loss(m(xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        p = torch.cat([m(xe[s:s + 512]) for s in range(0, len(xe), 512)]).cpu().numpy()
    return p * sd + mu


def main():
    t0 = time.time()
    lab = bp_labels()
    X, Y, subj, tpos = [], [], [], []
    for p in sorted(gv.DER.glob("*_resting.npz")):
        s = p.name[:7]
        d = np.load(p)
        for k, w in enumerate(d["starts"]):
            key = (s, int(round(w)))
            if key in lab:
                X.append(d["X"][k]); Y.append(lab[key]); subj.append(s); tpos.append(w)
    X = np.stack(X); Y = np.array(Y, np.float32); subj = np.array(subj); tpos = np.array(tpos)
    names = np.unique(subj); n = len(Y); idx = np.arange(n)
    print(f"{n} windows, {len(names)} subjects; SBP {Y[:, 0].mean():.1f} +- {Y[:, 0].std():.1f}, DBP {Y[:, 1].mean():.1f} +- {Y[:, 1].std():.1f}", flush=True)
    res = {"n_windows": int(n), "n_subjects": int(len(names)), "protocols": {}}

    def run(name, folds):
        acc = {k: [] for k in ["cnn_sbp", "cnn_dbp", "subj_sbp", "subj_dbp", "pop_sbp", "pop_dbp"]}
        for tr, te in folds:
            pop = np.median(Y[tr], 0)
            sm = {s: np.median(Y[tr][subj[tr] == s], 0) for s in np.unique(subj[tr])}
            ps = np.array([sm.get(s, pop) for s in subj[te]])
            acc["pop_sbp"].append(np.abs(pop[0] - Y[te, 0]).mean()); acc["pop_dbp"].append(np.abs(pop[1] - Y[te, 1]).mean())
            acc["subj_sbp"].append(np.abs(ps[:, 0] - Y[te, 0]).mean()); acc["subj_dbp"].append(np.abs(ps[:, 1] - Y[te, 1]).mean())
            preds = [cnn2(X[tr], Y[tr], X[te], s) for s in SEEDS]
            acc["cnn_sbp"].append(float(np.mean([np.abs(p[:, 0] - Y[te, 0]).mean() for p in preds])))
            acc["cnn_dbp"].append(float(np.mean([np.abs(p[:, 1] - Y[te, 1]).mean() for p in preds])))
        r = {k: float(np.mean(v)) for k, v in acc.items()}; r["n_folds"] = len(folds)
        res["protocols"][name] = r
        print(f"{name:<28} CNN SBP {r['cnn_sbp']:5.2f} DBP {r['cnn_dbp']:5.2f} | subj-median {r['subj_sbp']:5.2f} {r['subj_dbp']:5.2f} | "
              f"pop-median {r['pop_sbp']:5.2f} {r['pop_dbp']:5.2f}   ({time.time() - t0:.0f}s)", flush=True)

    rs = np.random.RandomState(0); perm = rs.permutation(n)
    te = np.zeros(n, bool); te[perm[: n // 5]] = True
    run("random_windows_80_20", [(~te, te)])
    te = np.zeros(n, bool)
    for s in names:
        r_ = idx[subj == s]; r_ = r_[np.argsort(tpos[subj == s])]; te[r_[int(0.8 * len(r_)):]] = True
    run("within_subject_temporal_8_2", [(~te, te)])
    gkf = GroupKFold(5)
    run("subject_disjoint_5fold", [(np.isin(idx, tr), np.isin(idx, tt)) for tr, tt in gkf.split(X, Y, subj)])
    res["published"] = bp_floor.main.__doc__ or ""
    json.dump(res, open(OUT / "gate_bp.json", "w"), indent=1)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
