"""Is the resting subject-disjoint failure an undertrained network? Train the gate's CNN on resting windows,
subject-disjoint (the gate's 5 folds, 3 seeds), for 80 epochs, and record the MAE on the training subjects and on
the held-out subjects at checkpoints. Also record the fit on the training subjects' windows, so that memorisation
(training error falling, held-out error flat at the population median) is visible. Writes results/epoch_sweep.json.
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

import gate_vitals as gv
from prep_scenarios import OUTDIR

SEEDS = [0, 1, 2]
CHECK = [5, 10, 20, 40, 60, 80]


def train_sweep(Xtr, ytr, Xte, yte, seed):
    torch.manual_seed(seed)
    m = gv.Net().to(gv.DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CHECK[-1])
    mu, sd = ytr.mean(), ytr.std()
    xt = torch.as_tensor(Xtr, device=gv.DEV); yt = torch.as_tensor((ytr - mu) / sd, device=gv.DEV, dtype=torch.float32)
    xe = torch.as_tensor(Xte, device=gv.DEV)
    n = len(xt); out = {}
    for ep in range(1, CHECK[-1] + 1):
        m.train()
        perm = torch.randperm(n, device=gv.DEV)
        for s in range(0, n, 64):
            b = perm[s:s + 64]
            if len(b) < 2:
                continue
            loss = F.smooth_l1_loss(m(xt[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep in CHECK:
            m.eval()
            with torch.no_grad():
                pe = torch.cat([m(xe[s:s + 512]) for s in range(0, len(xe), 512)]).cpu().numpy() * sd + mu
                pt = torch.cat([m(xt[s:s + 512]) for s in range(0, len(xt), 512)]).cpu().numpy() * sd + mu
            out[ep] = {"test_mae": float(np.abs(pe - yte).mean()), "train_mae": float(np.abs(pt - ytr).mean())}
    return out


def main():
    t0 = time.time()
    X, hr, subj = [], [], []
    for p in sorted(OUTDIR.glob("GDN*_Resting.npz")):
        d = np.load(p); ok = np.isfinite(d["hr"])
        X.append(d["X"][ok]); hr.append(d["hr"][ok]); subj += [p.name[:7]] * int(ok.sum())
    X = np.concatenate(X); hr = np.concatenate(hr); subj = np.array(subj)
    res = {"epochs": CHECK, "folds": []}
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, hr, subj)):
        pop = float(np.abs(np.median(hr[tr]) - hr[te]).mean())
        runs = [train_sweep(X[tr], hr[tr], X[te], hr[te], s) for s in SEEDS]
        fold = {"pop_median": pop, "by_epoch": {str(ep): {"test_mae": float(np.mean([r[ep]["test_mae"] for r in runs])),
                                                        "train_mae": float(np.mean([r[ep]["train_mae"] for r in runs]))} for ep in CHECK}}
        res["folds"].append(fold)
        print(f"fold {k}: pop-median {pop:.2f}; " + "  ".join(f"ep{ep}: test {fold['by_epoch'][str(ep)]['test_mae']:.2f} train {fold['by_epoch'][str(ep)]['train_mae']:.2f}" for ep in CHECK) + f"  ({time.time() - t0:.0f}s)", flush=True)
    res["mean"] = {str(ep): {"test_mae": float(np.mean([f["by_epoch"][str(ep)]["test_mae"] for f in res["folds"]])),
                             "train_mae": float(np.mean([f["by_epoch"][str(ep)]["train_mae"] for f in res["folds"]]))} for ep in CHECK}
    res["pop_median"] = float(np.mean([f["pop_median"] for f in res["folds"]]))
    print("mean:", json.dumps(res["mean"]), "pop-median", round(res["pop_median"], 2))
    json.dump(res, open(gv.OUT / "epoch_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
