"""Heart rate and blood pressure per scenario under the three protocols, from the cache of prep_scenarios.py.

For each scenario (and all pooled): CNN (gate_vitals.Net; two heads run separately for HR and for SBP/DBP), the
per-subject median of the training part, the population median, and the subject-independent Schmidt comb+prior
estimate for HR. Protocols: random windows 80/20; within-subject temporal 8:2; subject-disjoint 5-fold. 3 seeds.
Writes results/eval_scenarios.json. Run from the project folder: python scripts/eval_scenarios.py [scenario ...]
"""
import json
import sys
import time

import numpy as np
from sklearn.model_selection import GroupKFold

import gate_bp
import gate_vitals as gv
from prep_scenarios import OUTDIR, SCEN

SEEDS = [0, 1, 2]


def load(scen):
    X, hr, bp, est, subj, tpos = [], [], [], [], [], []
    for p in sorted(OUTDIR.glob(f"GDN*_{scen}.npz")):  # not the hs_ac_<scenario> caches
        d = np.load(p)
        X.append(d["X"]); hr.append(d["hr"]); bp.append(np.column_stack([d["sbp"], d["dbp"]])); est.append(d["est"])
        subj += [p.name[:7]] * len(d["hr"]); tpos.append(d["starts"])
    return (np.concatenate(X), np.concatenate(hr), np.concatenate(bp), np.concatenate(est), np.array(subj),
            np.concatenate(tpos))


def folds_for(subj, tpos, n):
    idx = np.arange(n)
    rs = np.random.RandomState(0); perm = rs.permutation(n)
    te = np.zeros(n, bool); te[perm[: n // 5]] = True
    out = {"random_windows_80_20": [(~te, te)]}
    te = np.zeros(n, bool)
    for s in np.unique(subj):
        r = idx[subj == s]; r = r[np.argsort(tpos[subj == s])]; te[r[int(0.8 * len(r)):]] = True
    out["within_subject_temporal_8_2"] = [(~te, te)]
    k = min(5, len(np.unique(subj)))
    out["subject_disjoint_5fold"] = [(np.isin(idx, tr), np.isin(idx, tt)) for tr, tt in GroupKFold(k).split(np.zeros(n), groups=subj)]
    return out


def mae(a, b):
    return float(np.abs(a - b).mean())


def evaluate(scen, X, hr, bp, est, subj, tpos, t0):
    res = {}
    # ---- heart rate
    ok = np.isfinite(hr)
    Xh, yh, eh, sh, th = X[ok], hr[ok], est[ok], subj[ok], tpos[ok]
    stats = {"n": int(ok.sum()), "n_subjects": int(len(np.unique(sh))), "hr_mean": float(yh.mean()), "hr_sd": float(yh.std()),
             "across_subject_sd": float(np.std([yh[sh == s].mean() for s in np.unique(sh)])),
             "within_subject_sd": float(np.mean([yh[sh == s].std() for s in np.unique(sh)])),
             "schmidt_mae_all": mae(eh, yh)}
    res["hr_stats"] = stats
    print(f"[{scen}] HR: {stats['n']} windows, {stats['n_subjects']} subjects, mean {stats['hr_mean']:.1f}, across-subject SD "
          f"{stats['across_subject_sd']:.1f}, within-subject SD {stats['within_subject_sd']:.1f}, Schmidt MAE {stats['schmidt_mae_all']:.2f}", flush=True)
    res["hr"] = {}
    for name, folds in folds_for(sh, th, len(yh)).items():
        acc = {"cnn": [], "subj_median": [], "pop_median": [], "schmidt": []}
        for tr, te in folds:
            pop = np.median(yh[tr]); sm = {s: np.median(yh[tr][sh[tr] == s]) for s in np.unique(sh[tr])}
            acc["pop_median"].append(mae(pop, yh[te]))
            acc["subj_median"].append(mae(np.array([sm.get(s, pop) for s in sh[te]]), yh[te]))
            acc["schmidt"].append(mae(eh[te], yh[te]))
            acc["cnn"].append(float(np.mean([mae(gv.cnn(Xh[tr], yh[tr], Xh[te], s), yh[te]) for s in SEEDS])))
        r = {k: float(np.mean(v)) for k, v in acc.items()}
        res["hr"][name] = r
        print(f"  HR {name:<28} CNN {r['cnn']:5.2f}  subj-median {r['subj_median']:5.2f}  pop-median {r['pop_median']:5.2f}  "
              f"Schmidt {r['schmidt']:5.2f}   ({time.time() - t0:.0f}s)", flush=True)
    # ---- blood pressure
    okb = np.isfinite(bp).all(1)
    if okb.sum() < 200:
        return res
    Xb, yb, sb, tb = X[okb], bp[okb], subj[okb], tpos[okb]
    res["bp_stats"] = {"n": int(okb.sum()), "n_subjects": int(len(np.unique(sb))),
                       "within_subject_sd_sbp": float(np.mean([yb[sb == s, 0].std() for s in np.unique(sb)])),
                       "across_subject_sd_sbp": float(np.std([yb[sb == s, 0].mean() for s in np.unique(sb)]))}
    res["bp"] = {}
    for name, folds in folds_for(sb, tb, len(yb)).items():
        acc = {k: [] for k in ["cnn_sbp", "cnn_dbp", "subj_sbp", "subj_dbp", "pop_sbp", "pop_dbp"]}
        for tr, te in folds:
            pop = np.median(yb[tr], 0); sm = {s: np.median(yb[tr][sb[tr] == s], 0) for s in np.unique(sb[tr])}
            ps = np.array([sm.get(s, pop) for s in sb[te]])
            acc["pop_sbp"].append(mae(pop[0], yb[te, 0])); acc["pop_dbp"].append(mae(pop[1], yb[te, 1]))
            acc["subj_sbp"].append(mae(ps[:, 0], yb[te, 0])); acc["subj_dbp"].append(mae(ps[:, 1], yb[te, 1]))
            preds = [gate_bp.cnn2(Xb[tr], yb[tr], Xb[te], s) for s in SEEDS]
            acc["cnn_sbp"].append(float(np.mean([mae(p[:, 0], yb[te, 0]) for p in preds])))
            acc["cnn_dbp"].append(float(np.mean([mae(p[:, 1], yb[te, 1]) for p in preds])))
        r = {k: float(np.mean(v)) for k, v in acc.items()}
        res["bp"][name] = r
        print(f"  BP {name:<28} CNN {r['cnn_sbp']:5.2f}/{r['cnn_dbp']:5.2f}  subj-median {r['subj_sbp']:5.2f}/{r['subj_dbp']:5.2f}  "
              f"pop-median {r['pop_sbp']:5.2f}/{r['pop_dbp']:5.2f}   ({time.time() - t0:.0f}s)", flush=True)
    return res


def main():
    t0 = time.time()
    scens = sys.argv[1:] or SCEN + ["all"]
    out = {}
    if sys.argv[1:] and (gv.OUT / "eval_scenarios.json").exists():  # a partial rerun merges into the existing file
        out = json.load(open(gv.OUT / "eval_scenarios.json"))
    for scen in scens:
        if scen == "all":
            parts = [load(s) for s in SCEN]
            X = np.concatenate([p[0] for p in parts]); hr = np.concatenate([p[1] for p in parts])
            bp = np.concatenate([p[2] for p in parts]); est = np.concatenate([p[3] for p in parts])
            subj = np.concatenate([p[4] for p in parts])
            # make windows of different scenarios distinct in time order by offsetting each scenario by 1e5 s
            tpos = np.concatenate([p[5] + 1e5 * k for k, p in enumerate(parts)])
        else:
            X, hr, bp, est, subj, tpos = load(scen)
        out[scen] = evaluate(scen, X, hr, bp, est, subj, tpos, t0)
        json.dump(out, open(gv.OUT / "eval_scenarios.json", "w"), indent=1)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
