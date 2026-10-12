"""Per-subject errors for Figure 1: resting heart rate under the within-subject temporal 8:2 split, CNN (mean of
3 seeds) and per-subject median, one value per subject. Writes results/per_subject_temporal.json.
"""
import json

import numpy as np

import gate_vitals as gv
from prep_scenarios import OUTDIR


def main():
    X, hr, subj, tpos = [], [], [], []
    for p in sorted(OUTDIR.glob("GDN*_Resting.npz")):
        d = np.load(p); ok = np.isfinite(d["hr"])
        X.append(d["X"][ok]); hr.append(d["hr"][ok]); subj += [p.name[:7]] * int(ok.sum()); tpos.append(d["starts"][ok])
    X = np.concatenate(X); hr = np.concatenate(hr); subj = np.array(subj); tpos = np.concatenate(tpos)
    n = len(hr); idx = np.arange(n); te = np.zeros(n, bool)
    for s in np.unique(subj):
        r = idx[subj == s]; r = r[np.argsort(tpos[subj == s])]; te[r[int(0.8 * len(r)):]] = True
    tr = ~te
    preds = np.mean([gv.cnn(X[tr], hr[tr], X[te], s) for s in [0, 1, 2]], 0)
    out = {}
    for s in np.unique(subj):
        m = subj[te] == s
        out[s] = {"cnn_mae": float(np.abs(preds[m] - hr[te][m]).mean()),
                  "subj_median_mae": float(np.abs(np.median(hr[tr][subj[tr] == s]) - hr[te][m]).mean()),
                  "hr_mean": float(hr[subj == s].mean())}
    json.dump(out, open(gv.OUT / "per_subject_temporal.json", "w"), indent=1)
    c = np.array([v["cnn_mae"] for v in out.values()]); l = np.array([v["subj_median_mae"] for v in out.values()])
    print(f"{len(out)} subjects: CNN mean {c.mean():.2f}, lookup mean {l.mean():.2f}, lookup better on {(l < c).sum()} subjects")


if __name__ == "__main__":
    main()
