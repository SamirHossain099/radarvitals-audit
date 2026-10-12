"""Size-matched control for the pooled subject-disjoint result: the pooled training set (five scenarios, 24
subjects per fold) subsampled at random to the size of the resting-only training set, so that the gain of pooling
is separated from the gain of four times more windows. 5 folds, 3 seeds. Writes results/pooled_sizematched.json.
"""
import json
import time

import numpy as np
from sklearn.model_selection import GroupKFold

import gate_vitals as gv
from eval_scenarios import load
from prep_scenarios import SCEN

SEEDS = [0, 1, 2]


def main():
    t0 = time.time()
    parts = [load(s) for s in SCEN]
    X = np.concatenate([p[0] for p in parts]); hr = np.concatenate([p[1] for p in parts])
    subj = np.concatenate([p[4] for p in parts]); scen = np.concatenate([[s] * len(p[1]) for s, p in zip(SCEN, parts)])
    ok = np.isfinite(hr); X, hr, subj, scen = X[ok], hr[ok], subj[ok], scen[ok]
    n = len(hr)
    n_rest = int((scen == "Resting").sum())
    folds = list(GroupKFold(5).split(np.zeros(n), groups=subj))
    res = {"n_pooled": int(n), "n_resting": n_rest, "folds": []}
    for k, (tr_i, te_i) in enumerate(folds):
        tr = np.zeros(n, bool); tr[tr_i] = True; te = np.zeros(n, bool); te[te_i] = True
        rs = np.random.RandomState(k)
        sub = rs.choice(np.flatnonzero(tr), size=int(round(n_rest * tr.sum() / n)), replace=False)
        trs = np.zeros(n, bool); trs[sub] = True
        pred = np.mean([gv.cnn(X[trs], hr[trs], X[te], s) for s in SEEDS], 0)
        pop = np.median(hr[trs])
        res["folds"].append({"n_train_sub": int(trs.sum()), "cnn": float(np.abs(pred - hr[te]).mean()),
                             "pop_median": float(np.abs(pop - hr[te]).mean())})
        print(f"fold {k}: subsampled train {trs.sum()} -> CNN {res['folds'][-1]['cnn']:.2f}, pop-median {res['folds'][-1]['pop_median']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    res["cnn"] = float(np.mean([f["cnn"] for f in res["folds"]])); res["pop_median"] = float(np.mean([f["pop_median"] for f in res["folds"]]))
    print(f"size-matched pooled, subject-disjoint: CNN {res['cnn']:.2f}  pop-median {res['pop_median']:.2f}")
    json.dump(res, open(gv.OUT / "pooled_sizematched.json", "w"), indent=1)


if __name__ == "__main__":
    main()
