"""Is the pooled subject-disjoint heart-rate result a scenario artefact? Train the CNN on all five scenarios of
24 subjects (the eval_scenarios 'all' folds), then score the held-out subjects' windows scenario by scenario,
against the population median of that scenario (computed on training subjects, i.e. a predictor that knows the
scenario) and the pooled population median. If the network only learned scenario-level means, its per-scenario
error would sit at the per-scenario median. Writes results/pooled_breakdown.json.
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
    X = np.concatenate([p[0] for p in parts]); hr = np.concatenate([p[1] for p in parts]); est = np.concatenate([p[3] for p in parts])
    subj = np.concatenate([p[4] for p in parts]); scen = np.concatenate([[s] * len(p[1]) for s, p in zip(SCEN, parts)])
    ok = np.isfinite(hr)
    X, hr, est, subj, scen = X[ok], hr[ok], est[ok], subj[ok], scen[ok]
    n = len(hr); idx = np.arange(n)
    folds = [(np.isin(idx, tr), np.isin(idx, te)) for tr, te in GroupKFold(5).split(np.zeros(n), groups=subj)]
    acc = {s: {"cnn": [], "scen_median": [], "pooled_median": [], "schmidt": [], "n": 0} for s in SCEN}
    acc["all"] = {"cnn": [], "scen_median": [], "pooled_median": [], "schmidt": [], "n": 0}
    for k, (tr, te) in enumerate(folds):
        pred = np.mean([gv.cnn(X[tr], hr[tr], X[te], s) for s in SEEDS], 0)
        pooled_med = np.median(hr[tr])
        for s in SCEN + ["all"]:
            m = (scen[te] == s) if s != "all" else np.ones(te.sum(), bool)
            if m.sum() == 0:
                continue
            smed = np.median(hr[tr][scen[tr] == s]) if s != "all" else pooled_med
            y = hr[te][m]
            acc[s]["cnn"].append(float(np.abs(pred[m] - y).mean()))
            acc[s]["scen_median"].append(float(np.abs(smed - y).mean()))
            acc[s]["pooled_median"].append(float(np.abs(pooled_med - y).mean()))
            acc[s]["schmidt"].append(float(np.abs(est[te][m] - y).mean()))
            acc[s]["n"] += int(m.sum())
        print(f"fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    res = {s: {m: (float(np.mean(v)) if isinstance(v, list) else v) for m, v in d.items()} for s, d in acc.items()}
    res["folds"] = {s: acc[s]["cnn"] for s in acc}
    for s in SCEN + ["all"]:
        r = res[s]
        print(f"{s:<9} n {r['n']:5d}  pooled-trained CNN {r['cnn']:5.2f}  scenario median {r['scen_median']:5.2f}  pooled median {r['pooled_median']:5.2f}  Schmidt {r['schmidt']:5.2f}")
    json.dump(res, open(gv.OUT / "pooled_breakdown.json", "w"), indent=1)


if __name__ == "__main__":
    main()
