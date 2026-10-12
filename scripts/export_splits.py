"""Write the split files the paper releases: for every scenario's 30-s / 5-s windows (prep_scenarios.py), which
windows are test under the random-window and within-subject temporal protocols, and the subject folds of the
subject-disjoint protocol, exactly as eval_scenarios.py draws them. Output: splits/<scenario>.json with one record
per labelled window (subject, start_s, random_test, temporal_test, fold) and the fold membership by subject; plus
splits/pooled.json for the union. A window is identified by its subject and start time in seconds from the start
of the (cut) recording, so the files apply to any re-derivation of the windows.
"""
import json
from pathlib import Path

import numpy as np

import gate_vitals as gv
from eval_scenarios import folds_for, load
from prep_scenarios import SCEN

OUT = Path(__file__).resolve().parents[1] / "splits"
OUT.mkdir(exist_ok=True)


def export(name, hr, subj, tpos):
    ok = np.isfinite(hr)
    s, t = subj[ok], tpos[ok]
    f = folds_for(s, t, len(s))
    rand = f["random_windows_80_20"][0][1]; temp = f["within_subject_temporal_8_2"][0][1]
    fold = np.full(len(s), -1)
    for k, (_, te) in enumerate(f["subject_disjoint_5fold"]):
        fold[te] = k
    rec = {"protocols": {"random_windows_80_20": "20% of labelled windows drawn with numpy RandomState(0)",
                         "within_subject_temporal_8_2": "the last 20% of each subject's windows in time",
                         "subject_disjoint_5fold": "sklearn GroupKFold(5) over subjects, fold = test fold of the window"},
           "window_s": gv.WIN, "stride_s": gv.STRIDE, "n": int(len(s)),
           "subject_folds": {str(k): sorted(set(s[fold == k].tolist())) for k in range(5)},
           "windows": [{"subject": str(a), "start_s": float(b), "random_test": bool(c), "temporal_test": bool(d), "fold": int(e)}
                       for a, b, c, d, e in zip(s, t, rand, temp, fold)]}
    json.dump(rec, open(OUT / f"{name}.json", "w"), indent=0)
    print(f"{name}: {len(s)} windows, random test {int(rand.sum())}, temporal test {int(temp.sum())}, folds {[int((fold == k).sum()) for k in range(5)]}")


def main():
    parts = {sc: load(sc) for sc in SCEN}
    for sc, (X, hr, bp, est, subj, tpos) in parts.items():
        export(sc, hr, subj, tpos)
    hr = np.concatenate([p[1] for p in parts.values()]); subj = np.concatenate([p[4] for p in parts.values()])
    tpos = np.concatenate([p[5] + 1e5 * k for k, p in enumerate(parts.values())])
    export("pooled", hr, subj, tpos)


if __name__ == "__main__":
    main()
