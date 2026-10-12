"""Mean error +- SD of the no-radar pressure predictors under the random-window and temporal protocols, on exactly
the windows and splits of gate_bp.py, so the paper can place the lookup against the AAMI/ISO 81060-2 criterion
(mean error within +-5 mmHg and SD at most 8 mmHg, on at least 85 subjects, which this dataset does not have).
Writes results/bp_lookup_stats.json.
"""
import json

import numpy as np

import gate_bp
import gate_vitals as gv


def main():
    lab = gate_bp.bp_labels()
    Y, subj, tpos = [], [], []
    for p in sorted(gv.DER.glob("*_resting.npz")):
        s = p.name[:7]; d = np.load(p)
        for w in d["starts"]:
            key = (s, int(round(w)))
            if key in lab:
                Y.append(lab[key]); subj.append(s); tpos.append(w)
    Y = np.array(Y, np.float32); subj = np.array(subj); tpos = np.array(tpos)
    n = len(Y); idx = np.arange(n)
    rs = np.random.RandomState(0); perm = rs.permutation(n)
    te_r = np.zeros(n, bool); te_r[perm[: n // 5]] = True
    te_t = np.zeros(n, bool)
    for s in np.unique(subj):
        r = idx[subj == s]; r = r[np.argsort(tpos[subj == s])]; te_t[r[int(0.8 * len(r)):]] = True
    out = {"n_windows": int(n), "n_subjects": int(len(np.unique(subj)))}
    for name, te in (("random_windows_80_20", te_r), ("within_subject_temporal_8_2", te_t)):
        tr = ~te
        pop = np.median(Y[tr], 0); sm = {s: np.median(Y[tr][subj[tr] == s], 0) for s in np.unique(subj[tr])}
        ps = np.array([sm[s] for s in subj[te]])
        for pname, pred in (("subj_median", ps), ("pop_median", np.broadcast_to(pop, Y[te].shape))):
            err = pred - Y[te]
            out[f"{name}/{pname}"] = {"me_sbp": float(err[:, 0].mean()), "sd_sbp": float(err[:, 0].std()),
                                      "me_dbp": float(err[:, 1].mean()), "sd_dbp": float(err[:, 1].std()),
                                      "mae_sbp": float(np.abs(err[:, 0]).mean()), "mae_dbp": float(np.abs(err[:, 1]).mean()),
                                      "n_test": int(te.sum())}
            r = out[f"{name}/{pname}"]
            print(f"{name:<28} {pname:<12} SBP ME {r['me_sbp']:+5.2f} SD {r['sd_sbp']:5.2f} | DBP ME {r['me_dbp']:+5.2f} SD {r['sd_dbp']:5.2f} | MAE {r['mae_sbp']:.2f}/{r['mae_dbp']:.2f}")
    json.dump(out, open(gv.OUT / "bp_lookup_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
