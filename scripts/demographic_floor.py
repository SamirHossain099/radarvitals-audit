"""Subject-disjoint floors that use no radar: blood pressure and heart rate predicted from demographics only
(age, sex, height, weight, BMI from additional_data.xlsx), ridge regression, 5-fold by subject on the resting window
labels. Any radar model evaluated subject-disjoint should beat this, not only the population median.
Also checks why GDN0009 has no heart-rate labels in the Apnea scenario. Writes results/demographic_floor.json.
"""
import json

import numpy as np
import openpyxl
from sklearn.linear_model import Ridge

import gate_vitals as gv
from prep_scenarios import OUTDIR


def demographics():
    wb = openpyxl.load_workbook(gv.RAW.parent / "zips" / "additional_data.xlsx", read_only=True)
    ws = wb["Subject overview"]
    out = {}
    for row in ws.iter_rows(values_only=True):
        if row[1] and str(row[1]).startswith("GDN"):
            age, sex, h, w = float(row[2]), 1.0 if row[3] == "M" else 0.0, float(row[4]), float(row[5])
            out[row[1]] = np.array([age, sex, h, w, w / (h / 100) ** 2])
    return out


def main():
    demo = demographics()
    X, Y, S = [], [], []
    for p in sorted(OUTDIR.glob("GDN*_Resting.npz")):
        s = p.name[:7]; d = np.load(p)
        y = np.column_stack([d["hr"], d["sbp"], d["dbp"]])
        X.append(np.tile(demo[s], (len(y), 1))); Y.append(y); S += [s] * len(y)
    X = np.concatenate(X); Y = np.concatenate(Y); S = np.array(S)
    # the same subject folds as gate_vitals.py / gate_bp.py: GroupKFold(5) over the labelled windows of each target
    from sklearn.model_selection import GroupKFold
    res = {}
    for j, name in enumerate(["hr", "sbp", "dbp"]):
        ok = np.isfinite(Y[:, j]); maes, pops = [], []
        idx = np.flatnonzero(ok)
        for tr_i, te_i in GroupKFold(5).split(np.zeros(len(idx)), groups=S[idx]):
            tr = np.zeros(len(S), bool); te = np.zeros(len(S), bool); tr[idx[tr_i]] = True; te[idx[te_i]] = True
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            m = Ridge(alpha=10.0).fit((X[tr] - mu) / sd, Y[tr, j])
            maes.append(np.abs(m.predict((X[te] - mu) / sd) - Y[te, j]).mean())
            pops.append(np.abs(np.median(Y[tr, j]) - Y[te, j]).mean())
        res[name] = {"ridge_demographics_mae": float(np.mean(maes)), "pop_median_mae": float(np.mean(pops))}
        print(f"{name}: demographics ridge {np.mean(maes):.2f}  population median {np.mean(pops):.2f}")
    # Apnea label check for GDN0009
    d = np.load(OUTDIR / "GDN0009_Apnea.npz")
    res["apnea_GDN0009"] = {"windows": int(len(d["hr"])), "labelled": int(np.isfinite(d["hr"]).sum())}
    import scipy.io as sio
    m = sio.loadmat(next((gv.RAW / "GDN0009").glob("GDN0009_*_Apnea.mat")))
    e2 = m["tfm_ecg2"].ravel(); e1 = m["tfm_ecg1"].ravel()
    res["apnea_GDN0009"]["ecg2_finite"] = bool(np.isfinite(e2).all()); res["apnea_GDN0009"]["ecg2_std"] = float(np.nanstd(e2))
    res["apnea_GDN0009"]["ecg1_finite"] = bool(np.isfinite(e1).all()); res["apnea_GDN0009"]["ecg1_std"] = float(np.nanstd(e1))
    hr2 = gv.ecg_hr_per_window(np.nan_to_num(e2), gv.FS_IN, d["starts"], gv.WIN)
    res["apnea_GDN0009"]["lead2_alone_labelled"] = int(np.isfinite(hr2).sum())
    print("GDN0009 Apnea:", res["apnea_GDN0009"])
    json.dump(res, open(gv.OUT / "demographic_floor.json", "w"), indent=1)


if __name__ == "__main__":
    main()
