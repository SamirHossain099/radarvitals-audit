"""Blood-pressure floors on Schellenberger 2020 (resting): what a per-subject median and a population median score
under the split protocols the BP papers use, against their published MAEs (TriDP-PTM 4.73 mmHg; Liu et al. 2026
SBP 3.21 / DBP 2.25; Dual-Stage 2025 SBP 5.00 / DBP 3.96; MARU-MTL subject-disjoint SBP 4.05 / DBP 3.08;
Wang et al. Sensors 2025 subject-wise SBP 6.52 / DBP 3.98).

Labels: tfm_bp (continuous finger pressure, 200 Hz) -> per-beat systolic maxima and diastolic minima -> window
means over the same 30-s windows, stride 5 s, as gate_vitals.py. No radar used: these are label-only baselines.
Writes results/bp_floor.json.
"""
import json

import numpy as np
import scipy.io as sio
import scipy.signal as sg

import gate_vitals as gv

RAW = gv.RAW
OUT = gv.OUT
FS_BP = 200
WIN, STRIDE = 30, 5


def beat_pressures(bp, fs):
    b, a = sg.butter(3, 10 / (fs / 2))
    x = sg.filtfilt(b, a, bp)
    pk, _ = sg.find_peaks(x, distance=int(0.33 * fs), prominence=5)
    tr, _ = sg.find_peaks(-x, distance=int(0.33 * fs), prominence=5)
    return pk / fs, x[pk], tr / fs, x[tr]


def main():
    rows = []
    for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
        s = d.name
        m = sio.loadmat(next(d.glob(f"{s}_*_Resting.mat")))
        bp = m["tfm_bp"].ravel()
        if not np.isfinite(bp).all() or bp.std() < 1:
            print(s, "no usable BP"); continue
        ts, sv, td, dv = beat_pressures(bp, FS_BP)
        dur = len(bp) / FS_BP
        for w in np.arange(0, dur - WIN, STRIDE):
            ms = (ts >= w) & (ts < w + WIN); md = (td >= w) & (td < w + WIN)
            if ms.sum() < 10 or md.sum() < 10:
                continue
            sbp, dbp = np.median(sv[ms]), np.median(dv[md])
            if 70 < sbp < 220 and 30 < dbp < 130 and sbp > dbp + 15:
                rows.append((s, w, sbp, dbp))
    subj = np.array([r[0] for r in rows]); t = np.array([r[1] for r in rows])
    Y = np.array([[r[2], r[3]] for r in rows])
    names = np.unique(subj)
    print(f"{len(Y)} windows, {len(names)} subjects; SBP {Y[:, 0].mean():.1f} +- {Y[:, 0].std():.1f}, DBP {Y[:, 1].mean():.1f} +- {Y[:, 1].std():.1f}")
    print(f"across-subject SD of subject means: SBP {np.std([Y[subj == s, 0].mean() for s in names]):.1f}, DBP {np.std([Y[subj == s, 1].mean() for s in names]):.1f}; "
          f"mean within-subject SD: SBP {np.mean([Y[subj == s, 0].std() for s in names]):.1f}, DBP {np.mean([Y[subj == s, 1].std() for s in names]):.1f}")
    res = {"n_windows": int(len(Y)), "n_subjects": int(len(names)), "protocols": {}}
    n = len(Y); idx = np.arange(n)

    def ev(name, tr, te):
        pop = np.median(Y[tr], 0)
        sm = {s: np.median(Y[tr][subj[tr] == s], 0) for s in np.unique(subj[tr])}
        ps = np.array([sm.get(s, pop) for s in subj[te]])
        r = {"pop_median_mae_sbp": float(np.abs(pop[0] - Y[te, 0]).mean()), "pop_median_mae_dbp": float(np.abs(pop[1] - Y[te, 1]).mean()),
             "subj_median_mae_sbp": float(np.abs(ps[:, 0] - Y[te, 0]).mean()), "subj_median_mae_dbp": float(np.abs(ps[:, 1] - Y[te, 1]).mean())}
        res["protocols"][name] = r
        print(f"{name:<28} per-subject median SBP {r['subj_median_mae_sbp']:5.2f} DBP {r['subj_median_mae_dbp']:5.2f} | "
              f"population median SBP {r['pop_median_mae_sbp']:5.2f} DBP {r['pop_median_mae_dbp']:5.2f}")

    rs = np.random.RandomState(0); perm = rs.permutation(n)
    te = np.zeros(n, bool); te[perm[: n // 5]] = True
    ev("random_windows_80_20", ~te, te)
    te = np.zeros(n, bool)
    for s in names:
        r_ = idx[subj == s]; r_ = r_[np.argsort(t[subj == s])]; te[r_[int(0.8 * len(r_)):]] = True
    ev("within_subject_temporal_8_2", ~te, te)
    # subject-disjoint: per-subject median is unavailable, so only the population median applies
    rs = np.random.RandomState(1); order = rs.permutation(names)
    maes = []
    for k in range(5):
        held = order[k::5]; te = np.isin(subj, held)
        pop = np.median(Y[~te], 0)
        maes.append([np.abs(pop[0] - Y[te, 0]).mean(), np.abs(pop[1] - Y[te, 1]).mean()])
    maes = np.mean(maes, 0)
    res["protocols"]["subject_disjoint_5fold"] = {"pop_median_mae_sbp": float(maes[0]), "pop_median_mae_dbp": float(maes[1])}
    print(f"{'subject_disjoint_5fold':<28} population median SBP {maes[0]:5.2f} DBP {maes[1]:5.2f}")
    res["published"] = {"TriDP-PTM 2026 (within-subject 8:2)": {"BP MAE": 4.73}, "Liu 2026 (segment shuffle 3:1:1)": {"SBP": 3.21, "DBP": 2.25},
                        "Dual-Stage 2025 (65/15/20, unit unspecified)": {"SBP": 5.00, "DBP": 3.96},
                        "MARU-MTL 2026 (subject-disjoint)": {"SBP": 4.05, "DBP": 3.08}, "Wang Sensors 2025 (subject-wise 5-fold)": {"SBP": 6.52, "DBP": 3.98}}
    json.dump(res, open(OUT / "bp_floor.json", "w"), indent=1)


if __name__ == "__main__":
    main()
