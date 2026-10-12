"""Exact-window replications of two published protocols on the resting recordings.

Liu et al. 2026 (Sensors): 5.12-s segments, randomly shuffled 3:1:1 at the segment level, resting subset, SBP/DBP.
   Here: 5.12-s windows (512 samples at 100 Hz), non-overlapping, shuffled 60/20/20 with the 20% test part scored.
TriDP-PTM (arXiv 2605.25725): for each individual 8:2 in time; training windows of 2048 samples at 250 Hz
   (8.192 s) with stride 256 (1.024 s); test windows non-overlapping. Here: 8.192-s windows (819 samples at 100 Hz),
   training stride 1.024 s, test non-overlapping, the last 20% of each subject's recording as test; HR and SBP/DBP.
Labels for short windows: median RR of the beats inside the window (>= 4 beats); median beat systolic/diastolic
pressure (>= 4 beats). Predictors: the gate's CNN (3 seeds), per-subject median of the training part, population
median. Writes results/replicate_protocols.json.
"""
import json
import time

import numpy as np
import scipy.io as sio
import scipy.signal as sg

import bp_floor
import gate_bp
import gate_vitals as gv

SEEDS = [0, 1, 2]
FS_IN, FS = gv.FS_IN, gv.FS


def beats(ecg, fs):
    b, a = sg.butter(3, [5 / (fs / 2), 25 / (fs / 2)], "band")
    x = sg.filtfilt(b, a, ecg) ** 2
    x = np.convolve(x, np.ones(int(0.12 * fs)) / int(0.12 * fs), mode="same")
    pk, _ = sg.find_peaks(x, distance=int(0.33 * fs), prominence=np.percentile(x, 98) * 0.2)
    return pk / fs


def load_subject(sid):
    m = sio.loadmat(next((gv.RAW / sid).glob(f"{sid}_*_Resting.mat")))
    i, q = m["radar_i"].ravel(), m["radar_q"].ravel()
    b, a = sg.butter(4, 20 / (FS_IN / 2))
    i = sg.decimate(sg.filtfilt(b, a, i), FS_IN // FS, zero_phase=True)
    q = sg.decimate(sg.filtfilt(b, a, q), FS_IN // FS, zero_phase=True)
    ph = np.unwrap(np.arctan2(q - np.median(q), i - np.median(i)))
    rt = beats(m["tfm_ecg2"].ravel(), FS_IN)
    rr = np.diff(rt)
    bp = m["tfm_bp"].ravel()
    bpok = np.isfinite(bp).all() and bp.std() > 1
    if bpok:
        ts, sv, td, dv = bp_floor.beat_pressures(bp, bp_floor.FS_BP)
    return (i, q, ph), (rt, rr), (ts, sv, td, dv) if bpok else None


def windows(sig, starts, wlen):
    i, q, ph = sig
    X = np.zeros((len(starts), 3, wlen), np.float32)
    for k, s in enumerate(starts):
        a0 = int(round(s * FS)); a1 = a0 + wlen
        for c, x in enumerate((i, q, ph)):
            seg = sg.detrend(x[a0:a1]); X[k, c] = seg / (seg.std() + 1e-9)
    return X


def labels(ecg, bpb, starts, wsec):
    rt, rr = ecg
    hr = np.full(len(starts), np.nan); sbp = np.full(len(starts), np.nan); dbp = np.full(len(starts), np.nan)
    for k, s in enumerate(starts):
        m = (rt[1:] >= s) & (rt[1:] < s + wsec)
        r = rr[m]; r = r[(r > 0.33) & (r < 2.0)]
        need = 4 if wsec >= 5 else 2  # a 2-s window holds about two beats
        if len(r) >= need and (len(r) < 3 or np.std(r) / np.mean(r) < 0.25):
            hr[k] = 60 / np.median(r)
        if bpb is not None:
            ts, sv, td, dv = bpb
            ms = (ts >= s) & (ts < s + wsec); md = (td >= s) & (td < s + wsec)
            if ms.sum() >= need and md.sum() >= need:
                a, b_ = np.median(sv[ms]), np.median(dv[md])
                if 70 < a < 220 and 30 < b_ < 130 and a > b_ + 15:
                    sbp[k], dbp[k] = a, b_
    return hr, sbp, dbp


def cnn_fit(Xtr, ytr, Xte, seed, n_out):
    if n_out == 1:
        return gv.cnn(Xtr, ytr, Xte, seed)
    return gate_bp.cnn2(Xtr, ytr, Xte, seed)


def score(name, Xtr, Ytr, str_, Xte, Yte, ste, res, t0):
    """Ytr/Yte columns: hr, sbp, dbp (NaN allowed per column)."""
    out = {}
    for tgt, cols in (("hr", [0]), ("bp", [1, 2])):
        oktr = np.isfinite(Ytr[:, cols]).all(1); okte = np.isfinite(Yte[:, cols]).all(1)
        if oktr.sum() < 100 or okte.sum() < 50:
            continue
        ytr, yte = Ytr[oktr][:, cols], Yte[okte][:, cols]
        pop = np.median(ytr, 0); sm = {s: np.median(ytr[str_[oktr] == s], 0) for s in np.unique(str_[oktr])}
        ps = np.array([sm.get(s, pop) for s in ste[okte]])
        preds = [cnn_fit(Xtr[oktr], ytr.squeeze(1) if len(cols) == 1 else ytr, Xte[okte], s, len(cols)) for s in SEEDS]
        preds = [p.reshape(len(p), -1) for p in preds]
        r = {"n_train": int(oktr.sum()), "n_test": int(okte.sum()),
             "cnn": [float(np.mean([np.abs(p[:, j] - yte[:, j]).mean() for p in preds])) for j in range(len(cols))],
             "subj_median": [float(np.abs(ps[:, j] - yte[:, j]).mean()) for j in range(len(cols))],
             "pop_median": [float(np.abs(pop[j] - yte[:, j]).mean()) for j in range(len(cols))]}
        out[tgt] = r
        fmt = lambda v: "/".join(f"{x:5.2f}" for x in v)
        print(f"{name:<34} {tgt:<3} CNN {fmt(r['cnn'])}  subj-median {fmt(r['subj_median'])}  pop-median {fmt(r['pop_median'])}  "
              f"(train {r['n_train']}, test {r['n_test']}; {time.time() - t0:.0f}s)", flush=True)
    res[name] = out


def main():
    t0 = time.time()
    subs = sorted(p.name for p in gv.RAW.iterdir() if p.is_dir())
    data = {s: load_subject(s) for s in subs}
    print(f"loaded {len(data)} subjects ({time.time() - t0:.0f}s)", flush=True)
    res = {}

    # ---- Liu et al. 2026: 5.12-s non-overlapping segments, shuffled 60/20/20
    wsec, wlen = 5.12, 512
    X, Y, S = [], [], []
    for s, (sig, ecg, bpb) in data.items():
        dur = len(sig[0]) / FS
        starts = np.arange(0, dur - wsec, wsec)
        X.append(windows(sig, starts, wlen)); Y.append(np.column_stack(labels(ecg, bpb, starts, wsec))); S += [s] * len(starts)
    X = np.concatenate(X); Y = np.concatenate(Y); S = np.array(S)
    rs = np.random.RandomState(0); perm = rs.permutation(len(Y)); n = len(Y)
    te = np.zeros(n, bool); te[perm[: n // 5]] = True
    tr = np.zeros(n, bool); tr[perm[int(0.4 * n):]] = True  # 60% train, 20% unused "validation", 20% test
    score("Liu2026_5.12s_shuffled_60_20_20", X[tr], Y[tr], S[tr], X[te], Y[te], S[te], res, t0)
    # the same windows split by subject (5-fold) for contrast
    uniq = np.unique(S); rs = np.random.RandomState(1); order = rs.permutation(uniq)
    agg = {}
    for k in range(5):
        held = order[k::5]; te = np.isin(S, held)
        score(f"Liu2026_5.12s_subject_disjoint_fold{k}", X[~te], Y[~te], S[~te], X[te], Y[te], S[te], agg, t0)
    res["Liu2026_5.12s_subject_disjoint_5fold"] = {tgt: {k: list(np.mean([agg[f][tgt][k] for f in agg if tgt in agg[f]], 0))
                                                         for k in ["cnn", "subj_median", "pop_median"]}
                                                   for tgt in ["hr", "bp"]}
    print("Liu2026 subject-disjoint mean:", json.dumps(res["Liu2026_5.12s_subject_disjoint_5fold"]), flush=True)

    # ---- Dual-Stage 2025: 2-s windows, 65/15/20 split (unit unstated; taken here as a shuffle of non-overlapping windows)
    wsec, wlen = 2.0, 200
    X, Y, S = [], [], []
    for s, (sig, ecg, bpb) in data.items():
        dur = len(sig[0]) / FS
        starts = np.arange(0, dur - wsec, wsec)
        X.append(windows(sig, starts, wlen)); Y.append(np.column_stack(labels(ecg, bpb, starts, wsec))); S += [s] * len(starts)
    X = np.concatenate(X); Y = np.concatenate(Y); S = np.array(S)
    rs = np.random.RandomState(0); perm = rs.permutation(len(Y)); n = len(Y)
    te = np.zeros(n, bool); te[perm[: n // 5]] = True
    tr = np.zeros(n, bool); tr[perm[int(0.35 * n):]] = True
    score("DualStage2025_2s_shuffled_65_15_20", X[tr], Y[tr], S[tr], X[te], Y[te], S[te], res, t0)

    # ---- TriDP-PTM: per subject 8:2 in time; train windows 8.192 s stride 1.024 s; test windows non-overlapping
    wsec, wlen = 8.192, 819
    Xtr, Ytr, Str, Xte, Yte, Ste = [], [], [], [], [], []
    for s, (sig, ecg, bpb) in data.items():
        dur = len(sig[0]) / FS; cut = 0.8 * dur
        st_tr = np.arange(0, cut - wsec, 1.024); st_te = np.arange(cut, dur - wsec, wsec)
        Xtr.append(windows(sig, st_tr, wlen)); Ytr.append(np.column_stack(labels(ecg, bpb, st_tr, wsec))); Str += [s] * len(st_tr)
        Xte.append(windows(sig, st_te, wlen)); Yte.append(np.column_stack(labels(ecg, bpb, st_te, wsec))); Ste += [s] * len(st_te)
    score("TriDP_8.192s_per_subject_8_2", np.concatenate(Xtr), np.concatenate(Ytr), np.array(Str),
          np.concatenate(Xte), np.concatenate(Yte), np.array(Ste), res, t0)
    json.dump(res, open(gv.OUT / "replicate_protocols.json", "w"), indent=1)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
