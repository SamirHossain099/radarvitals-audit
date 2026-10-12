"""Gate for the radar vital-signs audit on Schellenberger et al. (Sci Data 2020), resting scenario, heart rate.

Labels: ECG R-peaks (tfm_ecg1, 2000 Hz) -> median RR per window -> bpm. Inputs: radar I/Q at 2000 Hz, lowpassed and
decimated to 100 Hz, per-window standardised I, Q and unwrapped phase (3, 3000) for 30-s windows, stride 5 s.
Protocols: random windows 80/20 (pooled), within-subject temporal 8:2 (TriDP-PTM's protocol), leave-subjects-out
(5-fold GroupKFold, 6 subjects per fold). Baselines: population median, per-subject median (training part), classical
FFT peak of the phase signal in 0.7-2.5 Hz, 1-NN on the band spectrum. Model: a small 1-D CNN, 3 seeds.
Writes results/gate_vitals.json; caches per-subject windows in N:/Datasets/Schellenberger2020/derived.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.signal as sg
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

ROOT = Path(os.environ.get("SCHELLENBERGER_ROOT", r"N:\Datasets\Schellenberger2020"))  # raw/GDN00xx/*.mat below it
RAW = ROOT / "raw"
DER = ROOT / "derived"
DER.mkdir(parents=True, exist_ok=True)
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FS_IN, FS = 2000, 100
WIN, STRIDE = 30, 5  # seconds
SEEDS = [0, 1, 2]
EPOCHS = 40


def ecg_hr_per_window(ecg, fs, starts, win):
    """R-peaks by band-pass, squaring and peak picking; HR = 60 / median RR within the window; NaN if unreliable."""
    b, a = sg.butter(3, [5 / (fs / 2), 25 / (fs / 2)], "band")
    x = sg.filtfilt(b, a, ecg)
    x = x ** 2
    x = np.convolve(x, np.ones(int(0.12 * fs)) / int(0.12 * fs), mode="same")
    peaks, _ = sg.find_peaks(x, distance=int(0.33 * fs), prominence=np.percentile(x, 98) * 0.2)
    t = peaks / fs
    rr = np.diff(t)
    hr = np.full(len(starts), np.nan)
    for i, s in enumerate(starts):
        m = (t[1:] >= s) & (t[1:] < s + win)
        r = rr[m]
        if len(r) < 0.5 * win * 40 / 60:  # fewer beats than 40 bpm would give over half the window
            continue
        r = r[(r > 0.33) & (r < 2.0)]
        if len(r) < 10 or np.std(r) / np.mean(r) > 0.25:
            continue
        hr[i] = 60 / np.median(r)
    return hr


def radar_windows(i, q, starts, win):
    b, a = sg.butter(4, 20 / (FS_IN / 2))
    i = sg.decimate(sg.filtfilt(b, a, i), FS_IN // FS, zero_phase=True)
    q = sg.decimate(sg.filtfilt(b, a, q), FS_IN // FS, zero_phase=True)
    ph = np.unwrap(np.arctan2(q - np.median(q), i - np.median(i)))
    X = np.zeros((len(starts), 3, win * FS), np.float32)
    for k, s in enumerate(starts):
        a0, a1 = int(s * FS), int(s * FS) + win * FS
        for c, sig in enumerate((i, q, ph)):
            seg = sg.detrend(sig[a0:a1])
            X[k, c] = seg / (seg.std() + 1e-9)
    return X


def fft_hr(X):
    """Classical estimator: band-pass the phase channel 0.7-2.5 Hz, take the spectral peak."""
    b, a = sg.butter(3, [0.7 / (FS / 2), 2.5 / (FS / 2)], "band")
    out = np.zeros(len(X))
    f = np.fft.rfftfreq(X.shape[-1] * 4, 1 / FS)
    band = (f >= 0.7) & (f <= 2.5)
    for k in range(len(X)):
        x = sg.filtfilt(b, a, X[k, 2])
        p = np.abs(np.fft.rfft(x * np.hanning(len(x)), X.shape[-1] * 4))
        out[k] = 60 * f[band][np.argmax(p[band])]
    return out


def band_spectrum(X):
    f = np.fft.rfftfreq(X.shape[-1], 1 / FS)
    band = (f >= 0.5) & (f <= 3.0)
    p = np.abs(np.fft.rfft(X[:, 2] * np.hanning(X.shape[-1]), axis=-1))[:, band]
    return (p / (p.sum(1, keepdims=True) + 1e-9)).astype(np.float32)


def prepare():
    subs = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    for s in subs:
        out = DER / f"{s}_resting.npz"
        if out.exists():
            continue
        f = next((RAW / s).glob(f"{s}_*_Resting.mat"))
        m = sio.loadmat(f)
        assert int(m["fs_radar"].ravel()[0]) == FS_IN and int(m["fs_ecg"].ravel()[0]) == FS_IN
        i, q = m["radar_i"].ravel(), m["radar_q"].ravel()
        dur = len(i) / FS_IN
        starts = np.arange(0, dur - WIN, STRIDE)
        # lead 2 is the cleaner lead throughout (lead 1 is NaN for GDN0003 and noisy for GDN0001/0018); where lead 1
        # is usable the two must agree within 2 bpm, otherwise the window is dropped
        hr2 = ecg_hr_per_window(m["tfm_ecg2"].ravel(), FS_IN, starts, WIN)
        e1 = m["tfm_ecg1"].ravel()
        hr = hr2.copy()
        agree = None
        if np.isfinite(e1).all():
            hr1 = ecg_hr_per_window(e1, FS_IN, starts, WIN)
            both = np.isfinite(hr1) & np.isfinite(hr2)
            agree = float(np.mean(np.abs(hr1[both] - hr2[both]) <= 2)) if both.any() else None
            hr[both & (np.abs(hr1 - hr2) > 2)] = np.nan
        # the Task Force Monitor's own beat-to-beat HR, where the file carries it, as an external check on the labels
        tfm_note = ""
        tp = m["tfm_param"]
        if tp.dtype.names and "HR" in tp.dtype.names:
            thr = np.asarray(tp["HR"][0, 0]).ravel().astype(float)
            thr = thr[np.isfinite(thr) & (thr > 30) & (thr < 200)]
            if len(thr):
                tfm_note = f", TFM beat HR median {np.median(thr):.1f}"
        X = radar_windows(i, q, starts, WIN)
        np.savez_compressed(out, X=X, hr=hr, starts=starts)
        print(f"{s}: {dur:.0f} s, {len(starts)} windows, {np.isfinite(hr).sum()} labelled, HR median {np.nanmedian(hr):.1f}"
              f", lead agreement {agree}{tfm_note}", flush=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [3, 16, 32, 64, 64, 128]
        self.convs = nn.ModuleList([nn.Conv1d(ch[i], ch[i + 1], 7, padding=3) for i in range(5)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(c) for c in ch[1:]])
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        for c, b in zip(self.convs, self.bns):
            x = F.relu(b(c(x)))
            if x.shape[-1] >= 6:  # 2-s windows (200 samples) run out of length before the fifth pool
                x = F.max_pool1d(x, 3)
        x = torch.cat([x.mean(2), x.amax(2)], 1)
        return self.fc(F.dropout(x, 0.3, self.training)).squeeze(1)


def cnn(Xtr, ytr, Xte, seed):
    torch.manual_seed(seed)
    m = Net().to(DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    mu, sd = ytr.mean(), ytr.std()
    xt = torch.as_tensor(Xtr, device=DEV); yt = torch.as_tensor((ytr - mu) / sd, device=DEV, dtype=torch.float32)
    xe = torch.as_tensor(Xte, device=DEV)
    n = len(xt)
    for _ in range(EPOCHS):
        m.train()
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, 64):
            b = perm[s:s + 64]
            if len(b) < 2:
                continue
            loss = F.smooth_l1_loss(m(xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        p = torch.cat([m(xe[s:s + 512]) for s in range(0, len(xe), 512)]).cpu().numpy()
    return p * sd + mu


def main():
    t0 = time.time()
    prepare()
    X, y, subj, tpos = [], [], [], []
    for p in sorted(DER.glob("*_resting.npz")):
        d = np.load(p)
        ok = np.isfinite(d["hr"])
        X.append(d["X"][ok]); y.append(d["hr"][ok]); subj += [p.name[:7]] * int(ok.sum()); tpos.append(d["starts"][ok])
    X = np.concatenate(X); y = np.concatenate(y); subj = np.array(subj); tpos = np.concatenate(tpos)
    spec = band_spectrum(X)
    fft_est = fft_hr(X)
    print(f"{len(y)} windows, {len(np.unique(subj))} subjects; HR {y.mean():.1f} +- {y.std():.1f} bpm; "
          f"across-subject SD of subject means {np.std([y[subj == s].mean() for s in np.unique(subj)]):.1f}, "
          f"mean within-subject SD {np.mean([y[subj == s].std() for s in np.unique(subj)]):.1f}; "
          f"FFT estimator MAE {np.abs(fft_est - y).mean():.2f} bpm  ({time.time() - t0:.0f}s)", flush=True)
    res = {"n_windows": int(len(y)), "n_subjects": int(len(np.unique(subj))), "hr_mean": float(y.mean()), "hr_sd": float(y.std()),
           "fft_mae_all": float(np.abs(fft_est - y).mean()), "protocols": {}}

    def evaluate(name, folds):
        acc = {"cnn": [], "pop_median": [], "subj_median": [], "fft": [], "knn_spec": []}
        for tr, te in folds:
            acc["pop_median"].append(np.abs(np.median(y[tr]) - y[te]).mean())
            sm = {s: np.median(y[tr][subj[tr] == s]) for s in np.unique(subj[tr])}
            pm = np.array([sm.get(s, np.median(y[tr])) for s in subj[te]])
            acc["subj_median"].append(np.abs(pm - y[te]).mean())
            acc["fft"].append(np.abs(fft_est[te] - y[te]).mean())
            a = torch.as_tensor(spec[tr], device=DEV); b = torch.as_tensor(spec[te], device=DEV)
            idx = torch.cdist(b, a).argmin(1).cpu().numpy()
            acc["knn_spec"].append(np.abs(y[tr][idx] - y[te]).mean())
            acc["cnn"].append(float(np.mean([np.abs(cnn(X[tr], y[tr], X[te], s) - y[te]).mean() for s in SEEDS])))
        r = {k: float(np.mean(v)) for k, v in acc.items()}
        r["folds_cnn"] = acc["cnn"]; r["n_folds"] = len(folds)
        res["protocols"][name] = r
        print(f"{name:<26} CNN {r['cnn']:5.2f}  FFT {r['fft']:5.2f}  1-NN {r['knn_spec']:5.2f}  subj-median {r['subj_median']:5.2f}  "
              f"pop-median {r['pop_median']:5.2f}   ({time.time() - t0:.0f}s)", flush=True)

    n = len(y); idx = np.arange(n)
    rs = np.random.RandomState(0); perm = rs.permutation(n)
    te = np.zeros(n, bool); te[perm[: n // 5]] = True
    evaluate("random_windows_80_20", [(~te, te)])
    te = np.zeros(n, bool)
    for s in np.unique(subj):
        rows = idx[subj == s]; rows = rows[np.argsort(tpos[subj == s])]
        te[rows[int(0.8 * len(rows)):]] = True
    evaluate("within_subject_temporal_8_2", [(~te, te)])
    gkf = GroupKFold(5)
    evaluate("subject_disjoint_5fold", [(np.isin(idx, tr), np.isin(idx, tt)) for tr, tt in gkf.split(X, y, subj)])
    json.dump(res, open(OUT / "gate_vitals.json", "w"), indent=1)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
