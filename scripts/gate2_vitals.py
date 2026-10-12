"""Follow-ups to gate_vitals.py.

1. A subject-independent classical estimator that uses the radar heart sounds (the dataset authors' route) rather
   than the Doppler pulse: phase at 2000 Hz, band-pass 16-80 Hz, Hilbert envelope, autocorrelation per 30-s window,
   dominant lag in 0.33-1.5 s -> bpm. No training, so its error is the same under every protocol; it shows whether
   the subject-disjoint regime is beatable at all with this data.
2. A subject-identity probe: the same 1-D CNN trained to classify the 30 subjects from radar windows under the
   random-window and within-subject temporal splits. High accuracy means a heart-rate regressor trained under
   those splits can look the subject up (TriDP-PTM reports 98.9% on the same data).
Writes results/gate2_vitals.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.signal as sg
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_vitals import DER, DEV, EPOCHS, FS_IN, OUT, RAW, WIN, Net

SEEDS = [0, 1, 2]


def heart_sound_hr(i, q, starts, win):
    ph = np.unwrap(np.arctan2(q - np.median(q), i - np.median(i)))
    b, a = sg.butter(4, [16 / (FS_IN / 2), 80 / (FS_IN / 2)], "band")
    env = np.abs(sg.hilbert(sg.filtfilt(b, a, ph)))
    b2, a2 = sg.butter(3, 10 / (FS_IN / 2))
    env = sg.filtfilt(b2, a2, env)
    env = sg.decimate(env, 10, zero_phase=True)  # 200 Hz
    fs = FS_IN // 10
    out = np.full(len(starts), np.nan)
    lo, hi = int(0.33 * fs), int(1.5 * fs)
    for k, s in enumerate(starts):
        x = env[int(s * fs): int(s * fs) + win * fs]
        x = x - x.mean()
        ac = sg.correlate(x, x, mode="full", method="fft")[len(x) - 1:]
        ac /= ac[0] + 1e-12
        seg = ac[lo:hi]
        pk, _ = sg.find_peaks(seg)
        if len(pk) == 0:
            continue
        lag = lo + pk[np.argmax(seg[pk])]
        out[k] = 60 * fs / lag
    return out


def main():
    t0 = time.time()
    res = {}
    # ---- 1. heart-sound estimator against the ECG labels (labels from the cache written by gate_vitals.py)
    errs, per_subject = [], {}
    for p in sorted(DER.glob("*_resting.npz")):
        s = p.name[:7]
        d = np.load(p)
        f = next((RAW / s).glob(f"{s}_*_Resting.mat"))
        m = sio.loadmat(f)
        est = heart_sound_hr(m["radar_i"].ravel(), m["radar_q"].ravel(), d["starts"], WIN)
        ok = np.isfinite(d["hr"]) & np.isfinite(est)
        e = np.abs(est[ok] - d["hr"][ok])
        errs.append(e)
        per_subject[s] = {"mae": float(e.mean()), "within_3bpm": float((e <= 3).mean()), "n": int(ok.sum())}
        print(f"{s}: heart-sound MAE {e.mean():5.2f} bpm, within 3 bpm {100 * (e <= 3).mean():4.0f}%", flush=True)
    errs = np.concatenate(errs)
    res["heart_sound_estimator"] = {"mae_all": float(errs.mean()), "median_abs_err": float(np.median(errs)),
                                    "within_3bpm": float((errs <= 3).mean()), "per_subject": per_subject}
    print(f"heart-sound estimator: MAE {errs.mean():.2f} bpm, median |err| {np.median(errs):.2f}, "
          f"within 3 bpm {100 * (errs <= 3).mean():.0f}%  ({time.time() - t0:.0f}s)", flush=True)

    # ---- 2. subject-identity probe
    X, subj, tpos = [], [], []
    for p in sorted(DER.glob("*_resting.npz")):
        d = np.load(p)
        X.append(d["X"]); subj += [p.name[:7]] * len(d["X"]); tpos.append(d["starts"])
    X = np.concatenate(X); subj = np.array(subj); tpos = np.concatenate(tpos)
    names, y = np.unique(subj, return_inverse=True)
    n = len(y); idx = np.arange(n)

    def probe(tr, te, seed):
        torch.manual_seed(seed)
        m = Net().to(DEV)
        m.fc = torch.nn.Linear(256, len(names)).to(DEV)
        opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        xt = torch.as_tensor(X[tr], device=DEV); yt = torch.as_tensor(y[tr], device=DEV)
        xe = torch.as_tensor(X[te], device=DEV)
        for _ in range(EPOCHS):
            m.train()
            perm = torch.randperm(len(xt), device=DEV)
            for s in range(0, len(xt), 64):
                b = perm[s:s + 64]
                if len(b) < 2:
                    continue
                loss = F.cross_entropy(m(xt[b]), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        m.eval()
        with torch.no_grad():
            pred = torch.cat([m(xe[s:s + 512]).argmax(1) for s in range(0, len(xe), 512)]).cpu().numpy()
        return float((pred == y[te]).mean())

    rs = np.random.RandomState(0); perm = rs.permutation(n)
    te = np.zeros(n, bool); te[perm[: n // 5]] = True
    acc_r = [probe(~te, te, s) for s in SEEDS]
    te = np.zeros(n, bool)
    for k in range(len(names)):
        rows = idx[y == k]; rows = rows[np.argsort(tpos[y == k])]
        te[rows[int(0.8 * len(rows)):]] = True
    acc_t = [probe(~te, te, s) for s in SEEDS]
    res["subject_id_probe"] = {"random_windows_acc": float(np.mean(acc_r)), "temporal_8_2_acc": float(np.mean(acc_t)),
                               "chance": 1 / len(names), "seeds": {"random": acc_r, "temporal": acc_t}}
    print(f"subject-ID probe: random windows {100 * np.mean(acc_r):.1f}%, temporal 8:2 {100 * np.mean(acc_t):.1f}% "
          f"(chance {100 / len(names):.1f}%)  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(OUT / "gate2_vitals.json", "w"), indent=1)


if __name__ == "__main__":
    main()
