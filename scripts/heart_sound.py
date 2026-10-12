"""Subject-independent classical heart-rate estimators on Schellenberger 2020 (resting), per 30-s window.

A. The dataset authors' own pipeline, read from their saved validation output (code/scidata_phase1/
   output_validation.mat): radar heart-sound S1 locations from an HSMM (Springer 2016, Schmidt bands 16-80 Hz),
   inter-beat intervals median-filtered (5) and smoothed (6), sampled every second, alongside the ECG R-peak IBIs
   from the same run. Window HR = 60 / mean IBI over 30 s, stride 5 s, radar against ECG. No re-implementation,
   no tuning: this is what the authors' subject-independent processing achieves on their own data.
B. A Python re-implementation of the Schmidt autocorrelation heart-rate step with the authors' parameters: ellipse
   compensation of I/Q, arctangent phase, 16-80 Hz band, spike removal, homomorphic envelope (log-Hilbert, 8 Hz,
   exp), autocorrelation peak between 0.45 and 1.45 s. Evaluated on the gate's windows against the gate's ECG labels.
Writes results/heart_sound.json.
"""
import json

import numpy as np
import scipy.io as sio
import scipy.signal as sg

import gate_vitals as gv

RAW, DER, OUT = gv.RAW, gv.DER, gv.OUT
# the authors' validation output: clone https://gitlab.com/sven_schellenberger/scidata_phase1 to <root>/code/
VAL = gv.ROOT / "code" / "scidata_phase1" / "output_validation.mat"
FS = 2000
WIN, STRIDE = 30, 5


# ------------------------------------------------------------------ A. authors' pipeline
def authors_pipeline():
    m = sio.loadmat(VAL, squeeze_me=True, struct_as_record=False)["output"]
    per, errs = {}, []
    for sid in m._fieldnames:
        r = getattr(m, sid).Resting
        e, h = np.asarray(r.tfm_ecg_ibi, float).ravel(), np.asarray(r.radar_hs_ibi, float).ravel()
        n = len(e)
        ws = np.arange(0, n - WIN + 1, STRIDE)
        hr_e = np.array([60 / e[s:s + WIN].mean() for s in ws])
        hr_r = np.array([60 / h[s:s + WIN].mean() for s in ws])
        d = np.abs(hr_r - hr_e)
        per[sid] = {"n": int(len(ws)), "mae": float(d.mean()), "within_3bpm": float((d <= 3).mean()),
                    "ibi_rmse_ms": float(np.sqrt(np.mean((e - h) ** 2)) * 1000)}
        errs.append(d)
    errs = np.concatenate(errs)
    return {"mae": float(errs.mean()), "median_abs_err": float(np.median(errs)), "within_3bpm": float((errs <= 3).mean()),
            "n_windows": int(len(errs)), "per_subject": per}


# ------------------------------------------------------------------ B. Schmidt-style re-implementation
def ellipse_compensate(i, q):
    """Fit the I/Q trajectory to an ellipse (algebraic least squares on a decimated subset), map it to the unit circle."""
    x = sg.decimate(i, 20, zero_phase=True); y = sg.decimate(q, 20, zero_phase=True)
    D = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    S = D.T @ D
    C = np.zeros((6, 6)); C[0, 2] = C[2, 0] = 2; C[1, 1] = -1
    w, v = np.linalg.eig(np.linalg.solve(S + 1e-9 * np.eye(6), C))
    k = np.argmax(np.real(w))
    A, B, Cc, Dd, E, Fc = np.real(v[:, k])
    den = B * B - 4 * A * Cc
    x0 = (2 * Cc * Dd - B * E) / den; y0 = (2 * A * E - B * Dd) / den
    theta = 0.5 * np.arctan2(B, A - Cc)
    ct, st = np.cos(theta), np.sin(theta)
    # semi-axes
    up = 2 * (A * E * E + Cc * Dd * Dd + Fc * B * B - 2 * B * Dd * E - A * Cc * Fc) * (-1)
    a2 = up / (den * (np.sqrt((A - Cc) ** 2 + B * B) - (A + Cc)))
    b2 = up / (den * (-np.sqrt((A - Cc) ** 2 + B * B) - (A + Cc)))
    a, b = np.sqrt(abs(a2)), np.sqrt(abs(b2))
    xi = (i - x0) * ct + (q - y0) * st
    yi = -(i - x0) * st + (q - y0) * ct
    return xi / a, yi / b


def homomorphic_envelope(x, fs, lpf=8.0):
    b, a = sg.butter(1, lpf / (fs / 2))
    return np.exp(sg.filtfilt(b, a, np.log(np.abs(sg.hilbert(x)) + 1e-12)))


def spike_removal(x, fs):
    w = fs // 2
    n = len(x) // w * w
    frames = x[:n].reshape(-1, w).copy()
    maa = np.abs(frames).max(1)
    while (maa > np.median(maa) * 3).any():
        k = int(np.argmax(maa))
        f = frames[k]
        p = int(np.argmax(np.abs(f)))
        zc = np.flatnonzero(np.abs(np.diff(np.sign(f))) > 1)
        s = zc[zc < p].max() if (zc < p).any() else 0
        e = zc[zc > p].min() if (zc > p).any() else w - 1
        f[s:e + 1] = 1e-37
        maa[k] = np.abs(f).max()
    out = x.copy(); out[:n] = frames.ravel()
    return out


def schmidt_hr(i, q, starts, win, fs=FS):
    xi, yi = ellipse_compensate(i - np.median(i), q - np.median(q))
    ph = np.unwrap(np.arctan2(yi, xi))
    bl, al = sg.butter(2, 80 / (fs / 2)); bh, ah = sg.butter(2, 16 / (fs / 2), "high")
    hs = sg.filtfilt(bh, ah, sg.filtfilt(bl, al, ph))
    out = np.full(len(starts), np.nan)
    lo, hi = int(0.45 * fs), int(1.45 * fs)
    for k, s in enumerate(starts):
        x = hs[int(s * fs): int(s * fs) + win * fs]
        x = spike_removal(x, fs)
        env = homomorphic_envelope(x, fs)
        y = env - env.mean()
        ac = sg.correlate(y, y, mode="full", method="fft")[len(y) - 1:]
        ac = ac / (ac[0] + 1e-12)
        out[k] = 60 * fs / (lo + int(np.argmax(ac[lo:hi])))
    return out


def reimplementation():
    per, errs = {}, []
    for p in sorted(DER.glob("*_resting.npz")):
        sid = p.name[:7]
        d = np.load(p)
        m = sio.loadmat(next((RAW / sid).glob(f"{sid}_*_Resting.mat")))
        est = schmidt_hr(m["radar_i"].ravel(), m["radar_q"].ravel(), d["starts"], WIN)
        ok = np.isfinite(d["hr"]) & np.isfinite(est)
        e = np.abs(est[ok] - d["hr"][ok])
        per[sid] = {"n": int(ok.sum()), "mae": float(e.mean()), "within_3bpm": float((e <= 3).mean())}
        errs.append(e)
        print(f"{sid}: Schmidt MAE {e.mean():5.2f} bpm, within 3 bpm {100 * (e <= 3).mean():4.0f}%", flush=True)
    errs = np.concatenate(errs)
    return {"mae": float(errs.mean()), "median_abs_err": float(np.median(errs)), "within_3bpm": float((errs <= 3).mean()),
            "n_windows": int(len(errs)), "per_subject": per}


def main():
    A = authors_pipeline()
    print(f"authors' HSMM pipeline (from output_validation.mat): MAE {A['mae']:.2f} bpm, median |err| {A['median_abs_err']:.2f}, "
          f"within 3 bpm {100 * A['within_3bpm']:.0f}%, {A['n_windows']} windows", flush=True)
    worst = sorted(A["per_subject"].items(), key=lambda kv: -kv[1]["mae"])[:5]
    print("  worst subjects:", ", ".join(f"{k} {v['mae']:.1f} bpm (IBI RMSE {v['ibi_rmse_ms']:.0f} ms)" for k, v in worst))
    B = reimplementation()
    print(f"Schmidt re-implementation: MAE {B['mae']:.2f} bpm, median |err| {B['median_abs_err']:.2f}, within 3 bpm {100 * B['within_3bpm']:.0f}%")
    json.dump({"authors_pipeline": A, "schmidt_reimplementation": B}, open(OUT / "heart_sound.json", "w"), indent=1)


if __name__ == "__main__":
    main()
