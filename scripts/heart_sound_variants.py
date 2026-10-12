"""Octave-error fixes for the Schmidt re-implementation (heart_sound.py B). Caches the per-window heart-sound
envelope autocorrelation once, then scores lag-selection rules:
  schmidt     global max of ac in 0.45-1.45 s (the original rule);
  comb        among local maxima in 0.40-1.60 s, maximise ac[L] + ac[2L] - ac[1.5L] (multiples high, half-multiples low);
  pulse_prior among local maxima, pick the one nearest the Doppler-pulse spectral estimate (0.7-2.5 Hz band of the
              phase) when that estimate is within 35% of a candidate, else the Schmidt choice;
  comb+prior  comb score with the pulse estimate breaking factor-of-two ties.
All rules are subject-independent and have no trained parameters. Writes results/heart_sound_variants.json.
"""
import json

import numpy as np
import scipy.io as sio
import scipy.signal as sg

import heart_sound as hs

OUT = hs.OUT
CACHE = hs.DER / "hs_autocorr.npz"
FS = hs.FS


def build_cache():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=False)
        return d["ac"], d["hr"], d["subj"], d["pulse"]
    acs, hrs, subj, pulse = [], [], [], []
    for p in sorted(hs.DER.glob("*_resting.npz")):
        sid = p.name[:7]
        d = np.load(p)
        m = sio.loadmat(next((hs.RAW / sid).glob(f"{sid}_*_Resting.mat")))
        i, q = m["radar_i"].ravel(), m["radar_q"].ravel()
        xi, yi = hs.ellipse_compensate(i - np.median(i), q - np.median(q))
        ph = np.unwrap(np.arctan2(yi, xi))
        bl, al = sg.butter(2, 80 / (FS / 2)); bh, ah = sg.butter(2, 16 / (FS / 2), "high")
        hsig = sg.filtfilt(bh, ah, sg.filtfilt(bl, al, ph))
        # Doppler pulse band on the phase, decimated to 100 Hz
        b, a = sg.butter(3, [0.7 / (FS / 2), 2.5 / (FS / 2)], "band")
        pl = sg.decimate(sg.filtfilt(b, a, ph), 20, zero_phase=True)
        f = np.fft.rfftfreq(hs.WIN * 100 * 4, 1 / 100); band = (f >= 0.7) & (f <= 2.5)
        for k, s in enumerate(d["starts"]):
            if not np.isfinite(d["hr"][k]):
                continue
            x = hs.spike_removal(hsig[int(s * FS): int(s * FS) + hs.WIN * FS], FS)
            env = hs.homomorphic_envelope(x, FS)
            y = env - env.mean()
            ac = sg.correlate(y, y, mode="full", method="fft")[len(y) - 1:]
            ac = (ac / (ac[0] + 1e-12))[: int(3.2 * FS)]
            acs.append(sg.decimate(ac, 4, zero_phase=True).astype(np.float32))  # 500 Hz lag resolution
            seg = pl[int(s * 100): int(s * 100) + hs.WIN * 100]
            P = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), hs.WIN * 100 * 4))
            pulse.append(60 * f[band][np.argmax(P[band])])
            hrs.append(d["hr"][k]); subj.append(sid)
        print(sid, "cached", flush=True)
    ac = np.stack(acs); hr = np.array(hrs); subj = np.array(subj); pulse = np.array(pulse)
    np.savez_compressed(CACHE, ac=ac, hr=hr, subj=subj, pulse=pulse)
    return ac, hr, subj, pulse


def main():
    ac, hr, subj, pulse = build_cache()
    fs = 500
    lo, hi = int(0.45 * fs), int(1.45 * fs)
    lo2, hi2 = int(0.40 * fs), int(1.60 * fs)
    est = {k: np.zeros(len(hr)) for k in ["schmidt", "comb", "pulse_prior", "comb_prior"]}
    for n in range(len(hr)):
        a = ac[n]
        L = lo + int(np.argmax(a[lo:hi]))
        est["schmidt"][n] = 60 * fs / L
        pk, _ = sg.find_peaks(a[lo2:hi2]); pk = pk + lo2
        if len(pk) == 0:
            for k in ["comb", "pulse_prior", "comb_prior"]:
                est[k][n] = est["schmidt"][n]
            continue
        def at(idx):
            return a[idx] if idx < len(a) else 0.0
        comb = np.array([at(p) + at(2 * p) - at(int(1.5 * p)) for p in pk])
        Lc = pk[np.argmax(comb)]
        est["comb"][n] = 60 * fs / Lc
        cand_hr = 60 * fs / pk
        rel = np.abs(cand_hr - pulse[n]) / pulse[n]
        if rel.min() < 0.35:
            est["pulse_prior"][n] = cand_hr[np.argmin(rel)]
        else:
            est["pulse_prior"][n] = est["schmidt"][n]
        # comb with the pulse estimate deciding between the comb choice and its octave partners
        h_c = 60 * fs / Lc
        options = [h_c, h_c / 2, h_c * 2]
        options = [o for o in options if 40 <= o <= 150]
        est["comb_prior"][n] = min(options, key=lambda o: abs(o - pulse[n])) if abs(pulse[n] - h_c) / h_c > 0.35 else h_c
    res = {"n_windows": int(len(hr)), "pulse_fft": {"mae": float(np.abs(pulse - hr).mean()),
                                                   "within_25pct": float((np.abs(pulse - hr) / hr < 0.25).mean())}}
    print(f"{len(hr)} windows; pulse FFT MAE {res['pulse_fft']['mae']:.2f} bpm, within 25% of truth {100 * res['pulse_fft']['within_25pct']:.0f}%")
    for k, e in est.items():
        d = np.abs(e - hr)
        per = {s: float(d[subj == s].mean()) for s in np.unique(subj)}
        res[k] = {"mae": float(d.mean()), "median_abs_err": float(np.median(d)), "within_3bpm": float((d <= 3).mean()),
                  "subjects_over_5bpm": [s for s, v in per.items() if v > 5], "per_subject_mae": per}
        print(f"{k:<12} MAE {d.mean():5.2f}  median {np.median(d):4.2f}  within 3 bpm {100 * (d <= 3).mean():3.0f}%  "
              f"subjects > 5 bpm: {res[k]['subjects_over_5bpm']}")
    json.dump(res, open(OUT / "heart_sound_variants.json", "w"), indent=1)


if __name__ == "__main__":
    main()
