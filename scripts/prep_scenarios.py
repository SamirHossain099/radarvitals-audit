"""Window cache for every subject and scenario (Resting, Valsalva, Apnea, TiltUp, TiltDown).

Per 30-s window (stride 5 s): radar I/Q/phase at 100 Hz (as gate_vitals.py), ECG heart rate (lead 2, dropped where
lead 1 disagrees by > 2 bpm), window-median systolic and diastolic pressure (NaN where the pressure channel is
unusable), and the subject-independent Schmidt estimate with comb and pulse-prior lag selection (heart_sound_variants
"comb_prior"), computed from the raw 2000 Hz signal. Tilt scenarios are cut from the second button press (the upright
or horizontal position) to the end, as the authors' getInterventions.m does; Valsalva and Apnea keep the whole file.
Output: N:/Datasets/Schellenberger2020/derived/scen/<GDN>_<scenario>.npz. Run from the project folder.
"""

import numpy as np
import scipy.io as sio
import scipy.signal as sg

import bp_floor
import gate_vitals as gv
import heart_sound as hs

OUTDIR = gv.DER / "scen"
OUTDIR.mkdir(exist_ok=True)
SCEN = ["Resting", "Valsalva", "Apnea", "TiltUp", "TiltDown"]
FS = gv.FS_IN


def press_edges(iv):
    """Falling edges of the button channel (a press pulls the line low for ~0.3 s). None if the line never drops."""
    if iv.min() > 2:
        return np.array([], int)
    mid = (iv.min() + iv.max()) / 2
    d = np.diff((iv > mid).astype(int))
    return np.flatnonzero(d == -1)


def scenario_bounds(scenario, iv, fs_iv, n):
    if scenario not in ("TiltUp", "TiltDown"):
        return 0, n
    e = press_edges(iv)
    if len(e) >= 2:
        start = e[1]
    elif len(e) == 1:
        start = e[0]
    else:
        return 0, n
    return int(start * FS / fs_iv), n


def comb_prior_hr(i, q, starts, win):
    xi, yi = hs.ellipse_compensate(i - np.median(i), q - np.median(q))
    ph = np.unwrap(np.arctan2(yi, xi))
    bl, al = sg.butter(2, 80 / (FS / 2)); bh, ah = sg.butter(2, 16 / (FS / 2), "high")
    hsig = sg.filtfilt(bh, ah, sg.filtfilt(bl, al, ph))
    b, a = sg.butter(3, [0.7 / (FS / 2), 2.5 / (FS / 2)], "band")
    pl = sg.decimate(sg.filtfilt(b, a, ph), 20, zero_phase=True)
    f = np.fft.rfftfreq(win * 100 * 4, 1 / 100); band = (f >= 0.7) & (f <= 2.5)
    fs = 500; lo2, hi2 = int(0.40 * fs), int(1.60 * fs); lo, hi = int(0.45 * fs), int(1.45 * fs)
    out = np.full(len(starts), np.nan)
    for k, s in enumerate(starts):
        x = hs.spike_removal(hsig[int(s * FS): int(s * FS) + win * FS], FS)
        env = hs.homomorphic_envelope(x, FS)
        y = env - env.mean()
        ac = sg.correlate(y, y, mode="full", method="fft")[len(y) - 1:]
        ac = sg.decimate((ac / (ac[0] + 1e-12))[: int(3.2 * FS)], 4, zero_phase=True)
        seg = pl[int(s * 100): int(s * 100) + win * 100]
        P = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), win * 100 * 4))
        pulse = 60 * f[band][np.argmax(P[band])]
        pk, _ = sg.find_peaks(ac[lo2:hi2]); pk = pk + lo2
        if len(pk) == 0:
            out[k] = 60 * fs / (lo + int(np.argmax(ac[lo:hi]))); continue
        at = lambda idx: ac[idx] if idx < len(ac) else 0.0
        comb = np.array([at(p) + at(2 * p) - at(int(1.5 * p)) for p in pk])
        h_c = 60 * fs / pk[np.argmax(comb)]
        if abs(pulse - h_c) / h_c > 0.35:
            opts = [o for o in (h_c, h_c / 2, h_c * 2) if 40 <= o <= 150]
            h_c = min(opts, key=lambda o: abs(o - pulse)) if opts else h_c
        out[k] = h_c
    return out


def main():
    for d in sorted(p for p in gv.RAW.iterdir() if p.is_dir()):
        sid = d.name
        for scen in SCEN:
            files = list(d.glob(f"{sid}_*_{scen}.mat"))
            if not files:
                continue
            out = OUTDIR / f"{sid}_{scen}.npz"
            if out.exists():
                continue
            m = sio.loadmat(files[0])
            i, q = m["radar_i"].ravel(), m["radar_q"].ravel()
            s0, s1 = scenario_bounds(scen, m["tfm_intervention"].ravel(), int(m["fs_intervention"].ravel()[0]), len(i))
            i, q = i[s0:s1], q[s0:s1]
            # the authors fill missing ECG samples with 0 (fillmissing in TechnicalValidation.m); isolated NaNs occur
            # in both leads for GDN0009 Apnea. A lead that is NaN throughout (GDN0003 lead 1) stays excluded below.
            e2 = m["tfm_ecg2"].ravel()[s0:s1]; e1 = m["tfm_ecg1"].ravel()[s0:s1]
            if np.isfinite(e2).mean() > 0.99:
                e2 = np.nan_to_num(e2)
            if np.isfinite(e1).mean() > 0.99:
                e1 = np.nan_to_num(e1)
            dur = len(i) / FS
            starts = np.arange(0, dur - gv.WIN, gv.STRIDE)
            if len(starts) == 0:
                continue
            hr = gv.ecg_hr_per_window(e2, FS, starts, gv.WIN)
            if np.isfinite(e1).all():
                hr1 = gv.ecg_hr_per_window(e1, FS, starts, gv.WIN)
                both = np.isfinite(hr1) & np.isfinite(hr)
                hr[both & (np.abs(hr1 - hr) > 2)] = np.nan
            X = gv.radar_windows(i, q, starts, gv.WIN)
            est = comb_prior_hr(i, q, starts, gv.WIN)
            sbp = np.full(len(starts), np.nan); dbp = np.full(len(starts), np.nan)
            bp = m["tfm_bp"].ravel()
            b0, b1 = int(s0 * bp_floor.FS_BP / FS), int(s1 * bp_floor.FS_BP / FS)
            bp = bp[b0:b1]
            if np.isfinite(bp).all() and bp.std() > 1:
                ts, sv, td, dv = bp_floor.beat_pressures(bp, bp_floor.FS_BP)
                for k, w in enumerate(starts):
                    ms = (ts >= w) & (ts < w + gv.WIN); md = (td >= w) & (td < w + gv.WIN)
                    if ms.sum() >= 10 and md.sum() >= 10:
                        a, b_ = np.median(sv[ms]), np.median(dv[md])
                        if 70 < a < 220 and 30 < b_ < 130 and a > b_ + 15:
                            sbp[k], dbp[k] = a, b_
            np.savez_compressed(out, X=X, hr=hr, sbp=sbp, dbp=dbp, est=est, starts=starts, cut_start_s=s0 / FS)
            ok = np.isfinite(hr)
            print(f"{sid} {scen:<9} {dur:6.0f} s (cut {s0 / FS:5.1f} s) {len(starts):4d} windows, {ok.sum():4d} HR, "
                  f"{np.isfinite(sbp).sum():4d} BP; HR {np.nanmedian(hr):5.1f} (sd {np.nanstd(hr):4.1f}); Schmidt MAE "
                  f"{np.nanmean(np.abs(est[ok] - hr[ok])):5.2f}", flush=True)


if __name__ == "__main__":
    main()
