"""Cache the heart-sound envelope autocorrelation (500 Hz lag resolution, 0-3.2 s) and the Doppler-pulse spectral
estimate for every window of every subject and scenario, so lag-selection rules can be compared without touching
the raw files again. Windows and cuts follow prep_scenarios.py. Output: derived/scen/hs_ac_<scenario>.npz with
ac (n, 1600), pulse (n,), hr (n,), subj (n,), start (n,).
"""
import numpy as np
import scipy.io as sio
import scipy.signal as sg

import gate_vitals as gv
import heart_sound as hs
from prep_scenarios import OUTDIR, SCEN, scenario_bounds

FS = gv.FS_IN


def main():
    for scen in SCEN:
        out = OUTDIR / f"hs_ac_{scen}.npz"
        if out.exists():
            continue
        acs, pulses, hrs, subj, starts_all = [], [], [], [], []
        for p in sorted(OUTDIR.glob(f"GDN*_{scen}.npz")):
            sid = p.name[:7]
            d = np.load(p)
            m = sio.loadmat(next((gv.RAW / sid).glob(f"{sid}_*_{scen}.mat")))
            i, q = m["radar_i"].ravel(), m["radar_q"].ravel()
            s0, s1 = scenario_bounds(scen, m["tfm_intervention"].ravel(), int(m["fs_intervention"].ravel()[0]), len(i))
            i, q = i[s0:s1], q[s0:s1]
            xi, yi = hs.ellipse_compensate(i - np.median(i), q - np.median(q))
            ph = np.unwrap(np.arctan2(yi, xi))
            bl, al = sg.butter(2, 80 / (FS / 2)); bh, ah = sg.butter(2, 16 / (FS / 2), "high")
            hsig = sg.filtfilt(bh, ah, sg.filtfilt(bl, al, ph))
            b, a = sg.butter(3, [0.7 / (FS / 2), 2.5 / (FS / 2)], "band")
            pl = sg.decimate(sg.filtfilt(b, a, ph), 20, zero_phase=True)
            f = np.fft.rfftfreq(gv.WIN * 100 * 4, 1 / 100); band = (f >= 0.7) & (f <= 2.5)
            for k, s in enumerate(d["starts"]):
                x = hs.spike_removal(hsig[int(s * FS): int(s * FS) + gv.WIN * FS], FS)
                env = hs.homomorphic_envelope(x, FS)
                y = env - env.mean()
                ac = sg.correlate(y, y, mode="full", method="fft")[len(y) - 1:]
                ac = sg.decimate((ac / (ac[0] + 1e-12))[: int(3.2 * FS)], 4, zero_phase=True).astype(np.float32)
                seg = pl[int(s * 100): int(s * 100) + gv.WIN * 100]
                P = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), gv.WIN * 100 * 4))
                acs.append(ac); pulses.append(60 * f[band][np.argmax(P[band])]); hrs.append(d["hr"][k]); subj.append(sid); starts_all.append(s)
            print(scen, sid, flush=True)
        np.savez_compressed(out, ac=np.stack(acs), pulse=np.array(pulses), hr=np.array(hrs), subj=np.array(subj), start=np.array(starts_all))


if __name__ == "__main__":
    main()
