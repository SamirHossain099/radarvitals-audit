"""Lag-selection rules for the heart-sound estimator, scored per scenario from the hs_ac_<scenario> caches.

Rules (all subject-independent, no trained parameters):
  schmidt      global autocorrelation maximum in 0.45-1.45 s (Schmidt 2010 / Springer 2016, the authors' setting)
  comb         local maxima in 0.40-1.60 s, maximise ac[L] + ac[2L] - ac[1.5L]
  comb_prior   comb, with the Doppler-pulse estimate deciding factor-of-two alternatives when it disagrees by > 35%
  comb_small   among local maxima whose comb score is within 80% of the best, take the smallest lag (anti-subharmonic)
  comb_small_prior  comb_small with the pulse tie-break
  shs          sub-harmonic summation on the envelope spectrum, 40-150 bpm, three harmonics weighted 1, 0.8, 0.64
Writes results/hs_rules_scen.json. The rule chosen for the paper is picked here on all scenarios and must be
reported as such.
"""
import json

import numpy as np
import scipy.signal as sg

import gate_vitals as gv
from prep_scenarios import OUTDIR, SCEN

FS = 500
LO, HI = int(0.45 * FS), int(1.45 * FS)
LO2, HI2 = int(0.40 * FS), int(1.60 * FS)


def rules_for(ac, pulse):
    out = {}
    L = LO + int(np.argmax(ac[LO:HI]))
    out["schmidt"] = 60 * FS / L
    pk, _ = sg.find_peaks(ac[LO2:HI2]); pk = pk + LO2
    if len(pk) == 0:
        for k in ["comb", "comb_prior", "comb_small", "comb_small_prior"]:
            out[k] = out["schmidt"]
    else:
        at = lambda i: ac[i] if i < len(ac) else 0.0
        comb = np.array([at(p) + at(2 * p) - at(int(1.5 * p)) for p in pk])
        best = pk[np.argmax(comb)]
        out["comb"] = 60 * FS / best
        small = pk[comb >= comb.max() - 0.2 * abs(comb.max())].min()  # always contains the argmax, also when negative
        out["comb_small"] = 60 * FS / small
        for k, h in (("comb_prior", out["comb"]), ("comb_small_prior", out["comb_small"])):
            if abs(pulse - h) / h > 0.35:
                opts = [o for o in (h, h / 2, h * 2) if 40 <= o <= 150]
                h = min(opts, key=lambda o: abs(o - pulse)) if opts else h
            out[k] = h
    # sub-harmonic summation on the envelope spectrum (ac is the envelope autocorrelation; its FFT is the power spectrum)
    n = 4096
    spec = np.abs(np.fft.rfft(ac * np.hanning(len(ac)), n))
    f = np.fft.rfftfreq(n, 1 / FS)
    cands = np.arange(40, 150.5, 0.5) / 60
    score = np.zeros(len(cands))
    for h, w in ((1, 1.0), (2, 0.8), (3, 0.64)):
        score += w * np.interp(cands * h, f, spec)
    out["shs"] = 60 * cands[np.argmax(score)]
    return out


def main():
    res = {}
    allerr = {}
    for scen in SCEN:
        d = np.load(OUTDIR / f"hs_ac_{scen}.npz")
        ok = np.isfinite(d["hr"])
        ac, pulse, hr, subj = d["ac"][ok], d["pulse"][ok], d["hr"][ok], d["subj"][ok]
        est = {}
        for n in range(len(hr)):
            for k, v in rules_for(ac[n], pulse[n]).items():
                est.setdefault(k, []).append(v)
        res[scen] = {"n": int(len(hr))}
        line = f"[{scen:<9}] n={len(hr):5d} "
        for k, v in est.items():
            e = np.abs(np.array(v) - hr)
            allerr.setdefault(k, []).append(e)
            per = {s: float(e[subj == s].mean()) for s in np.unique(subj)}
            res[scen][k] = {"mae": float(e.mean()), "median": float(np.median(e)), "within_3bpm": float((e <= 3).mean()),
                            "subjects_over_5bpm": int(sum(v > 5 for v in per.values()))}
            line += f" {k} {e.mean():5.2f}/{100 * (e <= 3).mean():3.0f}%/{res[scen][k]['subjects_over_5bpm']:2d}"
        print(line, flush=True)
    res["all"] = {}
    line = "[all      ]          "
    for k, es in allerr.items():
        e = np.concatenate(es)
        res["all"][k] = {"mae": float(e.mean()), "median": float(np.median(e)), "within_3bpm": float((e <= 3).mean())}
        line += f" {k} {e.mean():5.2f}/{100 * (e <= 3).mean():3.0f}%"
    print(line)
    print("format: MAE bpm / within 3 bpm / subjects with MAE > 5 bpm")
    json.dump(res, open(gv.OUT / "hs_rules_scen.json", "w"), indent=1)


if __name__ == "__main__":
    main()
