"""Every number the paper quotes, from results/ (and the label caches) into results/manuscript_numbers.json
(standing rule 4).

    python manuscript_numbers.py && python render_manuscript.py
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
SCEN = Path(os.environ.get("SCHELLENBERGER_ROOT", r"N:\Datasets\Schellenberger2020")) / "derived" / "scen"

# Published values, each from the paper's own text (references-notes.md)
PUB = {"liu_sbp": 3.21, "liu_dbp": 2.25, "dual_sbp": 5.00, "dual_dbp": 3.96, "tridp": 4.73,
       "maru_sbp": 4.05, "maru_dbp": 3.08, "wang_sbp": 6.52, "wang_dbp": 3.98, "kundu_hr": 0.63}


def j(n):
    return json.loads((R / n).read_text(encoding="utf-8"))


def f1(v):
    return f"{v:.1f}"


def f2(v):
    return f"{v:.2f}"


def main():
    gv, g2, hs, rules = j("gate_vitals.json"), j("gate2_vitals.json"), j("heart_sound.json"), j("hs_rules_scen.json")
    bp, ev, rep, demo = j("gate_bp.json"), j("eval_scenarios.json"), j("replicate_protocols.json"), j("demographic_floor.json")
    N = {}

    def put(k, v, text=None):
        N[k] = {"value": v, "text": text if text is not None else f2(v)}

    # ---- label statistics from the caches (resting)
    hr, sbp, dbp, subj = [], [], [], []
    for p in sorted(SCEN.glob("GDN*_Resting.npz")):
        d = np.load(p)
        hr.append(d["hr"]); sbp.append(d["sbp"]); dbp.append(d["dbp"]); subj += [p.name[:7]] * len(d["hr"])
    hr, sbp, dbp, subj = np.concatenate(hr), np.concatenate(sbp), np.concatenate(dbp), np.array(subj)
    ok = np.isfinite(hr)
    put("rest_n", int(ok.sum()), f"{int(ok.sum()):,}")
    put("rest_within_sd", float(np.mean([hr[ok & (subj == s)].std() for s in np.unique(subj)])), None)
    N["rest_within_sd"]["text"] = f1(N["rest_within_sd"]["value"])
    put("rest_across_sd", float(np.std([hr[ok & (subj == s)].mean() for s in np.unique(subj)])), None)
    N["rest_across_sd"]["text"] = f1(N["rest_across_sd"]["value"])
    okb = np.isfinite(sbp)
    put("rest_within_sd_sbp", float(np.mean([sbp[okb & (subj == s)].std() for s in np.unique(subj[okb])])), None)
    N["rest_within_sd_sbp"]["text"] = f1(N["rest_within_sd_sbp"]["value"])
    put("rest_across_sd_sbp", float(np.std([sbp[okb & (subj == s)].mean() for s in np.unique(subj[okb])])), None)
    N["rest_across_sd_sbp"]["text"] = f1(N["rest_across_sd_sbp"]["value"])
    put("bp_n", int(okb.sum()), f"{int(okb.sum()):,}")
    put("bp_n_subj", int(len(np.unique(subj[okb]))), str(int(len(np.unique(subj[okb])))))

    # label validation numbers, from the gate log (lead agreement and TFM comparison were printed, not stored)
    agree, tfm = [], []
    for line in (R / "gate_vitals.log").read_text(encoding="utf-8").splitlines():
        if "lead agreement" in line:
            a = line.split("lead agreement ")[1].split(",")[0].strip()
            if a not in ("None",) and float(a) > 0:  # GDN0001's lead 1 is unusable; its 0.0 means no comparable windows
                agree.append(float(a))
            if "TFM beat HR median" in line:
                tfm_v = float(line.split("TFM beat HR median ")[1]); ours = float(line.split("HR median ")[1].split(",")[0])
                tfm.append(abs(tfm_v - ours))
    put("lead_agree_min", min(agree) * 100, f"{100 * min(agree):.0f}")
    put("tfm_n", len(tfm), str(len(tfm)))
    put("tfm_agree_max", max(tfm), f1(max(tfm)))

    # ---- Table 1: resting HR
    P = gv["protocols"]
    for key, name in (("rand", "random_windows_80_20"), ("temp", "within_subject_temporal_8_2"), ("disj", "subject_disjoint_5fold")):
        for m, col in (("cnn", "cnn"), ("subj", "subj_median"), ("pop", "pop_median"), ("fft", "fft")):
            put(f"hr_{key}_{m}", P[name][col])
    put("demo_hr", demo["hr"]["ridge_demographics_mae"])
    put("auth_mae", hs["authors_pipeline"]["mae"]); put("auth_med", hs["authors_pipeline"]["median_abs_err"])
    put("auth_within3", 100 * hs["authors_pipeline"]["within_3bpm"], f"{100 * hs['authors_pipeline']['within_3bpm']:.0f}")
    put("probe_rand", 100 * g2["subject_id_probe"]["random_windows_acc"], f"{100 * g2['subject_id_probe']['random_windows_acc']:.1f}")
    put("probe_temp", 100 * g2["subject_id_probe"]["temporal_8_2_acc"], f"{100 * g2['subject_id_probe']['temporal_8_2_acc']:.1f}")

    # ---- Table 2: resting BP
    B = bp["protocols"]
    for key, name in (("rand", "random_windows_80_20"), ("temp", "within_subject_temporal_8_2"), ("disj", "subject_disjoint_5fold")):
        for m, col in (("cnn", "cnn"), ("subj", "subj"), ("pop", "pop")):
            put(f"bp_{key}_{m}_sbp", B[name][f"{col}_sbp"]); put(f"bp_{key}_{m}_dbp", B[name][f"{col}_dbp"])
    put("demo_sbp", demo["sbp"]["ridge_demographics_mae"]); put("demo_dbp", demo["dbp"]["ridge_demographics_mae"])
    for k, v in PUB.items():
        put(f"pub_{k}", v)

    # ---- Table 3: replications
    L, LD, T = rep["Liu2026_5.12s_shuffled_60_20_20"], rep["Liu2026_5.12s_subject_disjoint_5fold"], rep["TriDP_8.192s_per_subject_8_2"]
    for pre, src in (("liu", L), ("tridp", T)):
        for m, col in (("cnn", "cnn"), ("subj", "subj_median"), ("pop", "pop_median")):
            put(f"{pre}_{m}_hr", src["hr"][col][0]); put(f"{pre}_{m}_sbp", src["bp"][col][0]); put(f"{pre}_{m}_dbp", src["bp"][col][1])
    for m, col in (("cnn", "cnn"), ("pop", "pop_median")):
        put(f"liu_disj_{m}_hr", LD["hr"][col][0]); put(f"liu_disj_{m}_sbp", LD["bp"][col][0]); put(f"liu_disj_{m}_dbp", LD["bp"][col][1])
    put("liu_ntrain", L["bp"]["n_train"], f"{L['bp']['n_train']:,}")
    put("tridp_hr_gain", T["hr"]["subj_median"][0] - T["hr"]["cnn"][0], f1(T["hr"]["subj_median"][0] - T["hr"]["cnn"][0]))
    floors = [L["bp"]["subj_median"][0], T["bp"]["subj_median"][0], B["random_windows_80_20"]["subj_sbp"], B["within_subject_temporal_8_2"]["subj_sbp"]]
    put("bp_floor_sbp_min", min(floors), f1(min(floors))); put("bp_floor_sbp_max", max(floors), f1(max(floors)))

    # ---- Table 4: scenarios
    keys = {"Resting": "rest", "Valsalva": "vals", "Apnea": "apn", "TiltUp": "tup", "TiltDown": "tdn", "all": "all"}
    hsname = {"Resting": "Resting", "Valsalva": "Valsalva", "Apnea": "Apnea", "TiltUp": "TiltUp", "TiltDown": "TiltDown"}
    for scen, k in keys.items():
        e = ev[scen]
        st = e["hr_stats"]
        if k != "rest":
            put(f"{k}_n", st["n"], f"{st['n']:,}")
            put(f"{k}_within_sd", st["within_subject_sd"], f1(st["within_subject_sd"]))
            put(f"{k}_across_sd", st["across_subject_sd"], f1(st["across_subject_sd"]))
        for key, name in (("rand", "random_windows_80_20"), ("temp", "within_subject_temporal_8_2"), ("disj", "subject_disjoint_5fold")):
            if k != "rest":
                put(f"{k}_{key}_cnn", e["hr"][name]["cnn"]); put(f"{k}_{key}_subj", e["hr"][name]["subj_median"]); put(f"{k}_{key}_pop", e["hr"][name]["pop_median"])
        if scen == "all":
            put("hs_all", rules["all"]["comb_prior"]["mae"], f1(rules["all"]["comb_prior"]["mae"]))
            put("all_disj_cnn_sbp", e["bp"]["subject_disjoint_5fold"]["cnn_sbp"]); put("all_disj_cnn_dbp", e["bp"]["subject_disjoint_5fold"]["cnn_dbp"])
            put("all_disj_pop_sbp", e["bp"]["subject_disjoint_5fold"]["pop_sbp"]); put("all_disj_pop_dbp", e["bp"]["subject_disjoint_5fold"]["pop_dbp"])
        else:
            v = rules[hsname[scen]]["comb_prior"]["mae"]
            put(f"hs_{k}", v, f1(v))
    # tilt-up half-rate share (from the est/hr ratio of the cached estimates)
    r = []
    for p in sorted(SCEN.glob("GDN*_TiltUp.npz")):
        d = np.load(p); okk = np.isfinite(d["hr"]) & np.isfinite(d["est"]); r.append(d["est"][okk] / d["hr"][okk])
    r = np.concatenate(r)
    put("tup_half_share", 100 * float(((r > 0.4) & (r < 0.6)).mean()), f"{100 * ((r > 0.4) & (r < 0.6)).mean():.0f}")

    ps = j("per_subject_temporal.json")
    nb = sum(v["subj_median_mae"] < v["cnn_mae"] for v in ps.values())
    put("lookup_better_n", nb, str(nb))

    # ---- added after the adversarial review of 2026-10-12
    for key, name in (("rand", "random_windows_80_20"), ("temp", "within_subject_temporal_8_2"), ("disj", "subject_disjoint_5fold")):
        put(f"hr_{key}_knn", P[name]["knn_spec"])
    fo = P["subject_disjoint_5fold"]["folds_cnn"]
    put("hr_disj_fold_min", min(fo)); put("hr_disj_fold_max", max(fo))
    put("auth_n", hs["authors_pipeline"]["n_windows"], f"{hs['authors_pipeline']['n_windows']:,}")
    # test-set sizes: random = n // 5; temporal = sum over subjects of the last 20% of their windows
    n_hr = int(ok.sum()); put("hr_rand_nte", n_hr // 5, f"{n_hr // 5:,}")
    te = 0
    for s in np.unique(subj):
        m = int((ok & (subj == s)).sum()); te += m - int(0.8 * m)
    put("hr_temp_nte", te, f"{te:,}")
    n_bp = int(okb.sum()); put("bp_rand_nte", n_bp // 5, f"{n_bp // 5:,}")
    te = 0
    for s in np.unique(subj[okb]):
        m = int((okb & (subj == s)).sum()); te += m - int(0.8 * m)
    put("bp_temp_nte", te, f"{te:,}")
    put("liu_gap_sbp", L["bp"]["subj_median"][0] - PUB["liu_sbp"], f1(L["bp"]["subj_median"][0] - PUB["liu_sbp"]))
    put("liu_gap_dbp", L["bp"]["subj_median"][1] - PUB["liu_dbp"], f1(L["bp"]["subj_median"][1] - PUB["liu_dbp"]))
    D = rep["DualStage2025_2s_shuffled_65_15_20"]
    for m, col in (("cnn", "cnn"), ("subj", "subj_median"), ("pop", "pop_median")):
        put(f"dual_{m}_hr", D["hr"][col][0]); put(f"dual_{m}_sbp", D["bp"][col][0]); put(f"dual_{m}_dbp", D["bp"][col][1])
    # Table 4's resting row from the scenario run (same training run as the other rows)
    e = ev["Resting"]
    for key, name in (("rand", "random_windows_80_20"), ("temp", "within_subject_temporal_8_2"), ("disj", "subject_disjoint_5fold")):
        put(f"rest2_{key}_cnn", e["hr"][name]["cnn"]); put(f"rest2_{key}_subj", e["hr"][name]["subj_median"]); put(f"rest2_{key}_pop", e["hr"][name]["pop_median"])
    pb = j("pooled_breakdown.json")
    for scen, k in (("Resting", "rest"), ("Valsalva", "vals"), ("Apnea", "apn"), ("TiltUp", "tup"), ("TiltDown", "tdn")):
        put(f"pb_{k}", pb[scen]["cnn"]); put(f"pb_sm_{k}", pb[scen]["scen_median"])
    put("pb_scen_median_all", pb["all"]["scen_median"])
    put("pb_all", pb["all"]["cnn"])
    sm = j("pooled_sizematched.json")
    put("sm_cnn", sm["cnn"]); put("sm_pop", sm["pop_median"]); put("sm_n", sm["folds"][0]["n_train_sub"], f"{sm['folds'][0]['n_train_sub']:,}")
    put("hs_tup_comb", rules["TiltUp"]["comb"]["mae"], f1(rules["TiltUp"]["comb"]["mae"]))

    # ---- artifact placeholders until the release exists
    put("artifact_url", "https://github.com/SamirHossain099/radarvitals-audit", "https://github.com/SamirHossain099/radarvitals-audit")
    # concept DOI of the Zenodo archive (v0.1.0 = 10.5281/zenodo.23321446, released 2026-10-12), pinned against CITATION.cff
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    doi = cff.split('doi: "')[1].split('"')[0]
    put("artifact_doi", doi, f"https://doi.org/{doi}")

    (R / "manuscript_numbers.json").write_text(json.dumps(N, indent=1), encoding="utf-8")
    print(f"{len(N)} numbers written")


if __name__ == "__main__":
    main()
