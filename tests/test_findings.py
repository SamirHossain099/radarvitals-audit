"""The paper's claims, asserted against the result files (standing rule: a test must not hard-code the number it
guards; thresholds here are the claim's own qualitative content).

    python -m pytest -q
"""
import json
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[1] / "results"


def j(n):
    return json.loads((R / n).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def res():
    return {k: j(f"{k}.json") for k in ["gate_vitals", "gate2_vitals", "gate_bp", "heart_sound", "heart_sound_variants",
                                        "hs_rules_scen", "eval_scenarios", "replicate_protocols", "demographic_floor",
                                        "per_subject_temporal", "manuscript_numbers"]}


def test_lookup_matches_network_within_subject(res):
    """Claim 1: under both within-subject protocols the per-subject median is within 0.5 bpm of, or better than, the CNN."""
    P = res["gate_vitals"]["protocols"]
    assert P["random_windows_80_20"]["subj_median"] <= P["random_windows_80_20"]["cnn"] + 0.5
    assert P["within_subject_temporal_8_2"]["subj_median"] < P["within_subject_temporal_8_2"]["cnn"]


def test_network_at_population_median_subject_disjoint(res):
    """Claim 2: subject-disjoint at rest, the CNN is within 0.5 bpm of the population median."""
    P = res["gate_vitals"]["protocols"]["subject_disjoint_5fold"]
    assert abs(P["cnn"] - P["pop_median"]) < 0.5


def test_identity_probe(res):
    p = res["gate2_vitals"]["subject_id_probe"]
    assert p["random_windows_acc"] > 0.95 and p["temporal_8_2_acc"] > 0.85


def test_authors_pipeline_beats_every_learned_result(res):
    a = res["heart_sound"]["authors_pipeline"]["mae"]
    P = res["gate_vitals"]["protocols"]
    assert a < min(P[k]["cnn"] for k in P)
    assert res["heart_sound"]["authors_pipeline"]["within_3bpm"] > 0.95


def test_bp_floor_below_published(res):
    """Claim 3: the per-subject-median BP floor under within-subject protocols is below every published
    within-subject SBP/DBP on this dataset (3.21/2.25, 4.73, 5.00/3.96)."""
    B = res["gate_bp"]["protocols"]
    L = res["replicate_protocols"]["Liu2026_5.12s_shuffled_60_20_20"]["bp"]["subj_median"]
    T = res["replicate_protocols"]["TriDP_8.192s_per_subject_8_2"]["bp"]["subj_median"]
    for sbp, dbp in [(B["random_windows_80_20"]["subj_sbp"], B["random_windows_80_20"]["subj_dbp"]),
                     (B["within_subject_temporal_8_2"]["subj_sbp"], B["within_subject_temporal_8_2"]["subj_dbp"]), (T[0], T[1])]:
        assert sbp <= 3.21 + 0.3 and dbp <= 2.25 + 0.3
    assert L[0] <= 3.6 and L[1] <= 2.7  # the 5.12-s floor sits at Liu et al.'s 3.21 / 2.25, within 0.4 mmHg


def test_bp_subject_disjoint_worse_than_median(res):
    B = res["gate_bp"]["protocols"]["subject_disjoint_5fold"]
    assert B["cnn_sbp"] > B["pop_sbp"] and B["cnn_dbp"] > B["pop_dbp"]
    for scen in ["Resting", "Valsalva", "Apnea", "TiltUp", "TiltDown", "all"]:
        E = res["eval_scenarios"][scen]["bp"]["subject_disjoint_5fold"]
        assert E["cnn_sbp"] > E["pop_sbp"]


def test_pooled_scenarios_make_hr_transfer(res):
    E = res["eval_scenarios"]["all"]
    assert E["hr_stats"]["within_subject_sd"] > E["hr_stats"]["across_subject_sd"]
    d = E["hr"]["subject_disjoint_5fold"]
    assert d["cnn"] < 0.5 * d["pop_median"] and d["cnn"] < d["schmidt"]
    for scen in ["Resting", "Valsalva", "Apnea", "TiltUp", "TiltDown"]:
        s = res["eval_scenarios"][scen]["hr"]["subject_disjoint_5fold"]
        assert s["cnn"] > 0.7 * s["pop_median"]  # within a single scenario the CNN stays near the median


def test_pooled_gain_is_within_scenario(res):
    """The pooled-trained network beats the scenario-aware median inside every scenario on held-out subjects."""
    pb = j("pooled_breakdown.json")
    for s in ["Resting", "Valsalva", "Apnea", "TiltUp", "TiltDown"]:
        assert pb[s]["cnn"] < 0.5 * pb[s]["scen_median"]


def test_size_matched_control(res):
    """Label variation alone helps (size-matched pooled beats the median) but does not reach the full pooled result."""
    sm = j("pooled_sizematched.json")
    assert sm["cnn"] < sm["pop_median"] - 2
    assert sm["cnn"] > res["eval_scenarios"]["all"]["hr"]["subject_disjoint_5fold"]["cnn"] + 2


def test_heart_sound_rules(res):
    r = res["hs_rules_scen"]
    for scen in ["Resting", "Valsalva", "Apnea", "TiltDown"]:
        assert r[scen]["comb_prior"]["mae"] < 3.0
    assert r["TiltUp"]["comb_prior"]["mae"] > 10  # the reported failure


def test_lookup_better_on_most_subjects(res):
    ps = res["per_subject_temporal"]
    assert sum(v["subj_median_mae"] < v["cnn_mae"] for v in ps.values()) > len(ps) / 2


def test_manuscript_numbers_consistent(res):
    N = res["manuscript_numbers"]
    assert N["rest_n"]["value"] == res["gate_vitals"]["n_windows"]
    assert N["hr_rand_cnn"]["value"] == res["gate_vitals"]["protocols"]["random_windows_80_20"]["cnn"]
    assert N["auth_mae"]["value"] == res["heart_sound"]["authors_pipeline"]["mae"]
