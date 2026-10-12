# radarvitals-audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23321445.svg)](https://doi.org/10.5281/zenodo.23321445)

Subject identity as the signal in learned radar vital-sign estimation: evaluation floors, split files and code for
the public clinical radar dataset of Schellenberger et al. (Scientific Data 2020; 24 GHz continuous-wave radar, 30
adults, ECG and continuous finger pressure).

- At rest a person's heart rate moves by about 2 beats per minute over ten minutes and systolic pressure by about
  3 mmHg, while subjects differ by about 8 beats per minute and 16 mmHg. Under the within-subject protocols in use
  (random windows, or the last 20% of each recording), the median of each subject's own training windows, which
  reads no radar, scores 1.6 beats per minute and 2.3 / 1.5 mmHg, matching a convolutional network trained on the
  radar and the published results at their own window lengths.
- The same network identifies the subject from a 30-s radar window with 99% accuracy.
- On held-out subjects the network falls to the population median for heart rate and below it for pressure, while
  the dataset authors' own heart-sound pipeline scores 0.6 beats per minute with no subject in the loop.
- Pooling the five recording scenarios, where heart rate varies within a person more than between people, lets the
  same network reach 2.3 beats per minute on held-out subjects, within each scenario; pressure does not follow.

## Running it

Download the three dataset zips from https://doi.org/10.6084/m9.figshare.12186516 (CC BY 4.0), unzip them to
`<root>/raw/` (one folder per subject, `GDN0001` to `GDN0030`), place `additional_data.xlsx` in `<root>/zips/`,
clone the authors' validation code to `<root>/code/scidata_phase1/` (https://gitlab.com/sven_schellenberger/scidata_phase1,
for its `output_validation.mat`), and set `SCHELLENBERGER_ROOT=<root>`.

```
pip install -r requirements.txt
cd scripts
python gate_vitals.py && python gate2_vitals.py && python bp_floor.py && python gate_bp.py
python heart_sound.py && python heart_sound_variants.py
python prep_scenarios.py && python eval_scenarios.py && python hs_cache_scen.py && python hs_rules_scen.py
python replicate_protocols.py && python demographic_floor.py && python per_subject_temporal.py
python pooled_breakdown.py && python pooled_sizematched.py && python export_splits.py
cd .. && python manuscript_numbers.py && python figures.py
python -m pytest tests -q
```

Each script writes one JSON under `results/`; `manuscript_numbers.py` collects every number the paper quotes into
`results/manuscript_numbers.json`, and `tests/test_findings.py` asserts the paper's claims against the result files.
The whole set runs in about an hour on one GPU. `splits/` holds, for every scenario and for the pooled set, which
windows are test under the random-window and temporal protocols and the subject folds of the subject-disjoint one,
keyed by subject and window start time.

## Layout

- `scripts/gate_vitals.py`: windows, ECG heart-rate labels, the convolutional network, the three protocols.
- `scripts/gate_bp.py`, `scripts/bp_floor.py`: pressure labels from the finger-pressure waveform and the same protocols.
- `scripts/heart_sound.py`, `heart_sound_variants.py`, `hs_cache_scen.py`, `hs_rules_scen.py`: the authors'
  pipeline from their validation output, and the Schmidt-style re-implementation with its lag-selection rules.
- `scripts/prep_scenarios.py`, `eval_scenarios.py`, `pooled_breakdown.py`, `pooled_sizematched.py`: the four other
  scenarios, their union, and the controls.
- `scripts/replicate_protocols.py`: the published protocols at their own window lengths.
- `scripts/gate2_vitals.py`: the subject-identification probe. `scripts/demographic_floor.py`, `per_subject_temporal.py`.

## Citing

See `CITATION.cff`. The dataset is Schellenberger et al., Sci Data 7:291 (2020), https://doi.org/10.1038/s41597-020-00629-5.
