# ResolveX — Business Entity Resolution

An ML pipeline for the Amazon ML Challenge 2026 Business Entity Resolution
challenge: given noisy business records from three independent sources, find every
Source 2 / Source 3 record that refers to the same real-world business as each
Source 1 entity (zero, one, or many matches).

## Repository structure

```
resources/                  competition-provided data & utilities (read-only)
  dataset/{train,test}/      *_source1/2/3.tsv, train_ground_truth.tsv
  utils/validate_submission.py
  Amazon ML Challenge - Problem Statement.pdf
  Documentation_template.md

src/resolvex/                pipeline source code
  config.py                  paths, schema, tunable constants
  data_loader.py              TSV loading
  normalization.py            multi-representation name/address normalization
  blocking.py                  candidate generation (multi-pass inverted-index join)
  features.py                  pairwise similarity feature engineering
  models.py                    baselines + LightGBM matcher
  training.py                  grouped train/val split, model fitting
  evaluation.py                 official macro F_0.5 metric
  thresholding.py               decision layer (accept threshold, singleton abstention)
  postprocessing.py             output-rule enforcement
  inference.py                  batch scoring
  pipeline.py                    end-to-end orchestration

notebooks/                    entry points — see "Running the pipeline" below
  00_profile_data_local.ipynb    LOCAL — data profiling
  01_train_and_infer_kaggle.ipynb KAGGLE (CPU notebook) — blocking, features,
                                   training, threshold tuning, error analysis,
                                   test inference
  02_validate_and_package_local.ipynb  LOCAL — validate outputs, package submission zip
experiments/                  profiling reports, metrics, configs, error analysis
output/                       matching_results.tsv, candidate_pairs.tsv
tests/                        unit tests (pytest) — run locally, no dataset needed
docs/                         filled methodology document
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Python 3.11+ is expected (tested on 3.11.0).

## Running the pipeline

**Blocking and feature engineering are CPU-bound (pandas/regex/rapidfuzz string
work), not GPU-parallelizable — a GPU does not speed up the slow part of this
pipeline.** LightGBM training itself is the only GPU-capable step and was never
the bottleneck. What actually helps at this dataset's scale (millions of rows per
source) is more RAM/CPU cores than a typical laptop has, so the heavy stage is
designed to run on **Kaggle's CPU notebook** (not GPU — GPU notebooks give you
*fewer* CPU cores).

| Step | Where | Notebook |
|---|---|---|
| 1. Data profiling | **Local** (fast, ~1 min) | `notebooks/00_profile_data_local.ipynb` |
| 2. Blocking + features + training + threshold tuning + error analysis + test inference | **Kaggle, CPU notebook (accelerator = None)** | `notebooks/01_train_and_infer_kaggle.ipynb` |
| 3. Validate outputs + package submission zip | **Local** | `notebooks/02_validate_and_package_local.ipynb` |

### Before running the Kaggle notebook

1. Upload your local `src` folder (contains `resolvex/`) as a private Kaggle
   Dataset — name it e.g. `resolvex-src`.
2. Upload your local `resources/dataset` folder (contains `train/` and `test/`)
   as another private Kaggle Dataset — name it e.g. `resolvex-data`. This is
   still only the competition-provided data; you're relocating where it's
   processed, not adding any external information.
3. Create a new Kaggle Notebook, attach both datasets as inputs, set
   **Accelerator = None**, and upload/paste in
   `notebooks/01_train_and_infer_kaggle.ipynb`. Run all cells.
4. Download the `/kaggle/working` output (Kaggle's Output tab → Download All)
   and copy these files into the matching paths in your local `ResolveX` folder:

   ```
   output/matching_results.tsv
   output/candidate_pairs.tsv
   experiments/models/lgbm_matcher.joblib
   experiments/configs/threshold.json
   experiments/metrics/blocking_quality_full.json
   experiments/metrics/baseline_results.json
   experiments/metrics/threshold_grid.csv
   experiments/metrics/val_metrics.json
   experiments/error_analysis/error_examples.json
   ```

5. Run `notebooks/02_validate_and_package_local.ipynb` locally, which validates
   the two output files against every competition rule and builds
   `resolvex_submission.zip`.

The library code (`src/resolvex/`) is identical on both hosts — a small set of
`RESOLVEX_*` environment variables (set at the top of the Kaggle notebook) point
it at Kaggle's paths instead of the local `resources/`/`output/`/`experiments/`
folders; nothing in `src/resolvex/` itself needs editing per host.

## Tests

Run locally — fast, no dataset download required:

```bash
python -m pytest tests/ -q
```

## Design notes

- **Scale.** Source 1 has ~2.2M (train) / ~1.7M (test) rows; Source 2/3 each have
  ~5M rows. A full Cartesian product is infeasible, so blocking (multi-pass
  inverted-index join on name tokens, name prefixes, address digit tokens, and
  address tokens, partitioned by country) is the load-bearing component — it sets
  the recall ceiling for everything downstream.
- **Country is an open label.** The test set introduces France, unseen in
  training. Nothing in the pipeline hard-codes or one-hot-encodes a fixed country
  set; country is compared by string equality only.
- **No external data.** Only the competition-provided TSVs are used anywhere in
  the pipeline — no geocoding, no business-registry lookups, no internet access at
  inference time.
- **Model license/size.** The final matcher is LightGBM (MIT license,
  microsoft/LightGBM), a classical gradient-boosted tree ensemble — far under the
  8B-parameter ceiling and unambiguously license-compliant.
- **Metric-aware decisions.** F_0.5 weights precision 2x over recall and scores a
  true singleton 1.0 only for a correctly empty prediction; `thresholding.py`
  tunes a single global threshold on validation F_0.5 (never on test labels) and
  applies a singleton-abstention rule rather than a flat 0.5 cutoff.
