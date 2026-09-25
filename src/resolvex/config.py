"""Central configuration: paths, column names, and tunable pipeline constants.

All paths are relative to the repository root by default so the project runs
unmodified on another machine. Three environment variables let the same library
code run unmodified on Kaggle (or any other host with a different filesystem
layout) instead of hard-coding a second set of paths in the notebooks:

- ``RESOLVEX_DATA_ROOT``: directory containing ``train/`` and ``test/`` (defaults
  to ``resources/dataset``). On Kaggle, point this at the uploaded dataset, e.g.
  ``/kaggle/input/resolvex-data``.
- ``RESOLVEX_OUTPUT_ROOT``: where matching_results.tsv / candidate_pairs.tsv are
  written (defaults to ``output/``). On Kaggle: ``/kaggle/working/output``.
- ``RESOLVEX_EXPERIMENTS_ROOT``: where models/metrics/configs/reports are written
  (defaults to ``experiments/``). On Kaggle: ``/kaggle/working/experiments``.

Set these with ``os.environ[...] = ...`` *before* importing ``resolvex.config``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root = two levels up from this file (src/resolvex/config.py -> repo root)
ROOT = Path(__file__).resolve().parents[2]

RESOURCES_DIR = ROOT / "resources"
UTILS_DIR = RESOURCES_DIR / "utils"

DATASET_DIR = Path(os.environ.get("RESOLVEX_DATA_ROOT", RESOURCES_DIR / "dataset"))
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = Path(os.environ.get("RESOLVEX_OUTPUT_ROOT", ROOT / "output"))
EXPERIMENTS_DIR = Path(os.environ.get("RESOLVEX_EXPERIMENTS_ROOT", ROOT / "experiments"))
MODELS_DIR = EXPERIMENTS_DIR / "models"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
METRICS_DIR = EXPERIMENTS_DIR / "metrics"
REPORTS_DIR = EXPERIMENTS_DIR / "reports"
ERROR_ANALYSIS_DIR = EXPERIMENTS_DIR / "error_analysis"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# --- Schema -----------------------------------------------------------------

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

ID_COLUMN = "entity_id"
NAME_COLUMN = "business_name"
ADDRESS_COLUMN = "business_address"
COUNTRY_COLUMN = "country"

# dtypes for pandas.read_csv — keeps memory down vs. default object inference
SOURCE_DTYPES = {
    "entity_id": "string",
    "business_name": "string",
    "business_address": "string",
    "country": "string",
}

# --- Random seed --------------------------------------------------------------

RANDOM_SEED = 42

# --- Blocking -----------------------------------------------------------------

# Minimum token length to be used as a blocking key (drops very short/common tokens
# like "of", "inc" acting alone from generating explosive candidate sets)
MIN_BLOCKING_TOKEN_LEN = 3

# Character n-gram size used for the n-gram blocking pass on business names
NAME_NGRAM_SIZE = 4

# Cap on how many candidates a single blocking key is allowed to emit; keys that
# match more than this many records on both sides are considered too common
# (e.g. "LLC", "PVT LTD") and are skipped to keep candidate volume bounded.
MAX_BLOCK_KEY_POSTING_LIST = 500

# Hard cap on candidates kept per Source-1 entity after all blocking passes are
# unioned and ranked (keeps candidate_pairs.tsv / model inference bounded).
MAX_CANDIDATES_PER_ENTITY = 50

# --- Validation split -----------------------------------------------------------

VAL_FRACTION = 0.15

# --- Model --------------------------------------------------------------------

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_estimators": 400,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "verbosity": -1,
}

# Negative sampling: how many hard/random negatives to keep per positive pair
NEGATIVES_PER_POSITIVE = 5
