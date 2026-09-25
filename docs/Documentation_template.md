# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

ResolveX resolves Source 1 business entities against Source 2/3 candidate records
through a blocking-then-classify pipeline: a multi-pass inverted-index blocking
stage bounds the ~2.2M x 10M possible pair space down to a tractable candidate set,
a LightGBM classifier scores each (Source 1, candidate) pair on ~25 name/address/
country similarity features, and a validation-tuned, precision-heavy threshold with
singleton abstention produces the final match sets. *(Fill in final headline
numbers once training completes — see experiments/metrics/.)*

---

## 2. Methodology

### 2.1 Problem Analysis

Key facts from `experiments/reports/data_profile.md` (full-file profiling, no
sampling):

- Source 1: 2.21M rows (train) / 1.73M rows (test); Source 2/3: ~5M rows each.
- Ground truth: 5.6% true singletons, 5.4% single match, 89.0% multi-match
  (avg 3.46 matches, max 11) — this is a **multi-match-heavy** resolution problem,
  not the singleton-dominated case a naive one-to-one assumption would target.
- Test set adds France (absent from training) — country must stay an open string
  label throughout normalization/features/blocking.
- Non-ASCII business names (Devanagari script for India records) are common —
  normalization must not assume ASCII.
- Noise matches the problem statement's description: legal-suffix variants
  (Pvt/Private, Ltd/Limited, Inc/Incorporated), leading junk characters (`--`,
  `<<`), street abbreviations (Rd/Road, St/Street), missing address components.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (supervised pairwise matching)
**Core Innovation:** Multi-pass vectorized inverted-index blocking (name tokens,
name prefixes, address digit tokens, address tokens; country-partitioned;
common-key pruning bounded by a left×right posting-list-product cap) that scales
to multi-million-row sources without a Cartesian product, combined with a
metric-aware decision layer (global threshold tuned directly on validation macro
F_0.5, plus singleton abstention) rather than a naive 0.5 probability cutoff.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used** (see `src/resolvex/blocking.py`):

- `name_token`: individual normalized name tokens (legal-suffix variants
  collapsed to a canonical form first), length >= 3
- `name_prefix`: first 4 alphanumeric characters of the normalized name — robust
  to trailing typos/suffix differences
- `addr_digit`: numeric tokens in the address (house numbers, PIN/ZIP codes) —
  highest weight, since shared numeric tokens are strong precision signals
- `addr_token`: address tokens excluding a small generic street-type/direction
  stopword list

All keys are partitioned by normalized country (open-vocabulary string equality —
never a fixed/one-hot country set) before joining. Overly common keys are pruned
before the join: any key whose posting-list length exceeds
`MAX_BLOCK_KEY_POSTING_LIST` (500) on either side, or whose left×right count
product exceeds `MAX_KEY_PAIR_PRODUCT` (50,000), is dropped — this bounds the
join's row count without a hand-picked stopword list for every language/script in
the data. Candidates are scored by a weighted sum of matching key types and
truncated to `MAX_CANDIDATES_PER_ENTITY` (50) per Source 1 entity.

**Candidate pairs generated / recall / reduction ratio:** *(fill in from
`experiments/metrics/blocking_quality_full.json` once the full-scale run
completes)*

**How true matches were not lost:** candidate recall is measured directly against
`train_ground_truth.tsv` (see the blocking-quality cell in
`notebooks/01_train_and_infer_kaggle.ipynb`) rather than assumed; `candidate_pairs.tsv` is exactly
the set fed to the model (not an earlier, looser blocking pass), so the reported
recall is the true ceiling on the model's achievable recall.

---

## 4. Matching Model

**Features used** (`src/resolvex/features.py`, ~25 features):

- Name features: exact raw/normalized match, Levenshtein ratio, Jaro-Winkler
  (WRatio), token-sort ratio, token-set ratio, token Jaccard, character 3-gram
  Jaccard, TF-IDF (char 3-4-gram) cosine, length ratio/diff, token-count diff,
  shared rare-token IDF weight
- Address features: exact normalized match, Levenshtein ratio, token-sort ratio,
  token Jaccard, TF-IDF cosine, shared digit token (house number/PIN/ZIP), shared
  rare-token IDF weight, length ratio/diff
- Country: exact normalized-string equality (open vocabulary)
- Cross-field: name×address Jaccard product, country×name Jaccard,
  country×address Jaccard

**Model type:** LightGBM (`lightgbm.LGBMClassifier`, MIT license,
github.com/microsoft/LightGBM) — a classical gradient-boosted tree ensemble,
well under the competition's 8B-parameter ceiling. Baselines evaluated for
comparison: exact normalized name match, exact name+country match, a
hand-weighted fuzzy-similarity linear combination, and TF-IDF cosine retrieval
alone (baselines section of `notebooks/01_train_and_infer_kaggle.ipynb`).

**Threshold selection method:** a single global score threshold, grid-searched
directly against validation-set macro F_0.5 (never test labels — see
`src/resolvex/thresholding.py`), combined with a singleton-abstention rule: an
entity receives no predictions at all unless its best candidate score clears the
threshold. This directly targets the competition's precision-heavy metric instead
of a generic 0.5 probability cutoff.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** *(fill in from `experiments/metrics/val_metrics.json`)*
- **Common false positives (wrong merges):** *(fill in after error analysis)*
- **Common false negatives (missed matches):** *(fill in after error analysis)*

---

## 6. Conclusion

*(Fill in once training/evaluation is complete.)*

---

## Appendix

### A. Code Artefacts

Full runnable pipeline under `code/business_entity_resolution/src/` (mirrors
`src/resolvex/` in the development repo), with `README.md` (exact reproduction
commands) and `requirements.txt` (pinned versions). Entry points (notebooks,
not scripts — see the README for which run locally vs. on Kaggle):

- `notebooks/00_profile_data_local.ipynb` — data profiling report (local)
- `notebooks/01_train_and_infer_kaggle.ipynb` — blocking + feature engineering +
  LightGBM training + validation threshold tuning + error analysis + full
  test-set inference, writing `output/matching_results.tsv` and
  `output/candidate_pairs.tsv` (Kaggle CPU notebook)
- `notebooks/02_validate_and_package_local.ipynb` — validates outputs and
  packages the final submission zip (local)

### B. Additional Results

*(Fill in ablations / additional charts once available.)*

---

**Note:** Teams can modify sections according to their approach while maintaining
clarity and technical depth.
