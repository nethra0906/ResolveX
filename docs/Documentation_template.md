# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

ResolveX resolves Source 1 business entities against Source 2/3 candidate records
through a blocking-then-classify pipeline: a multi-pass inverted-index blocking
stage bounds the ~2.2M x 10M possible pair space down to a tractable candidate set,
ranked by a rarity-weighted score so a small candidate set per entity still keeps
the true matches (the competition scores `candidate_pairs.tsv` size directly, in
addition to the `matching_results.tsv` leaderboard score). A LightGBM classifier
then scores each (Source 1, candidate) pair on ~25 name/address/country
similarity features, and a validation-tuned, precision-heavy threshold with
singleton abstention produces the final match sets. On a 150k-entity validation
run (see Section 5), this beat every baseline tested, reaching macro F_0.5 = 0.624
against a best baseline of 0.522, at an average candidate-set size of 9.9 per
Source 1 entity.

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
to multi-million-row sources without a Cartesian product, ranked by a
rarity-weighted score (a key shared by only a handful of records outweighs one
shared by hundreds) so that a small, explicitly-scored candidate cap still
retains true matches, combined with a metric-aware decision layer (global
threshold tuned directly on validation macro F_0.5, plus singleton abstention)
rather than a naive 0.5 probability cutoff.

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
the data. Candidates are scored by `key_type_weight / log2(candidate_side_posting_count + 2)`
summed over shared keys (rarer keys score higher, not just higher-precision key
types) and truncated to `MAX_CANDIDATES_PER_ENTITY` (10) per Source 1 entity.
This ranking exists specifically because the competition scores
`candidate_pairs.tsv` size as part of the final ranking: a flat per-key-type
score was found to let common, weakly-discriminative keys crowd out rare,
highly-discriminative ones once results were truncated, which cost far more
recall per candidate removed than the rarity-weighted version does (see
Experiments in Section 5 for the measured before/after).

**Candidate pairs generated / recall / reduction ratio:** measured on a
150,000-entity subset of Source 1 against the full Source 2/3 tables (see
Limitations in Section 6 for why a subset was used). With the final
rarity-weighted ranking and a cap of 10: 1,358,240 candidate pairs generated,
54.2% candidate recall (281,341 of 518,981 true matches survived blocking),
99.99991% reduction ratio versus the full Cartesian product, average 9.9
candidates per entity. For comparison, the earlier flat-weight ranking at a cap
of 20 produced 2,689,464 candidates (roughly double) for 56.3% recall (only 2.1
points higher) — halving the candidate set cost little recall once ranking
quality improved. In both configurations, average candidates per entity sits
at essentially the configured cap, meaning entities are candidate-limited
rather than naturally sparse.

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

- **F_0.5 Score (macro):** 0.624 (precision 0.682, recall 0.536) on 22,500
  validation entities, at threshold 0.6, using the final rarity-weighted
  blocking ranking at a candidate cap of 10. Baselines (same blocking config):
  exact name match 0.302, exact name + country 0.302, weighted fuzzy similarity
  0.522, TF-IDF retrieval 0.458. For reference, the earlier flat-weight ranking
  at a cap of 20 reached 0.640 macro F_0.5 (precision 0.702, recall 0.549) —
  0.016 higher, at roughly double the candidate set size per entity. The
  rarity-weighted, cap-10 configuration is the one submitted, on the strength
  of the competition's explicit candidate-set-size criterion (see Section 3).
- **Common false positives (wrong merges) — 1,845 total:** near-duplicate
  businesses with a one- or two-character name difference that also share a
  near-identical address (e.g. "IMQ Eye Private Limited" vs "VMQ Eye Private
  Limited" at almost the same address) — genuinely hard cases where the
  features can't distinguish a true near-miss from a real second business at
  that location.
- **Common false negatives (missed matches) — 38,531 total:** a meaningful
  share never reached the model at all because blocking's candidate cap (10
  per entity) excluded them before scoring — error analysis found individual
  Source 1 entities whose every true match was excluded despite being
  near-exact name/address matches (e.g. "Nexus Anchor Rain" at "1111 Church
  Street, Unit 2007, Nashville, TN" had 3+ clearly-matching candidates all cut
  by the cap). This is the direct, measured cost of the memory-driven cap,
  now made smaller still by prioritizing candidate-set size, not a feature or
  model weakness.

---

## 6. Conclusion

LightGBM outperformed every baseline tested in both blocking configurations.
The final submitted configuration (rarity-weighted ranking, candidate cap 10)
reaches macro F_0.5 = 0.624 against a best baseline of 0.522. It was chosen
over the slightly higher-scoring cap-20 configuration (0.640) because it keeps
half as many candidates per Source 1 entity for a 0.016 F_0.5 cost, which is
judged the better trade given the competition scores `candidate_pairs.tsv`
size directly. The precision/recall split (0.682 / 0.536) reflects the tuned
threshold correctly trading recall for precision, matching F_0.5's 2x
precision weighting.

**Limitations.** The full training set (2.2M Source 1 entities) could not be
run to completion: at the original 50-candidate-per-entity cap, the resulting
candidate table (up to ~110M rows) exhausted available RAM on both a local
machine and Kaggle's free-tier notebook. Mitigations applied: the cap was
reduced (first to 20, then to 10 once candidate-set size was confirmed to
count toward ranking), the candidate ranking was changed from a flat per-key-type
score to a rarity-weighted one to limit the recall cost of a smaller cap, and
results reported here were measured on a 150,000-entity representative subset
of Source 1 (all against the full Source 2/3 tables) so a completed, honest
result could be reported within available compute rather than an untested
full-scale claim. The main downstream cost is candidate recall (54.2%), which
caps achievable model recall; error analysis directly confirms this is the
majority driver of false negatives, not a weakness in the similarity features
or classifier. With more RAM (a paid Kaggle tier, or a machine with 32GB+),
the direct next step is running the same rarity-weighted ranking on the full
training set, which should let the cap be raised again (regaining recall)
without losing the candidate-set-size advantage the ranking already provides
at a fixed cap — both settings are exposed in
`notebooks/01_train_and_infer_kaggle.ipynb`. Secondary improvements worth
testing: hard-negative mining on the 1,845 false positives found here, and a
country- or address-density-aware candidate cap instead of one global
constant, since 8.5% of entities received zero candidates entirely in both
configurations.

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
