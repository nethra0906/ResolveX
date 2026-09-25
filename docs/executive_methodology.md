# ResolveX — Executive Methodology Summary

*Business Entity Resolution Challenge — ML Challenge 2026*

## ML Approach

ResolveX resolves each Source 1 business entity against candidate records from
Source 2 and Source 3 through a three-stage blocking-then-classify pipeline:

1. **Candidate generation (blocking).** A full pairwise comparison between Source 1
   (~2.2M train / ~1.7M test rows) and Source 2+3 (~10M rows combined) is
   computationally infeasible, so candidates are generated via a multi-pass
   inverted-index join: business names and addresses are normalized into several
   representations (legal-suffix-collapsed name tokens, street-abbreviation-expanded
   address tokens, address digit tokens, name prefixes), and Source 1 records are
   matched against Source 2/3 records that share at least one blocking key within
   the same normalized country. Keys that would make the join explode (very common
   tokens) are pruned before joining. Candidates are ranked by a weighted count of
   matching key types and capped per Source 1 entity.
2. **Pairwise feature engineering.** For every (Source 1, candidate) pair that
   survives blocking, ~25 features are computed: string-similarity metrics
   (Levenshtein, Jaro-Winkler, token-sort/token-set ratio, Jaccard, character
   n-gram Jaccard, TF-IDF cosine) applied separately to name and address, plus
   length/token-count statistics, shared rare-token signals, exact country
   equality, and cross-field interaction terms.
3. **Supervised classification + metric-aware decision layer.** A LightGBM binary
   classifier scores each candidate pair. Rather than a flat 0.5 probability
   cutoff, the accept threshold is grid-searched directly against the
   competition's own macro F_0.5 metric on a held-out, entity-grouped validation
   split, and a singleton-abstention rule withholds all predictions for an entity
   whose best candidate score doesn't clear that threshold — directly targeting
   the metric's precision-heavy, singleton-aware scoring rather than a generic
   classification threshold.

## ML Models Used

- **Baselines** (baselines section of `notebooks/01_train_and_infer_kaggle.ipynb`): exact normalized-name match, exact
  name+country match, a hand-weighted linear combination of fuzzy similarity
  scores, and TF-IDF cosine retrieval alone — establishing what a
  non-learned approach achieves before crediting the supervised model with any
  improvement.
- **Candidate model:** LightGBM (`lightgbm.LGBMClassifier`) — MIT licensed,
  far under the competition's 8B-parameter ceiling, and a strong empirical fit
  for dense tabular pairwise features at multi-million-pair scale (fast to train
  and score, handles feature interactions and non-linearities without manual
  feature-crossing beyond what's already included).
- **Final selected model:** LightGBM, selected over the baselines by validation
  macro F_0.5 (see Experiments table below).

## Experiments

*(Fill in from `experiments/metrics/` once the full-scale run completes —
`blocking_quality_full.json`, `val_metrics.json`, `baseline_results.json`,
`threshold_grid.csv`.)*

| Experiment | Macro F_0.5 (val) | Notes |
|---|---|---|
| Exact normalized name | — | Baseline |
| Exact name + country | — | Baseline |
| Weighted fuzzy similarity | — | Baseline |
| TF-IDF retrieval | — | Baseline |
| LightGBM (final) | — | Selected model |

## Conclusion

*(Fill in once training/evaluation completes: what worked, what mattered most,
final validation performance, limitations, future improvements.)*
