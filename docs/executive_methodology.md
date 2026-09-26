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
   tokens) are pruned before joining. Candidates are ranked by a **rarity-weighted**
   score, not a flat per-key-type score: each shared key contributes
   `key_type_weight / log2(candidate_side_posting_count + 2)`, so a key shared by
   only two or three records (highly specific, almost certainly the same business)
   outweighs one shared by hundreds (common, weakly discriminative). This exists
   specifically because the competition scores `candidate_pairs.tsv` itself as
   part of the final ranking, in addition to `matching_results.tsv`'s leaderboard
   score: a smaller candidate set per Source 1 entity ranks higher, so the ranking
   needs to be precise enough to keep true matches under a small cap, not just
   under a large one. Candidates are truncated to a fixed cap per Source 1 entity
   after ranking.
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

Two full runs, both on the same 150,000-entity representative subset of
Source 1 (of 2.2M total) against the full Source 2 and Source 3 tables (~10M
rows combined), split 127,500 / 22,500 train/validation by entity, so the two
runs are directly comparable and differ only in the blocking configuration
described below. A full 2.2M-entity run was attempted first and repeatedly
exhausted available RAM (both on a local machine and on Kaggle's free tier);
the subset above is what let a completed, honest result be reported within
available compute — see Limitations below.

**Run 1 (cap 20, flat key-type ranking):** the original blocking scheme —
candidates ranked by a fixed weight per key type, truncated to 20 per entity.

**Run 2 (cap 10, rarity-weighted ranking):** after the organizers clarified
that `candidate_pairs.tsv` size counts toward the final ranking, the ranking
was changed to weight each shared key by how rare it is on the candidate side
(see ML Approach above), and the cap was halved to 10.

| Experiment | Macro F_0.5 (val) | Notes |
|---|---|---|
| Exact normalized name | 0.302 | Baseline, Run 2 |
| Exact name + country | 0.302 | Baseline, Run 2 |
| Weighted fuzzy similarity | 0.522 | Baseline, Run 2 |
| TF-IDF retrieval | 0.458 | Baseline, Run 2 |
| LightGBM, Run 1 (cap 20) | 0.640 | precision 0.702, recall 0.549, threshold 0.6 |
| LightGBM, Run 2 (cap 10) | **0.624** | precision 0.682, recall 0.536, threshold 0.6 — **final submitted approach** |

Blocking quality, Run 1 vs Run 2:

| Metric | Run 1 (cap 20) | Run 2 (cap 10) |
|---|---|---|
| Candidate pairs | 2,689,464 | 1,358,240 |
| Average candidates / entity | 19.6 | 9.9 |
| Candidate recall | 56.3% | 54.2% |
| Reduction ratio | 99.9998% | 99.99991% |

Halving the candidate budget cost 1.6 points of macro F_0.5 (0.640 to 0.624)
and 2.1 points of candidate recall (56.3% to 54.2%), while cutting the
candidate set size in half. The rarity-weighted ranking is what keeps that
cost small: a flat-weight scheme at the same cap would be expected to lose
substantially more, since it was already sitting at its cap under Run 1's
looser budget (average candidates per entity equalled the cap in both runs,
meaning blocking is finding far more raw matches than either cap keeps, and
ranking quality determines which ones survive).

## Conclusion

LightGBM outperformed every baseline in both runs by a wide margin. Run 2
(cap 10, the final submitted configuration) reaches macro F_0.5 = 0.624
against a best baseline of 0.522. The precision/recall split (0.682 / 0.536)
is consistent with F_0.5's precision weighting: the tuned threshold (0.6)
trades some recall for higher precision, the intended behavior for a metric
that penalizes false merges twice as heavily as missed matches.

Run 2 was selected as the final approach over Run 1 despite its slightly lower
F_0.5, because the competition scores `candidate_pairs.tsv` size directly: Run
2's candidate set is half the size of Run 1's for a 1.6-point F_0.5 cost, which
is judged a better trade given that explicit scoring criterion.

The single largest limitation is candidate recall (54.2% in Run 2), which caps
the system's achievable recall regardless of model quality — of the 38,531
validation-set false negatives, a meaningful share never appeared as
candidates at all rather than being scored incorrectly. This is a direct,
measured consequence of the memory-driven candidate cap, not a fundamental
limit of the blocking keys themselves: average candidates per entity sits at
essentially the cap in both runs, meaning most entities are candidate-limited
rather than naturally sparse. Error analysis surfaced individual entities whose
every true match was excluded by the cap despite being near-exact name/address
matches, which supports this diagnosis directly.

Given more compute (a machine or Kaggle tier with more RAM), the most direct
next step is running the same rarity-weighted ranking on the full 2.2M-entity
training set, which should let the cap be raised again (regaining recall)
without giving up the candidate-set-size advantage the ranking change already
provides at a fixed cap — `notebooks/01_train_and_infer_kaggle.ipynb` already
exposes both settings for exactly this purpose. Secondary improvements worth
testing: hard-negative mining on the false positives found here, and a country-
or address-density-aware candidate cap instead of a single global constant,
since 8.5% of entities in both runs received zero candidates entirely.
