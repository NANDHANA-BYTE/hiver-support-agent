# Decision Log

1. **Scoped the golden set and classifier to English-language tweets only**
   (365/500 = 73% of the sample; Spanish, Portuguese, Japanese, French,
   German, Italian excluded). Language detection via `langdetect` showed
   12 languages in just 500 rows — supporting all of them well in the
   time available would have diluted everything else. Multilingual support
   is listed under "what we chose not to build."

2. **Brand = Amazon**, not chosen from the raw multi-brand Kaggle dump —
   the provided sample file was already Amazon-only. Didn't second-guess
   a brand pick that was effectively made for us.

3. **Golden labels are a hybrid: regex weak-labeler + hand correction**,
   not 200 rows independently read from scratch. All 500 rows got a rule-based
   intent/escalation label first (`src/weak_label.py`). The 40 rows the rules
   dumped into "other" (i.e., where the labeler admitted it didn't know) were
   individually read and hand-labeled (`src/manual_corrections.py`).
   A `label_source` column (`manual` vs `rule_confirmed`) marks which is
   which — this bookkeeping is what makes the label-circularity finding in
   the report possible.

4. **Added a second manual audit pass**: a random 25-row spot-check of the
   *rule-confirmed* (not "other") rows found 6 clear mislabels, which were
   corrected too. This caught errors the "other"-bucket-only review would
   have missed (rules being confidently wrong, not just unsure).

5. **Sampling for the golden set is stratified, not random**: all rows from
   rare classes (account_access, billing_payment, complaint_escalation,
   product_technical, return_refund_request) were kept; common,
   low-information classes (positive_feedback, social_noise) were capped;
   the "other" bucket was randomly sampled specifically because that's
   where the weak labeler was most likely wrong. A uniform random sample of
   200 rows would have had ~4-5 examples of some rare-but-important intents.

6. **No LLM API key is available in this sandbox**, so the default pipeline
   is 100% offline: retrieval-based reply drafting (TF-IDF nearest neighbor
   over historical (customer, reply) pairs) and a heuristic proxy for the
   LLM-judge rubric. `src/llm_client.py` is written to be a drop-in swap if
   `ANTHROPIC_API_KEY` is set, but that code path is untested end-to-end
   here — flagged explicitly rather than claimed as working.

7. **Reply drafting is retrieval, not generation, by design** — it can only
   recombine phrasing that already exists in the 500-row corpus. This is
   "grounded in historical resolutions" by construction, but is a real
   ceiling on reply quality/novelty (see report failure mode #1).

8. **Escalation confidence threshold (0.15) was calibrated with 5-fold
   cross-validation on the training split only**, using the ~25th percentile
   of out-of-fold softmax confidence. Tuning it against the test set would
   have been a second, more subtle leakage bug on top of the one described
   in decision #9.

9. **Chose to expose, not hide, label circularity in the classifier eval.**
   The rule-based baseline scores 78.6% accuracy on the full test set —
   but ~77% of test labels were "rule_confirmed" (spot-checked, not
   independently derived), so scoring the rules against labels the rules
   themselves produced is close to circular. `eval/eval_classifier.py`
   also reports accuracy on the manual-only subset (rule baseline: 6.2%,
   our TF-IDF model: 25%), which is the fairer comparison.

10. **Escalation policy hard-overrides the classifier** for account-security
    language and explicit repeat-contact phrasing, regardless of predicted
    intent or confidence — these are the highest-cost misses (a wrongly
    auto-handled account-takeover complaint is much worse than a wrongly
    escalated late-delivery ping).

11. **Reported macro-F1 alongside accuracy** for the classifier, since
    intents are imbalanced (8-32 examples per class in the golden set).
    The trivial-majority baseline's near-zero macro-F1 (2.5%) versus its
    deceptively okay-looking accuracy (15.7%, matching class prevalence)
    is the whole reason macro-F1 is in the table at all.

12. **The reply-quality judge scores 4 separate axes** (grounded, relevance,
    tone, actionability) instead of one overall number, so failure modes are
    diagnosable — e.g. our system's tone/actionability scores are decent
    (3.7 / 4.1) while grounded/relevance are weak (2.2 / 2.3), which
    localizes the problem to retrieval quality, not reply phrasing.

13. **Reply text is cleaned of corpus artifacts** (the `@135790` anonymized
    handle prefix and the `^AR` agent-initials suffix) before being shown as
    a "draft reply" or fed to the retriever's similarity index — these are
    dataset/anonymization noise, not brand copy, and leaving them in would
    have made every draft look broken regardless of content quality.

14. **Train/test split (65/35, stratified) is done on the golden set only**,
    not on the full 500-row pool. This keeps the eval honest (no leakage)
    at the cost of a small training set (130 rows / 11 classes) — a
    deliberate trade discussed as the classifier's main limitation.

15. **Did not build multi-turn context.** Every row here is a single
    (customer tweet, brand reply) pair; the real Kaggle dataset has full
    threads. Building a system that reads only one turn was a scoping
    choice made explicit in the report, not an oversight discovered late.
