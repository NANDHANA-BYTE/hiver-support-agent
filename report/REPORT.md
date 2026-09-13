# Report — Amazon Twitter Support Agent

## 1. Problem framing

**Brand:** Amazon, on the `@AmazonHelp`-style public support handle traffic
in the provided sample (500 (customer, brand-reply) pairs).

**What "good" means for this brand, specifically:** Amazon's Twitter support
line is public, high-volume, and mostly *not* the place accounts get fixed —
almost every real resolution in this data ends with "please DM us" or "call
this number," because Twitter can't safely exchange order/account details.
Given that, a *good* agent here is not one that resolves issues end-to-end;
it's one that:

1. **Never confidently auto-handles a case it will get expensively wrong** —
   account security, payment disputes, and repeat/escalated complaints need
   a human, every time, even if that costs some auto-handle rate.
2. **Never fabricates a promise** (a refund amount, a delivery date, a policy)
   that isn't backed by how the brand has actually resolved similar cases.
3. **Is honest about its own uncertainty** — routing a message to a human
   because the model isn't sure is a feature, not a failure, for a
   support agent operating on a public, brand-reputation-sensitive channel.

**What we chose not to build** (see decision log for the "why," this is the
"what"):
- Multi-turn conversation handling — every example here is a single
  (customer, reply) pair, not a full thread.
- Non-English support (27% of the raw sample, excluded).
- A validated end-to-end LLM generation/judging path — the code exists
  (`src/llm_client.py`) but was never run against a real API key in this
  environment, so it isn't part of the headline numbers.
- Confidence calibration beyond a single CV-tuned threshold (no Platt/
  temperature scaling).
- Personalization using account history — nothing here uses anything
  beyond the text of the single tweet.

## 2. Results vs. baselines

### Intent classification (test set, n=70)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Trivial (always predict majority class) | 15.7% | 2.5% |
| Simple (keyword/regex rules) | **78.6%** | **77.3%** |
| Ours (TF-IDF + Logistic Regression) | 44.3% | 36.3% |

The simple baseline looks like it wins outright. **It doesn't, really** — see
Section 4.

### Escalation decision (test set, n=70; 51/200 golden rows are `escalate=True`)

| Model | Accuracy | Precision (escalate) | Recall (escalate) | F1 (escalate) |
|---|---|---|---|---|
| Trivial (never escalate) | 75.7% | — | 0% | — |
| Ours (classifier intent + confidence + hard-override rules) | 70.0% | 43.3% | **76.5%** | 55.3% |

### Reply quality (heuristic proxy judge, 1-5 scale, test set, n=70)

| Axis | Score |
|---|---|
| Grounded (matches historical resolution pattern) | 2.20 |
| Relevance (addresses the actual message) | 2.30 |
| Tone | 3.71 |
| Actionability (gives a concrete next step) | 4.11 |
| **Overall** | **3.08** |

**Judge-vs-human agreement** (20 replies, hand-scored independently):
Pearson r = 0.45, Spearman r = 0.44, mean absolute error = 0.81 (on a 1-5
scale). That's a real but modest correlation — good enough to catch gross
failures, not good enough to trust for fine-grained ranking. This is a proxy
judge (no LLM key available), and it shows.

## 3. Failure analysis — top 5 modes

1. **Retrieval grounds on the wrong historical case.** Example: customer
   message *"Yes and it makes no sense. It says they had no access!"*
   (an account-access follow-up) retrieved a delivery-date reply as its
   closest historical match. Hypothesis: TF-IDF nearest-neighbor over a
   500-row corpus is doing pure lexical matching on very short, pronoun-heavy
   messages ("it," "they," "no") that carry almost no topical signal —
   there's nothing in the words themselves to match on.

2. **Cleaned replies are sometimes truncated mid-thread fragments.** Example:
   a "draft reply" reading *"be personal information. Our page is visible
   to the public. (3/3)"* — the corpus actually stored replies 2/3 and 3/3
   of a multi-tweet answer as if they were standalone. Hypothesis: the
   original Twitter thread structure (multi-tweet replies) wasn't
   reconstructed before this row was created, so the retriever sometimes
   picks a reply's *middle*, not its start.

3. **Escalation over-triggers on low classifier confidence.** With only 130
   training rows across 11 classes, softmax confidence rarely exceeds 0.3;
   the CV-calibrated 0.15 threshold still causes 34/70 test escalations
   (out of 51 total) to fire purely on low confidence, dragging
   precision-on-escalate down to 43%. Hypothesis: this is a training-data-size
   problem, not a rule-design problem — more labeled data per class would
   raise confidence on genuinely easy cases and separate them from
   genuinely ambiguous ones.

4. **A quarter of the brand's real traffic is invisible to this system.**
   135/500 rows (27%) are non-English and were scoped out entirely (decision
   log #1). The headline numbers describe performance on English-speaking
   customers only; a production version serving all of Amazon's Twitter
   traffic would silently mishandle everything outside that slice today.

5. **Short, context-dependent follow-ups regress to generic replies.**
   Example: *"Thank you, sent over my message."* and *"Then? How do I do
   it?"* are clearly the 2nd/3rd turn of a longer conversation, but this
   dataset only gives the system a single isolated tweet. The system drafts
   a plausible-sounding but non-specific reply because it has no access to
   what was actually being followed up on. This isn't really a bug in this
   codebase — it's the consequence of the single-turn scoping decision
   (decision log #15), and would need multi-turn context to fix.

## 4. What is misleading about my headline number?

**"The simple baseline (78.6% accuracy) beats our system (44.3%)" is
misleading**, because of label circularity: ~77% of the 200-row golden set
(154/200) was labeled by *confirming* the same regex rules that the "simple
baseline" *is*. Scoring those rules against labels that are, for most rows,
literally their own output is close to grading a classifier on its own
answer key. The 46 rows that were actually hand-corrected from scratch (i.e.
the rows where the rules were wrong) are the fair test. On those 16
manual-only test rows: the rule baseline drops to **6.2% accuracy**, and our
TF-IDF model — itself weak, at 25% — is more than 4x better. Neither number
is good in absolute terms (11 classes, 130 training rows is a genuinely hard
regime), but the *comparison* flips once circularity is removed.

**"Our escalation system (70.0% accuracy) is worse than doing nothing
(75.7%)" is also misleading**, in the other direction: accuracy is the wrong
metric for a decision where false negatives (missing a case that needed a
human) are much more expensive than false positives (a human reviews
something that didn't strictly need it). The trivial "never escalate"
baseline gets 0% recall on the cases that matter — it would let every
account-security and repeated-complaint case go straight to an automated
reply. Our system's 76.5% recall on `escalate=True`, at the cost of some
extra human review load (43% precision), is the actual trade a real
deployment would want, and no single accuracy number shows that trade-off.

## 5. What I'd do with one more week

1. **Fix the retrieval quality problem directly** (failure mode #1): require
   a minimum cosine-similarity threshold before trusting a retrieved
   grounding example; below it, escalate with reason "no sufficiently
   similar historical case found" instead of drafting from a bad match.
2. **Filter out mid-thread reply fragments** from the retrieval corpus
   (failure mode #2) — e.g. drop replies that don't start with a capital
   letter or a greeting, or that contain a `(2/2)`/`(3/3)` marker, since
   those are provably partial.
3. **Wire in and validate the real LLM path** (`src/llm_client.py`) end to
   end with an actual API key, and re-run the exact same eval harness
   (already built to accept either path) to see whether generation beats
   retrieval on the grounded/relevance axes specifically.
4. **Double the golden set to ~400-500 rows with two independent
   annotators**, so an actual inter-annotator-agreement number replaces the
   current single-annotator (me) labels — right now there's no way to know
   how much of the 44%/70%/3.08 numbers reflects genuine model performance
   vs. label noise in a set I built alone.
5. **Calibrate the classifier properly** (temperature or Platt scaling on a
   held-out validation fold) instead of a raw-softmax + CV-percentile
   threshold, to directly attack failure mode #3.
6. **Scope in the top 1-2 non-English languages by volume** (Spanish and
   Japanese were the largest non-English groups here) rather than leaving
   27% of traffic unhandled.
