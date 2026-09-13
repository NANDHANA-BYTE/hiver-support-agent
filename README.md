# Amazon Twitter Support Agent — Hiver SDE Intern Assignment

An AI support agent for **Amazon** (brand chosen because the provided
sample, `amazon_golden_labeling.csv`, was already Amazon's Twitter support
traffic) that classifies intent, decides auto-handle vs. escalate, and
drafts a reply grounded in Amazon's own historical resolutions.

Full writeup: [`report/REPORT.md`](report/REPORT.md).
Non-obvious decisions: [`decision_log.md`](decision_log.md).

## Reproduce in under 15 minutes

```bash
pip install -r requirements.txt

# 1. Bootstrap weak labels + build the 200-row hand-reviewed golden set
python3 src/weak_label.py
python3 src/detect_lang.py
python3 src/sample_golden.py
python3 src/build_golden.py
python3 src/split.py

# 2. Train the classifier
python3 src/classify.py

# 3. Run the three evaluation harnesses
python3 eval/eval_classifier.py     # intent classification vs. 2 baselines
python3 eval/eval_escalation.py     # escalation decision quality
python3 eval/run_eval.py            # full pipeline + reply-quality judge
python3 eval/human_agreement.py     # judge-vs-human agreement check

# 4. Try it live
python3 src/pipeline.py "my package says delivered but I never got it, third time this happens"
```

No API key is required — the repo runs fully offline using a retrieval-based
reply drafter and a heuristic quality judge. If you set `ANTHROPIC_API_KEY`,
`src/llm_client.py` is picked up automatically for both reply generation and
judging (untested end-to-end in this environment — see decision log #6).

## Headline results (test set, n=70, held out from training)

| Task | Metric | Trivial baseline | Simple baseline | Our system |
|---|---|---|---|---|
| Intent classification | accuracy | 15.7% | 78.6%* | 44.3% |
| Intent classification | macro-F1 | 2.5% | 77.3%* | 36.3% |
| Escalation decision | accuracy | 75.7% | — | 70.0% |
| Escalation decision | recall on `escalate=True` | 0% | — | 76.5% |
| Reply quality (heuristic judge, 1-5) | overall | — | — | 3.08 |

*See ["What's misleading about my headline number"](report/REPORT.md#whats-misleading-about-my-headline-number)
— the simple rule-based baseline's 78.6% is inflated by label circularity
and drops to 6.2% on the strictly hand-corrected subset of the test set.

## Repo layout

```
data/           raw sample, weak labels, golden set, train/test split, trained model
src/            weak_label.py, sample_golden.py, build_golden.py, classify.py,
                draft_reply.py, llm_client.py, pipeline.py, manual_corrections.py
eval/           eval_classifier.py, eval_escalation.py, run_eval.py,
                llm_judge.py, human_agreement.py
report/         REPORT.md (problem framing, baselines, failure analysis, next steps)
decision_log.md 15 non-obvious decisions and why
```

## Citations / borrowed material

- Dataset structure follows Kaggle's *Customer Support on Twitter*
  (`thoughtvector/customer-support-on-twitter`); this run uses a 500-row
  Amazon-only sample rather than the full ~3M-tweet dataset (see decision log #1).
- Standard library usage only beyond `scikit-learn`, `pandas`, `scipy`,
  `langdetect`, `joblib`; no copied code from external repos or tutorials.
