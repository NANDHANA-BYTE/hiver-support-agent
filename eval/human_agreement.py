"""
Human-vs-heuristic-judge agreement check (assignment deliverable #3:
"evidence of how well your judge agrees with a human").

Methodology: 20 test-set (message, draft_reply) pairs were sampled
(random_state=11, see the sampling snippet in decision_log.md #9) and
each was read and scored 1-5 for overall quality by hand, independent of
the heuristic judge's score. HUMAN_SCORES below records those judgments,
keyed by golden_id, with a one-line reason for the harder calls.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from scipy.stats import pearsonr, spearmanr

HUMAN_SCORES = {
    192: (1.5, "retrieved grounding was about a delivery date, message was about account access - mismatch"),
    330: (3.5, "on-topic, appropriately cautious about a payment/fraud issue"),
    274: (3.5, "plausible troubleshooting step for a sign-in bug"),
    231: (1.0, "reply ('Enjoy your weekend') completely ignores a service complaint"),
    1: (2.0, "message was just a confirmation; reply asks an unrelated question"),
    478: (3.0, "generic but on-topic, gives a next step"),
    355: (3.5, "acknowledges disappointment and offers to help, reasonably on-topic"),
    16: (1.0, "reply text is a truncated/garbled fragment, not usable as-is"),
    156: (1.0, "message asks a simple info question, reply is about personal info - irrelevant"),
    5: (3.5, "matches the parcel-left-unattended situation well"),
    433: (2.0, "vague reply, doesn't address the delivery urgency in the message"),
    323: (4.0, "matches the thank-you tone appropriately"),
    26: (1.5, "reply about promo offers doesn't match the vague frustration message"),
    269: (3.5, "reasonable on-topic follow-up"),
    57: (2.5, "partially on-topic (tracking) but doesn't address the cancellation ask"),
    340: (3.0, "reasonable next step (ask for carrier) for a missing item"),
    449: (2.5, "generic but plausible given a very vague message"),
    89: (2.5, "generic acknowledgement, doesn't address the refund refusal directly"),
    331: (1.5, "reply about a help page/daily deals is off-topic for a movie-availability question"),
    361: (3.0, "reasonable follow-up given customer already contacted support once"),
}


def main():
    df = pd.read_csv("eval/full_eval_results.csv")
    df = df[df["golden_id"].isin(HUMAN_SCORES.keys())].copy()
    df["human_overall"] = df["golden_id"].map(lambda i: HUMAN_SCORES[i][0])

    r_pearson, _ = pearsonr(df["judge_overall"], df["human_overall"])
    r_spearman, _ = spearmanr(df["judge_overall"], df["human_overall"])
    mae = (df["judge_overall"] - df["human_overall"]).abs().mean()

    print(f"n = {len(df)}")
    print(f"Pearson r  = {r_pearson:.3f}")
    print(f"Spearman r = {r_spearman:.3f}")
    print(f"Mean abs error (heuristic vs human, 1-5 scale) = {mae:.2f}")

    out = df[["golden_id", "message", "draft_reply", "judge_overall", "human_overall"]]
    out.to_csv("eval/human_agreement_detail.csv", index=False)


if __name__ == "__main__":
    main()
