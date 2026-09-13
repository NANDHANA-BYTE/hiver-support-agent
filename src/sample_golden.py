"""
Builds the candidate golden set (data/golden_candidates.csv) via stratified
sampling over the weak-labeled, English-only rows.

Scoping decision (see decision_log.md #1): golden set and classifier are
scoped to English-language tweets (365/500 = 73% of this sample). Non-English
rows are kept in data/raw_sample.csv but excluded from evaluation.

Sampling strategy (decision_log.md #3): take ALL rows from rare weak-label
classes (account_access, billing_payment, complaint_escalation,
product_technical, return_refund_request) since they're the hardest to get
enough examples of; take all order_status/delivery_issue/general_inquiry;
cap positive_feedback and social_noise (low information for grading a
support agent); and include a random sample of the large 'other' bucket
specifically because that's where the regex weak-labeler is most likely to
be wrong -- every one of those gets manually re-read and corrected.
"""
import pandas as pd

RARE = ["account_access", "billing_payment", "complaint_escalation",
        "product_technical", "return_refund_request"]
FULL = ["order_status", "delivery_issue", "general_inquiry"]
CAPPED = {"positive_feedback": 20, "social_noise": 15}
OTHER_SAMPLE_N = 40
SEED = 42


def main():
    wl = pd.read_csv("data/weak_labels.csv")
    lg = pd.read_csv("data/with_lang.csv")
    df = wl.copy()
    df["lang"] = lg["lang"]
    en = df[df["lang"] == "en"].copy()

    parts = []
    parts.append(en[en["intent"].isin(RARE)])
    parts.append(en[en["intent"].isin(FULL)])
    for intent, n in CAPPED.items():
        sub = en[en["intent"] == intent]
        parts.append(sub.sample(n=min(n, len(sub)), random_state=SEED))
    other = en[en["intent"] == "other"]
    parts.append(other.sample(n=min(OTHER_SAMPLE_N, len(other)),
                               random_state=SEED))

    candidates = pd.concat(parts).drop_duplicates(
        subset="golden_id").sample(frac=1, random_state=SEED)
    candidates.to_csv("data/golden_candidates.csv", index=False)
    print(f"{len(candidates)} candidate rows written to "
          f"data/golden_candidates.csv")
    print(candidates["intent"].value_counts())


if __name__ == "__main__":
    main()
