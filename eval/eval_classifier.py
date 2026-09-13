import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from classify import TrivialBaseline, RuleBaseline, TfidfLogReg


def evaluate(model, name, train, test):
    model.fit(train["customer_clean"], train["intent"])
    preds = model.predict(test["customer_clean"])
    acc = accuracy_score(test["intent"], preds)
    f1 = f1_score(test["intent"], preds, average="macro", zero_division=0)
    report = classification_report(test["intent"], preds, zero_division=0,
                                    output_dict=True)
    return {
        "name": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "per_class_f1": {k: round(v["f1-score"], 3) for k, v in report.items()
                          if k not in ("accuracy", "macro avg", "weighted avg")},
    }


def main():
    train = pd.read_csv("data/golden_train.csv")
    test = pd.read_csv("data/golden_test.csv")

    results = [
        evaluate(TrivialBaseline(), "trivial_majority_class", train, test),
        evaluate(RuleBaseline(), "simple_rule_based", train, test),
        evaluate(TfidfLogReg(), "tfidf_logreg (ours)", train, test),
    ]

    print(f"{'model':<28}{'accuracy':<12}{'macro_f1':<12}")
    for r in results:
        print(f"{r['name']:<28}{r['accuracy']:<12}{r['macro_f1']:<12}")

    with open("eval/classifier_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nfull results written to eval/classifier_results.json")

    # --- Circularity check: the rule baseline was ALSO used to bootstrap
    # ~77% of the golden labels (label_source == 'rule_confirmed'), so
    # scoring it on those rows is partly grading it against its own output.
    # Re-score on manually-corrected rows only for a fairer read.
    manual_test = test[test["label_source"] == "manual"]
    if len(manual_test) >= 5:
        print(f"\n--- fairness check: manual-only test rows (n={len(manual_test)}) ---")
        for cls, name in [(RuleBaseline(), "simple_rule_based"),
                           (TfidfLogReg(), "tfidf_logreg (ours)")]:
            cls.fit(train["customer_clean"], train["intent"])
            preds = cls.predict(manual_test["customer_clean"])
            acc = accuracy_score(manual_test["intent"], preds)
            f1 = f1_score(manual_test["intent"], preds, average="macro",
                           zero_division=0)
            print(f"{name:<28}accuracy={acc:.3f}  macro_f1={f1:.3f}")


if __name__ == "__main__":
    main()
