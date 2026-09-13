import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

from pipeline import SupportAgent


def main():
    test = pd.read_csv("data/golden_test.csv")
    agent = SupportAgent()

    preds = []
    for _, row in test.iterrows():
        result = agent.handle(row["customer_clean"], golden_id=row["golden_id"])
        preds.append(result["escalate"])

    y_true = test["escalate"].tolist()
    y_pred = preds

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[True, False]).tolist()

    # A "trivial" baseline for escalation: never escalate (most rows in
    # this brand's traffic are auto-handleable, 149/200 = 74.5%)
    trivial_acc = accuracy_score(y_true, [False] * len(y_true))

    results = {
        "n_test": len(test),
        "our_system": {
            "accuracy": round(acc, 3),
            "precision_on_escalate": round(prec, 3),
            "recall_on_escalate": round(rec, 3),
            "f1_on_escalate": round(f1, 3),
            "confusion_matrix[[TP,FN],[FP,TN]]": cm,
        },
        "trivial_never_escalate_baseline_accuracy": round(trivial_acc, 3),
    }
    print(json.dumps(results, indent=2))
    with open("eval/escalation_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
