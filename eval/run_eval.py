import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pandas as pd

from pipeline import SupportAgent
from llm_judge import judge_reply


def main():
    test = pd.read_csv("data/golden_test.csv")
    agent = SupportAgent()

    rows = []
    for _, r in test.iterrows():
        result = agent.handle(r["customer_clean"], golden_id=r["golden_id"])
        scores = judge_reply(r["customer_clean"], result["draft_reply"],
                              result["grounded_on"])
        rows.append({
            "golden_id": r["golden_id"],
            "message": r["customer_clean"],
            "true_intent": r["intent"],
            "pred_intent": result["intent"],
            "true_escalate": bool(r["escalate"]),
            "pred_escalate": result["escalate"],
            "draft_reply": result["draft_reply"],
            **{f"judge_{k}": v for k, v in scores.items()},
        })

    out = pd.DataFrame(rows)
    out.to_csv("eval/full_eval_results.csv", index=False)

    summary = {
        "n": len(out),
        "intent_accuracy": round((out["true_intent"] == out["pred_intent"]).mean(), 3),
        "escalation_accuracy": round((out["true_escalate"] == out["pred_escalate"]).mean(), 3),
        "avg_judge_overall": round(out["judge_overall"].mean(), 2),
        "avg_judge_grounded": round(out["judge_grounded"].mean(), 2),
        "avg_judge_relevance": round(out["judge_relevance"].mean(), 2),
        "avg_judge_tone": round(out["judge_tone"].mean(), 2),
        "avg_judge_actionability": round(out["judge_actionability"].mean(), 2),
        "judge_mode": out["judge_judge_mode"].iloc[0],
    }
    print(json.dumps(summary, indent=2))
    with open("eval/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nrow-level results: eval/full_eval_results.csv")


if __name__ == "__main__":
    main()
