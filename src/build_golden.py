import pandas as pd
from manual_corrections import CORRECTIONS

df = pd.read_csv("data/golden_candidates.csv")

rows = []
for _, r in df.iterrows():
    gid = int(r["golden_id"])
    if gid in CORRECTIONS:
        intent, escalate, reason = CORRECTIONS[gid]
    else:
        intent, escalate, reason = r["intent"], bool(r["escalate"]), r["escalation_reason"]
    rows.append({
        "golden_id": gid,
        "customer_clean": r["customer_clean"],
        "text_brand": r["text_brand"],
        "intent": intent,
        "escalate": escalate,
        "escalation_reason": reason,
        "label_source": "manual" if gid in CORRECTIONS else "rule_confirmed",
    })

out = pd.DataFrame(rows).sort_values("golden_id")
out.to_csv("data/golden_eval.csv", index=False)
print(len(out), "rows written to data/golden_eval.csv")
print(out["intent"].value_counts())
print()
print(out["escalate"].value_counts())
