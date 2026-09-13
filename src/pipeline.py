"""
The actual "AI support agent" - ties together intent classification,
escalation decisioning, and grounded reply drafting into one call.

Usage:
    python3 src/pipeline.py "my package says delivered but never arrived"
"""
import sys
import json

from classify import TfidfLogReg
from weak_label import classify_escalation
from draft_reply import ReplyRetriever, draft_reply

# Chosen via 5-fold CV on golden_train.csv only (never touched test), as the
# ~25th percentile of out-of-fold softmax confidence (~0.14). Even so, with
# 11 classes and ~130 training rows the classifier is poorly calibrated -
# see report.md failure mode #1 and "misleading headline number" section.
CONFIDENCE_ESCALATION_THRESHOLD = 0.15


class SupportAgent:
    def __init__(self):
        self.classifier = TfidfLogReg.load("data/model.joblib")
        self.retriever = ReplyRetriever("data/raw_sample.csv")

    def handle(self, message: str, golden_id=None) -> dict:
        pred, conf = self.classifier.predict_proba_top([message])
        intent, confidence = pred[0], conf[0]

        escalate, reason = classify_escalation(message, intent)
        low_conf = confidence < CONFIDENCE_ESCALATION_THRESHOLD
        if low_conf and not escalate:
            escalate = True
            reason = (f"classifier confidence {confidence:.2f} is below the "
                       f"{CONFIDENCE_ESCALATION_THRESHOLD} threshold - "
                       f"routing to a human rather than guessing")

        reply_info = draft_reply(message, intent, self.retriever,
                                  exclude_id=golden_id)

        return {
            "message": message,
            "intent": intent,
            "intent_confidence": round(confidence, 3),
            "escalate": escalate,
            "escalation_reason": reason,
            "draft_reply": reply_info["draft"],
            "reply_mode": reply_info["mode"],
            "grounded_on": [m["customer_clean"] for m in reply_info["grounded_on"]],
        }


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) or (
        "My order says delivered but I never received it, this is the "
        "third time this has happened")
    agent = SupportAgent()
    result = agent.handle(msg)
    print(json.dumps(result, indent=2))
