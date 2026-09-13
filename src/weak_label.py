"""
Weak/bootstrap labeler for the Amazon Twitter support golden set.

This is NOT the ground truth. It produces a first-pass label for every row
using keyword/regex heuristics. A human (the assignment author) then reviews
and corrects a stratified sample of these to build the real golden set
(see label_golden.py / decision_log.md, decision #3).

Intent taxonomy (defined by reading ~150 sampled rows of this dataset):
  1. delivery_issue        - late / lost / misdelivered / damaged parcel
  2. order_status           - "where is my order", tracking questions, ETA
  3. account_access         - login, password, 2FA, "can't sign in"
  4. billing_payment        - wrong charge, refund status, payment failed
  5. return_refund_request  - explicit ask to return/cancel/refund an item
  6. product_technical      - device/app not working (Echo, Fire TV, Kindle...)
  7. general_inquiry        - stock, price, how-to, Prime benefits, policy Qs
  8. complaint_escalation   - repeated failure / explicit anger / "unacceptable"
  9. positive_feedback      - praise, excitement, thanks, no action needed
  10. social_noise          - not actually a support request (memes, hashtags,
                              unrelated chatter the brand happened to reply to)
  11. other                 - doesn't fit cleanly anywhere above
"""
import re
import pandas as pd

INTENTS = [
    "delivery_issue", "order_status", "account_access", "billing_payment",
    "return_refund_request", "product_technical", "general_inquiry",
    "complaint_escalation", "positive_feedback", "social_noise", "other",
]

# Order matters: first matching rule wins. More specific / higher-signal
# patterns are checked first.
RULES = [
    ("account_access", re.compile(
        r"\b(log ?in|log ?on|sign ?in|password|2fa|two.factor|locked out|"
        r"can'?t access my account|hacked|account.*(hack|suspend|lock)|"
        r"no email linked to (my )?account)\b", re.I)),
    ("return_refund_request", re.compile(
        r"\b(refund|return this|return it|send it back|cancel my order|"
        r"want (a|my) refund|money back|exchange (this|it)|"
        r"(replace|replacement) (this|it|the item))\b", re.I)),
    ("billing_payment", re.compile(
        r"\b(charged|overcharged|double charge|wrong amount|payment (failed|"
        r"declined)|card was charged|billing|invoice|gift card.*(error|not "
        r"work)|paid extra|extra (rs|\$|charge)|promo code|price match)\b",
        re.I)),
    ("delivery_issue", re.compile(
        r"\b(never arrived|not arrived|didn'?t arrive|only one arrived|"
        r"hasn'?t arrived|late deliver|lost (package|parcel)|missing "
        r"(item|package|parcel)|left (it |the package )?(outside|on the|"
        r"with a neighbou?r|unattended)|damaged|wrong item|delivered to "
        r"(the )?wrong|(delivery )?driver|courier|carrier|logistics|"
        r"(collect|pick ?up) (it|my order)|shipment|deliver(y)? (fee|"
        r"charge)|fax (info|the))\b", re.I)),
    ("order_status", re.compile(
        r"\b(where is my order|track(ing)?|when (will|does|is) (it|my "
        r"order|this)|eta|due (on|today)|dispatch(ed)?|still (says|shows|"
        r"processing)|arrive in|arriv(e|ing)|deliver(ed)? (today|"
        r"tomorrow)|order (id|number|#)|urgently|priority)\b", re.I)),
    ("product_technical", re.compile(
        r"\b(echo|alexa|fire ?tv|kindle|app (crashes|won'?t|not working|"
        r"keeps|freezes)|error (code|message)|won'?t (turn on|connect|"
        r"load)|stream(ing)?|buffer|device|firmware)\b", re.I)),
    ("complaint_escalation", re.compile(
        r"\b(pathetic|useless|joke|sucks|terrible|worst|unacceptable|"
        r"disgust|fraud|scam|ridiculous|again and again|for the (second|"
        r"third|\d)th? time|still (not|no) (response|reply)|no one (has|"
        r"is) (helping|responding)|consumer court|frustrat\w*|f['’]?k)\b",
        re.I)),
    ("positive_feedback", re.compile(
        r"\b(thank(s| you)|love (it|amazon)|awesome|great service|happy "
        r"with|appreciate|you guys were great|👍|😍|❤️)\b", re.I)),
    ("general_inquiry", re.compile(
        r"\b(how do i|how can i|where (can|do) i find|is there a (way|"
        r"discount)|price|stock|available|prime (video|day|benefits)|"
        r"claim code|when (will|does) .* (release|launch|restock)|why "
        r"(is|does|not)|exchange (offer|value))\b", re.I)),
]

SOCIAL_NOISE_HINTS = re.compile(
    r"(^|[^a-z])(#\w+){1,}|😂😂|🙌|#[A-Za-z]+Day\b", re.I)

NEGATIVE_WORDS = re.compile(
    r"\b(angry|furious|frustrat\w*|disappoint\w*|awful|worst|pathetic|"
    r"unacceptable|sucks|joke|useless|fraud|scam|terrible|ridiculous)\b",
    re.I)

REPEAT_CONTACT = re.compile(
    r"\b(again|still (haven'?t|not|no)|twice|third time|multiple times|"
    r"keep(s)? (happening|telling)|already (told|contacted|emailed))\b",
    re.I)

ACCOUNT_SECURITY = re.compile(
    r"\b(hack(ed)?|unauthorized|fraud(ulent)? (charge|purchase)|stolen|"
    r"someone (else )?(used|accessed) my account)\b", re.I)


def classify_intent(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "other"
    if SOCIAL_NOISE_HINTS.search(text) and not re.search(
            r"\b(order|delivery|refund|account|package|parcel|charge)\b",
            text, re.I):
        return "social_noise"
    for intent, pattern in RULES:
        if pattern.search(text):
            return intent
    return "other"


def classify_escalation(text: str, intent: str):
    """Returns (escalate: bool, reason: str)."""
    if ACCOUNT_SECURITY.search(text):
        return True, "possible account security / unauthorized-charge issue " \
                      "— requires identity verification a bot can't perform"
    if intent == "complaint_escalation":
        return True, "customer expresses strong dissatisfaction / repeated " \
                      "failure — auto-reply risks making it worse"
    if REPEAT_CONTACT.search(text) and intent in (
            "delivery_issue", "billing_payment", "return_refund_request"):
        return True, "customer indicates this is a repeat/unresolved contact"
    if intent in ("account_access", "billing_payment"):
        return True, "involves account or payment data — needs verified " \
                      "human channel, not public/bot reply"
    if intent in ("order_status", "general_inquiry", "positive_feedback",
                   "social_noise"):
        return False, "low-risk, answerable from public info / templated " \
                       "response, no PII exchange needed"
    if intent == "delivery_issue":
        return False, "common, well-covered by standard delivery-issue " \
                       "playbook (apologize + request tracking link)"
    return False, "no strong signal for escalation; default to auto-handle " \
                   "with low-confidence flag"


def main():
    df = pd.read_csv("data/raw_sample.csv")
    df["intent"] = df["customer_clean"].apply(classify_intent)
    esc = df.apply(
        lambda r: classify_escalation(r["customer_clean"], r["intent"]),
        axis=1)
    df["escalate"] = esc.apply(lambda t: t[0])
    df["escalation_reason"] = esc.apply(lambda t: t[1])
    df.to_csv("data/weak_labels.csv", index=False)
    print(df["intent"].value_counts())
    print()
    print(df["escalate"].value_counts())


if __name__ == "__main__":
    main()
