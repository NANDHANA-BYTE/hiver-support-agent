"""
Hand-labels for the 40 rows the weak labeler dumped into 'other'
(data/golden_candidates.csv). Each was read individually; label + escalation
call + one-line reason recorded here. This is the actual "hand-labelled"
part of the golden set methodology (decision_log.md #3).
"""

CORRECTIONS = {
    65: ("delivery_issue", False, "standard non-delivery complaint, no new risk signal"),
    87: ("complaint_escalation", True, "profanity + repeat contact ('sick of calling in')"),
    88: ("general_inquiry", False, "asking for an ETA on an open case, no new issue"),
    92: ("order_status", False, "conflicting delivery-date info, clarifiable from tracking"),
    93: ("delivery_issue", False, "annoyed but a standard timing complaint"),
    100: ("billing_payment", True, "failed payment caused an order cancellation"),
    114: ("delivery_issue", False, "marked-delivered-but-missing, standard playbook"),
    141: ("complaint_escalation", True, "product quality + rude phone agent, reputational risk"),
    165: ("other", False, "no actionable support content"),
    168: ("complaint_escalation", True, "repeated bad replacement + large sum of money involved"),
    178: ("account_access", True, "email/account verification issue needs human check"),
    191: ("general_inquiry", False, "pricing question, publicly answerable"),
    192: ("account_access", True, "continuation of an account-access dispute"),
    195: ("billing_payment", False, "checkout price discrepancy, likely deal-terms explanation"),
    203: ("complaint_escalation", True, "hostile tone + legal threat, needs human de-escalation"),
    208: ("product_technical", False, "Prime Video playback bug, standard troubleshooting"),
    210: ("product_technical", False, "app bug on a specific device, standard troubleshooting"),
    225: ("other", False, "vague, no specific request"),
    233: ("general_inquiry", False, "how-to question, publicly answerable"),
    252: ("other", False, "no support content, just says they'll DM a screenshot"),
    255: ("complaint_escalation", True, "safety-adjacent product complaint + 'no support' claim"),
    308: ("delivery_issue", True, "wrong item shipped + strong anger, warrants human check"),
    309: ("complaint_escalation", False, "generic service feedback, no specific order to action"),
    310: ("complaint_escalation", True, "customer says phone line isn't responding, needs escalation"),
    312: ("account_access", True, "identity-verification mismatch blocking account access"),
    320: ("general_inquiry", False, "question about sale duration, publicly answerable"),
    325: ("complaint_escalation", True, "long-running repeat contact, relationship at risk"),
    342: ("delivery_issue", False, "implied wrong/unexpected item, standard wrong-item flow"),
    367: ("billing_payment", True, "dispute over being asked to pay for a service-center estimate"),
    369: ("complaint_escalation", True, "repeated courier failures + threat to cancel Prime"),
    370: ("delivery_issue", False, "recurring late delivery, but customer isn't hostile"),
    383: ("complaint_escalation", True, "dispute over a refused return, customer is angry"),
    404: ("delivery_issue", False, "wrong item shipped, standard wrong-item flow"),
    411: ("complaint_escalation", True, "repeated missing deliveries + threat to stop using service"),
    419: ("other", False, "fragment, not enough content to act on"),
    433: ("order_status", True, "time-sensitive gift delivery, needs a prioritized human check"),
    440: ("complaint_escalation", True, "30 days unresolved, customer explicitly losing trust"),
    470: ("complaint_escalation", True, "4-day silence after an unexplained cancellation"),
    476: ("return_refund_request", True, "pickup failed 3x despite calling customer service"),
    478: ("product_technical", False, "missing buy/cart option, a site bug not a policy issue"),
}

# Found during a 25-row random spot-check of the rule-labeled remainder
# (decision_log.md #4: quality check on the labeler, not just the fallback
# bucket). Six clear mislabels corrected here.
SPOT_CHECK_FIXES = {
    243: ("general_inquiry", False, "feature/catalog gap complaint, not a specific order issue"),
    228: ("billing_payment", True, "charged more than the stated price, needs investigation"),
    199: ("complaint_escalation", True, "faulty product + long replacement wait"),
    23: ("order_status", True, "already complained once with no action taken, repeat contact"),
    496: ("complaint_escalation", False, "vague fraud accusation, no specific order to action"),
    99: ("return_refund_request", True, "repeat non-delivery plus an explicit refund request"),
}
CORRECTIONS.update(SPOT_CHECK_FIXES)
