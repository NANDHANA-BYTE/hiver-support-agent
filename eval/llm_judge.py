"""
LLM-as-judge for reply quality (assignment deliverable #3).

If an API key is present (see src/llm_client.py), judge_reply() sends the
rubric below to the LLM and parses its 1-5 scores. Without a key, it falls
back to a heuristic proxy scorer that approximates the same rubric with
cheap signals (grounding overlap, length, politeness markers, presence of
a concrete next step). The fallback is intentionally simple and is
validated against manual human scoring in eval/human_agreement.py -
see report.md for the agreement numbers and why the proxy is a weak
substitute for a real LLM judge.

Rubric (both modes score against this):
  - grounded (1-5): does the reply's content match what similar historical
    resolutions actually did, rather than inventing a new policy?
  - relevance (1-5): does it actually address the customer's message?
  - tone (1-5): polite, on-brand, not robotic-sounding?
  - actionability (1-5): does it give the customer a concrete next step
    when one is needed (vs. a dead-end non-answer)?
"""
import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_client import get_llm_client

POLITE_MARKERS = re.compile(
    r"\b(sorry|apolog|thank|happy to|glad to|please|we'd like|we would "
    r"like)\b", re.I)
NEXT_STEP_MARKERS = re.compile(
    r"(https?://|please (provide|share|reach|contact|dm|call)|let us "
    r"know|reach out|send us)", re.I)


def _word_overlap(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-z]+", a.lower()))
    wb = set(re.findall(r"[a-z]+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def heuristic_judge(message: str, draft: str, grounded_on: list) -> dict:
    ground_text = " ".join(m for m in grounded_on) if grounded_on else ""
    grounded_overlap = _word_overlap(draft, ground_text)
    grounded_score = 1 + round(4 * min(grounded_overlap * 4, 1.0))

    relevance = 1 + round(4 * min(_word_overlap(message, draft) * 6, 1.0))

    tone = 3
    if POLITE_MARKERS.search(draft):
        tone += 1
    if len(draft) > 280 or len(draft) < 10:
        tone -= 1
    tone = max(1, min(5, tone))

    actionability = 3
    if NEXT_STEP_MARKERS.search(draft):
        actionability += 2
    actionability = max(1, min(5, actionability))

    return {
        "grounded": grounded_score,
        "relevance": relevance,
        "tone": tone,
        "actionability": actionability,
        "overall": round((grounded_score + relevance + tone + actionability) / 4, 2),
        "judge_mode": "heuristic_proxy",
    }


def judge_reply(message: str, draft: str, grounded_on: list) -> dict:
    llm = get_llm_client()
    if llm is None:
        return heuristic_judge(message, draft, grounded_on)
    rubric_prompt = f"""Score this customer support reply on 4 dimensions,
1-5 each. Return ONLY a JSON object with keys grounded, relevance, tone,
actionability (integers 1-5).

Customer message: {message}
Similar historical resolutions: {grounded_on}
Draft reply: {draft}
"""
    raw = llm.judge(message, draft, rubric_prompt)
    import json
    try:
        scores = json.loads(raw)
        scores["overall"] = round(sum(
            scores[k] for k in ("grounded", "relevance", "tone",
                                 "actionability")) / 4, 2)
        scores["judge_mode"] = "llm"
        return scores
    except Exception:
        return heuristic_judge(message, draft, grounded_on)
