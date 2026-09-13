"""
Optional LLM client. If ANTHROPIC_API_KEY or OPENAI_API_KEY is set in the
environment, get_llm_client() returns a working client that:
  - drafts replies grounded in retrieved historical examples (few-shot)
  - acts as the LLM-as-judge for reply-quality scoring (eval/llm_judge.py)

If no key is present, get_llm_client() returns None and callers fall back to
non-LLM logic (pure retrieval for drafting, heuristic rubric for judging).
This repo is fully runnable with zero API keys - that's a deliberate
reproducibility decision, see decision_log.md #6.
"""
import os


SYSTEM_PROMPT = (
    "You are a support agent for {brand} replying on Twitter. Write ONE "
    "short reply (under 280 characters) to the customer message below. "
    "Match the brand's real historical tone and resolution pattern shown "
    "in the examples - do not invent policies, refunds, or links that "
    "aren't implied by the examples. Do not use a customer's real name "
    "unless given. Output only the reply text, nothing else."
)


class AnthropicLLM:
    def __init__(self, api_key, brand="Amazon"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.brand = brand

    def _examples_block(self, matches):
        lines = []
        for m in matches:
            lines.append(f"Customer: {m['customer_clean']}\n"
                          f"Agent reply: {m['reply_clean']}")
        return "\n\n".join(lines)

    def draft_reply(self, message, intent, matches):
        prompt = (
            f"Intent: {intent}\n\n"
            f"Similar historical cases:\n{self._examples_block(matches)}\n\n"
            f"New customer message: {message}\n\nReply:"
        )
        resp = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=SYSTEM_PROMPT.format(brand=self.brand),
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        ).strip()

    def judge(self, message, draft, rubric_prompt):
        resp = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": rubric_prompt}],
        )
        return "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        ).strip()


def get_llm_client(brand="Amazon"):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            return AnthropicLLM(key, brand=brand)
        except ImportError:
            return None
    # OPENAI_API_KEY support could be added the same way; omitted here
    # since we don't have a key to test it against (decision_log.md #6).
    return None
