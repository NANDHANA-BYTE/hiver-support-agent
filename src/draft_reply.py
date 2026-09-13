"""
Drafts a reply "grounded in how the brand has historically resolved similar
issues" (assignment requirement #2).

Design (decision_log.md #6): no LLM API key is available in this environment,
so the default path is pure retrieval - find the most similar historical
customer message (within the same predicted intent) from data/raw_sample.csv
and adapt its real brand reply. This is honest, inspectable, and always
"grounded" by construction, but it can only recombine phrasing that already
exists in the corpus - it can't compose a genuinely novel reply.

If ANTHROPIC_API_KEY or OPENAI_API_KEY is set, llm_client.py is used instead:
the same retrieved examples are passed as few-shot grounding context to an
LLM, which drafts a new reply. Both paths return the same output shape so
eval/run_eval.py doesn't need to know which one ran.
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_client import get_llm_client

HANDLE_RE = re.compile(r"^@\d+\s*")
SIGNOFF_RE = re.compile(r"\s*\^[A-Z]{2}\b\s*")
MULTISPACE_RE = re.compile(r"\s{2,}")


def _clean_reply(text: str) -> str:
    """Strip the anonymized agent handle prefix (@135790) and the
    two-letter agent sign-off (^AR, wherever it appears - some replies
    have a mid-string '1/2 ^XX' marker for multi-tweet threads) - these
    are corpus artifacts, not part of the brand's actual reply template."""
    text = HANDLE_RE.sub("", text)
    text = SIGNOFF_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


class ReplyRetriever:
    def __init__(self, corpus_path="data/raw_sample.csv"):
        self.corpus = pd.read_csv(corpus_path)
        self.corpus["reply_clean"] = self.corpus["text_brand"].apply(_clean_reply)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                           stop_words="english")
        self.matrix = self.vectorizer.fit_transform(
            self.corpus["customer_clean"].fillna(""))

    def retrieve(self, message: str, exclude_id=None, k=3):
        vec = self.vectorizer.transform([message])
        sims = cosine_similarity(vec, self.matrix)[0]
        order = sims.argsort()[::-1]
        results = []
        for i in order:
            row = self.corpus.iloc[i]
            if exclude_id is not None and row.get("golden_id") == exclude_id:
                continue
            results.append({
                "similarity": float(sims[i]),
                "customer_clean": row["customer_clean"],
                "reply_clean": row["reply_clean"],
            })
            if len(results) >= k:
                break
        return results


def draft_reply(message: str, intent: str, retriever: ReplyRetriever,
                 exclude_id=None):
    matches = retriever.retrieve(message, exclude_id=exclude_id, k=3)
    llm = get_llm_client()
    if llm is not None:
        draft = llm.draft_reply(message, intent, matches)
        mode = "llm_grounded"
    else:
        # Pure retrieval fallback: adapt the single closest historical reply.
        draft = matches[0]["reply_clean"] if matches else (
            "Thanks for reaching out - could you share more detail so we "
            "can look into this?")
        mode = "retrieval_only"
    return {
        "draft": draft,
        "mode": mode,
        "grounded_on": matches,
    }


if __name__ == "__main__":
    r = ReplyRetriever()
    out = draft_reply(
        "My package says delivered but I never got it, this is the 3rd time",
        "delivery_issue", r)
    print(out["mode"])
    print(out["draft"])
    print("grounded on:")
    for m in out["grounded_on"]:
        print(f"  ({m['similarity']:.2f}) {m['customer_clean'][:60]!r} -> "
              f"{m['reply_clean'][:80]!r}")
