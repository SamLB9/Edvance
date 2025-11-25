from __future__ import annotations

from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL, OPENAI_API_KEY


def one_liner_from_context(ctx: str) -> str:
    """Produce a concise one-line subject from context text.

    Returns a sentence (<=16 words) describing the document's main subject, with no meta mentions.
    """
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
    sys = (
        "You write ultra-concise subject lines. "
        "Output a single sentence (<=16 words) describing the document’s main subject. "
        "No filename, no 'document/text', no formatting, and no page/line mentions."
    )
    # Limit context to avoid token overrun; keep first ~6000 chars which is plenty for a one-liner
    snippet = ctx[:6000] if isinstance(ctx, str) else ""
    usr = f"Content:\n---\n{snippet}\n---\nReturn only the sentence."
    resp = llm.invoke([
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ])
    text = (resp.content or "").strip()
    text = text.strip('"').strip()
    if not text.endswith("."):
        text = text + "."
    try:
        print(f'[QUIZ DEBUG] auto_topic: produced one-liner="{text}"')
    except Exception:
        pass
    return text


