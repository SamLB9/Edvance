from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, TypedDict, Union, Optional

from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL


# ---- Types ---------------------------------------------------------------
class MCQ(TypedDict):
    type: Literal["mcq"]
    prompt: str
    options: List[str]
    answer: str  # e.g., "A" or full text, depending on generator


class ShortQ(TypedDict):
    type: Literal["short"]
    prompt: str
    answer: str


class Quiz(TypedDict):
    questions: List[Union[MCQ, ShortQ]]


class GradeResult(TypedDict):
    correct: bool
    feedback: str


# ---- System Prompts -----------------------------------------------------
SYS_PROMPT = (
    "You are a strict but helpful study coach. Generate concise, unambiguous questions.\n"
    "Prefer 3–4 MCQs (with exactly 4 options labeled A–D) plus 1 short-answer.\n"
    "Return STRICT JSON only, with this exact schema: {\n"
    "  \"questions\": [ { \"type\": \"mcq\"|\"short\", \"prompt\": \"...\", \"options\": [\"A) ...\", \"B) ...\", \"C) ...\", \"D) ...\"]?, \"answer\": \"...\" } ]\n"
    "}. No markdown, no prose outside JSON."
)


# ---- Helpers ------------------------------------------------------------

def _safe_json_loads(payload: str) -> Dict[str, Any]:
    """Parse JSON safely; if it fails, try to extract the first JSON object or raise."""
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # naive recovery: find first '{' and last '}' and try again
        start = payload.find("{")
        end = payload.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(payload[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise


def _validate_quiz_schema(obj: Dict[str, Any]) -> Quiz:
    if not isinstance(obj, dict) or "questions" not in obj or not isinstance(obj["questions"], list):
        raise ValueError("Quiz JSON must have a 'questions' list.")
    for q in obj["questions"]:
        if not isinstance(q, dict) or q.get("type") not in ("mcq", "short"):
            raise ValueError("Each question must have type 'mcq' or 'short'.")
        if "prompt" not in q or not isinstance(q["prompt"], str) or not q["prompt"].strip():
            raise ValueError("Each question must have a non-empty 'prompt'.")
        if q["type"] == "mcq":
            opts = q.get("options")
            if not isinstance(opts, list) or len(opts) != 4:
                raise ValueError("MCQ must include exactly 4 options.")
        if "answer" not in q or not isinstance(q["answer"], str) or not q["answer"].strip():
            raise ValueError("Each question must include an 'answer'.")
    return obj  # type: ignore[return-value]


# ---- Public API ---------------------------------------------------------

def generate_quiz(
    context: str,
    topic: str,
    n_questions: int = 4,
    excluded_prompts: Optional[List[str]] = None,
    difficulty: Optional[str] = None,
) -> Quiz:
    """Generate a quiz as a parsed JSON object (dict) with schema Quiz.

    excluded_prompts: optional list of prompt strings that must not be repeated.
    difficulty: optional hint among {'easy','medium','hard'}.
    """
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)

    avoidance_instructions = ""
    if excluded_prompts:
        avoided = "\n".join(f"- {p}" for p in excluded_prompts[:50])
        avoidance_instructions = (
            "\nDo NOT repeat any of the following previously asked prompts. "
            "If a prompt is similar, create a clearly different question.\n"
            f"Avoid these prompts:\n{avoided}\n"
        )

    difficulty_instructions = ""
    if difficulty == "easy":
        difficulty_instructions = (
            "\nAdjust difficulty: The student is struggling. "
            "Generate straightforward, foundational questions. Favor recall and recognition over synthesis. "
            "Keep language simple, avoid multi-step reasoning, and avoid tricky distractors."
        )
    elif difficulty == "medium":
        difficulty_instructions = (
            "\nAdjust difficulty: The student is at an intermediate level. "
            "Generate a balanced mix of conceptual understanding and light reasoning. "
            "Include some multi-step items but keep them approachable; avoid excessive trickiness."
        )
    elif difficulty == "hard":
        difficulty_instructions = (
            "\nAdjust difficulty: The student is excelling. "
            "Generate challenging, reasoning-based questions that require 2–3 steps of inference using the provided context. "
            "Prefer questions that combine multiple facts/formulas, include subtle distractors (still unambiguous), and require applying concepts, not just recalling them."
        )

    user_prompt = (
        f"Topic: {topic}\n"
        f"Context (from course notes):\n---\n{context}\n---\n"
        f"Create {n_questions} questions that can be answered from the context."
        f"\nChosen difficulty: {difficulty or 'auto'}"
        f"{avoidance_instructions}"
        f"{difficulty_instructions}"
    )

    resp = llm.invoke([
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    try:
        data = _safe_json_loads(resp.content)
        return _validate_quiz_schema(data)
    except Exception:
        return {"questions": []}