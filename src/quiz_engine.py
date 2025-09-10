from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, TypedDict, Union, Optional

from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL


# ---- Types ---------------------------------------------------------------
class QuizQuestion(TypedDict, total=False):
    type: Literal[
        "mcq_single",
        "mcq_multi",
        "tf",
        "tf_justify",
        "short",
        "matching",
        "ordering",
        "numeric",
        "explain",
    ]
    prompt: str
    answer: str
    options: List[str]
    left: List[str]
    right: List[str]
    items: List[str]


class Quiz(TypedDict):
    questions: List[QuizQuestion]


class GradeResult(TypedDict):
    correct: bool
    feedback: str


# ---- System Prompts -----------------------------------------------------
SYS_PROMPT = (
    "You are a strict but helpful study coach. Generate concise, unambiguous questions that assess the SUBJECT MATTER only.\n"
    "Absolutely forbid meta-questions about formatting, file structure, or how the context is written.\n"
    "NEVER ask about lines, pages, headings, fonts, bullets, figure numbers, filenames, encodings, separators like '---', or JSON structure.\n"
    "Only ask about concepts, definitions, theorems, procedures, examples, results, formulas, and interpretations contained in the content.\n"
    "Forbidden examples: 'True or False: Each line shows the same digit', 'How many bullet points are there?', 'What is written in the first line?', 'Which page mentions ...?'.\n"
    "Allowed examples: 'State Bayes' theorem and interpret each term', 'Which conditions are required for the law of large numbers?', 'Compute P(A|B=...) given ...', 'Explain why independence implies ...'.\n"
    "Return STRICT JSON only with this schema: {\n"
    "  \"questions\": [ {\n"
    "    \"type\": one of [\"mcq_single\", \"mcq_multi\", \"tf\", \"tf_justify\", \"short\", \"matching\", \"ordering\", \"numeric\", \"explain\"],\n"
    "    \"prompt\": string (must not mention 'context', 'text above', 'lines', 'pages', or formatting),\n"
    "    \"answer\": string (reference answer for grading),\n"
    "    // Additional fields by type:\n"
    "    // mcq_single: options: [\"A) ...\", \"B) ...\", \"C) ...\", \"D) ...\"]\n"
    "    // mcq_multi: options: [\"...\"], and answer MUST list correct option texts separated by '; ' (e.g., \"Option 1; Option 3\")\n"
    "    // tf: no extras, answer is 'True' or 'False'\n"
    "    // tf_justify: no extras, answer is 'True — <brief justification>'\n"
    "    // short: no extras\n"
    "    // matching: left: [labels], right: [choices]; answer should be mapping like 'A -> 2; B -> 1; C -> 3'\n"
    "    // ordering: items: [list to order]; answer is the correct order as '1) X, 2) Y, 3) Z'\n"
    "    // numeric: no extras; if units/tolerance matter, include them within the answer (e.g., '3.14 (±0.01), units: rad')\n"
    "    // explain: no extras; answer lists key points expected\n"
    "  } ]\n"
    "}. No markdown, no commentary outside JSON."
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
        if not isinstance(q, dict):
            raise ValueError("Each question must be an object.")
        qtype = q.get("type")
        if qtype not in {"mcq_single","mcq_multi","tf","tf_justify","short","matching","ordering","numeric","explain","mcq"}:
            raise ValueError("Unsupported question type.")
        if "prompt" not in q or not isinstance(q["prompt"], str) or not q["prompt"].strip():
            raise ValueError("Each question must have a non-empty 'prompt'.")
        if qtype in {"mcq_single","mcq_multi","mcq"}:
            opts = q.get("options")
            if not isinstance(opts, list) or len(opts) < 4:
                raise ValueError("MCQ must include options (≥4).")
        if qtype == "matching":
            left = q.get("left")
            right = q.get("right")
            if not isinstance(left, list) or not isinstance(right, list) or not left or not right:
                raise ValueError("Matching must include 'left' and 'right' lists.")
        if qtype == "ordering":
            items = q.get("items")
            if not isinstance(items, list) or len(items) < 3:
                raise ValueError("Ordering must include an 'items' list (≥3).")
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
    question_mix_counts: Optional[Dict[str, int]] = None,
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

    mix_instructions = ""
    if question_mix_counts:
        chosen = {k: int(v) for k, v in question_mix_counts.items() if int(v) > 0}
        if chosen:
            mix_lines = "\n".join(f"- {k}: {v}" for k, v in chosen.items())
            mix_instructions = (
                "\nQuestion type counts (strict):\n" + mix_lines + "\n"
                "Ensure the total number of questions equals the requested count."
            )

    user_prompt = (
        f"Topic: {topic}\n"
        f"Context (from course notes):\n---\n{context}\n---\n"
        f"Create {n_questions} questions that are answerable using only the subject-matter content above."
        f"\nRules:\n"
        f"- Ignore formatting artifacts (line breaks, bullets, numbering, page headers/footers, file names, code fences, base64, JSON, separators like '---').\n"
        f"- Do not write questions about the structure/format of the text; focus on concepts, results, definitions, proofs, computations, and interpretations present in the content.\n"
        f"- Do not reference 'the context', 'the text above', 'this document', or page/line numbers in prompts.\n"
        f"- Prefer precise, content-anchored prompts and unambiguous answers."
        f"\nChosen difficulty: {difficulty or 'auto'}"
        f"{mix_instructions}"
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