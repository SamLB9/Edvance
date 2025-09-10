from __future__ import annotations

from typing import Any, Dict, List
import json

from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL


SYS_PROMPT = (
    "You generate high-quality flash cards grounded ONLY in the provided content. "
    "Do NOT ask about formatting, pages, lines, digits, or filenames. "
    "Keep fronts concise and unambiguous; backs precise.\n"
    "If the content includes formulas or equations, you MUST produce formula-focused cards. "
    "Prefer accurate mathematical statements and concise explanations of each term.\n"
    "Return STRICT JSON only with this schema: {\n"
    "  \"cards\": [ { \n"
    "    \"type\": one of [\"term_def\", \"qa\", \"mcq\", \"tf\", \"cloze\"],\n"
    "    \"front\": \"...\", \n"
    "    \"back\": \"...\", \n"
    "    \"tags\": [\"...\"],\n"
    "    // Optional for MCQ: options: [\"A) ...\", \"B) ...\", \"C) ...\", \"D) ...\"], correct: \"A\"|[\"A\",\"C\"]\n"
    "    // Optional for TF: correct: \"True\" or \"False\" (back may include a one-line justification)\n"
    "  } ]\n"
    "}"
)


def _safe_json_loads(payload: str) -> Dict[str, Any]:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(payload[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise


def _normalize_text(s: str, max_words: int | None = None) -> str:
    text = (s or "").strip()
    if max_words:
        parts = text.split()
        if len(parts) > max_words:
            text = " ".join(parts[:max_words])
    return text


def generate_flashcards(context: str, topic: str, n: int = 12, allow_cloze: bool = True, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    typeline = "Use a mix of basic and cloze." if allow_cloze else "Use basic only."
    # Incorporate options into guidance (best-effort; engine still validates post hoc)
    mix_hint = ""
    if options and not options.get("auto_mix", True):
        cm = options.get("content_mix", {})
        try:
            total = sum(int(cm.get(k, 0)) for k in ("formula","definition","example","interpretation")) or 1
            p_formula = int(cm.get("formula", 0)) * 100 // total
            p_definition = int(cm.get("definition", 0)) * 100 // total
            p_example = int(cm.get("example", 0)) * 100 // total
            p_interp = int(cm.get("interpretation", 0)) * 100 // total
            mix_hint = (
                f"Target mix (percent of cards): formula {p_formula}%, definition {p_definition}%, example {p_example}%, interpretation {p_interp}%.\n"
            )
        except Exception:
            mix_hint = ""
    # Allowed card types
    type_hint = ""
    allowed_types = []
    if options and isinstance(options.get("card_types"), list) and options["card_types"]:
        allowed_types = [t for t in options["card_types"] if t in {"term_def","qa","mcq","tf","cloze"}]
    if allowed_types:
        human = {
            "term_def": "term/definition",
            "qa": "question→answer",
            "mcq": "multiple-choice",
            "tf": "true/false",
            "cloze": "cloze (fill-in-the-blank)",
        }
        type_hint = "Allowed card types: " + ", ".join(human[t] for t in allowed_types) + ".\n"

    # Length guidance
    len_hint = ""
    if options and isinstance(options.get("lengths"), dict):
        fr = str(options["lengths"].get("front", "medium")).lower()
        bk = str(options["lengths"].get("back", "medium")).lower()
        map_words = {"short": (10, 40), "medium": (18, 60), "long": (28, 90)}
        fmax, bmax = map_words.get(fr, (18, 60))[0], map_words.get(bk, (18, 60))[1]
        len_hint = f"Front ≤ {fmax} words; Back ≤ {bmax} words.\n"

    # Build the user prompt by concatenating strings to avoid f-string backslash issues
    latex_example = r"$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$"
    
    # Build the prompt in parts to avoid f-string backslash issues
    user_prompt_parts = [
        f"Topic hint: {topic or '(auto)'}\n",
        f"Content:\n---\n{context[:8000]}\n---\n",
        f"Create {n} cards. {typeline}\n",
        f"{mix_hint}",
        f"{type_hint}",
        "Rules:\n",
        "- Each card must test a concept from the content (definitions, key formulas, assumptions, implications, examples).\n",
        "- Forbidden: questions about document structure (pages, lines, fonts, numerals) or generic number trivia.\n",
        f"{len_hint or '- Front ≤ 18 words; Back ≤ 60 words.\n'}",
        "- Tags: 1–3 short tags (e.g., bayes, posterior, likelihood).\n",
        "- For cloze, use Anki syntax {{c1::...}} in the FRONT; BACK is a short elaboration.\n",
        "- If formulas are present (e.g., P(A|B), Bayes, sums/products, fractions), include formula-focused cards.\n",
        "  Prioritize: (1) the exact Bayes' theorem formula, (2) the multi-category version with a denominator sum, \n",
        "  (3) definitions of numerator/denominator terms, (4) a short worked example using the formula.\n",
        f"- You may include inline LaTeX for formulas using $...$ (e.g., {latex_example}).\n",
        "- Output only JSON. No extra text, no markdown, no additional keys."
    ]
    
    user_prompt = "".join(user_prompt_parts)
    resp = llm.invoke([
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    try:
        data = _safe_json_loads(resp.content)
    except Exception:
        return {"cards": []}

    raw_cards = data.get("cards", []) if isinstance(data, dict) else []
    clean: List[Dict[str, Any]] = []
    seen_fronts: set[str] = set()
    for c in raw_cards:
        if not isinstance(c, dict):
            continue
        # Filter by allowed types if provided
        ctype = str(c.get("type", "")).lower() or "qa"
        if allowed_types and ctype not in allowed_types:
            continue
        front = _normalize_text(str(c.get("front", "")), max_words=18)
        back = _normalize_text(str(c.get("back", "")), max_words=60)
        if not front or not back:
            continue
        key = " ".join(front.lower().split())
        if key in seen_fronts:
            continue
        seen_fronts.add(key)
        tags = c.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        # Flatten MCQ/TF into back content for export friendliness
        if ctype == "mcq":
            opts = c.get("options", []) or []
            correct = c.get("correct")
            lines = []
            if isinstance(opts, list) and opts:
                lines.append("Options:")
                for opt in opts:
                    lines.append(str(opt))
            if correct is not None:
                lines.append(f"Correct: {correct}")
            if lines:
                back = (back + "\n" + "\n".join(lines)).strip()
        elif ctype == "tf":
            correct_tf = c.get("correct")
            if correct_tf in ("True", "False"):
                back = (back + f"\nAnswer: {correct_tf}").strip()

        clean.append({
            "type": ctype if ctype in {"term_def","qa","mcq","tf","cloze"} else ("cloze" if c.get("type") == "cloze" else "qa"),
            "front": front,
            "back": back,
            "tags": [str(t)[:24] for t in tags[:5]],
        })

    return {"cards": clean[: max(1, int(n))]}


def to_tsv(cards: List[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for c in cards:
        tags = " ".join(c.get("tags", [])[:5])
        rows.append(f"{c.get('front','').strip()}\t{c.get('back','').strip()}\t{tags}")
    return "\n".join(rows)


def to_csv_quizlet(cards: List[Dict[str, Any]]) -> str:
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Term", "Definition"])
    for c in cards:
        w.writerow([c.get("front", "").strip(), c.get("back", "").strip()])
    return buf.getvalue()


def to_markdown(cards: List[Dict[str, Any]]) -> str:
    lines: List[str] = ["# Flash Cards"]
    for c in cards:
        tags = ", ".join(c.get("tags", []))
        lines.append(f"- **Q:** {c.get('front','').strip()}  \n  **A:** {c.get('back','').strip()}  \n  Tags: {tags}")
    return "\n".join(lines)


