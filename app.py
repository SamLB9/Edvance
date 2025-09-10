import streamlit as st
import os
from pathlib import Path
import time
import base64
import threading
import warnings
from typing import List, Dict, Any

# Suppress warnings for better performance
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

from src.ingest import load_documents, chunk_documents
from src.retriever import build_or_load_vectorstore, retrieve_context
from src.quiz_engine import generate_quiz
from src.evaluation import grade_answer
from src.memory import JsonMemory
from src.flashcards_engine import generate_flashcards, to_tsv, to_csv_quizlet, to_markdown
from src.performance_config import PDF_PREVIEW_CACHE_LIMIT, DISABLE_DEBUG_OUTPUT

NOTES_DIR = Path("data/notes")
PROGRESS_PATH = Path("progress.json")

# Global flag for background pre-loading
_preload_started = False

def _preload_embeddings():
    """Pre-load embeddings in background thread"""
    global _preload_started
    if not _preload_started:
        _preload_started = True
        try:
            # Pre-load embeddings to warm up the cache
            from src.retriever import _get_embeddings
            _get_embeddings()
        except Exception:
            pass  # Ignore errors in background thread


def ensure_notes_dir() -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_files(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Path]:
    saved = []
    ensure_notes_dir()
    for uf in files:
        dest = NOTES_DIR / uf.name
        dest.write_bytes(uf.getbuffer())
        saved.append(dest)
    # Clear notes cache when files are uploaded
    _list_notes_cached.clear()
    return saved


@st.cache_data(ttl=300)  # Cache for 5 minutes
def _list_notes_cached() -> List[Dict[str, Any]]:
    """Cached notes listing"""
    ensure_notes_dir()
    rows: List[Dict[str, Any]] = []
    for p in sorted(NOTES_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}:
            try:
                stat = p.stat()
                rows.append({
                    "file": str(p.relative_to(NOTES_DIR)),
                    "size_kb": int(round(stat.st_size / 1024)),
                    "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
                })
            except Exception:
                rows.append({"file": str(p.relative_to(NOTES_DIR)), "size_kb": None, "modified": None})
    return rows

def list_notes() -> List[Dict[str, Any]]:
    # Use cached version for better performance
    return _list_notes_cached()


def _render_brand_header() -> None:
    """Render a compact brand header with logo at top-left and title inline."""
    logo_img_tag = ""
    try:
        if os.path.exists("16.png"):
            with open("16.png", "rb") as lf:
                encoded = base64.b64encode(lf.read()).decode()
            # Keep logo modest so it aligns well with the H1 baseline
            logo_img_tag = f'<img src="data:image/png;base64,{encoded}" alt="Edvance logo" style="height:48px;" />'
    except Exception:
        logo_img_tag = ""
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between; padding:6px;">
          <div style="display:flex; align-items:center; gap:16px;">
            {logo_img_tag}
            <h1 style="margin:0;">Edvance Study Coach</h1>
          </div>
          <div style="font-size:16px; font-weight:600; color:#bbb;">Anticipated Access</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource
def _load_vectorstore():
    """Cached vector store loading"""
    return build_or_load_vectorstore([])

def ensure_vectorstore_loaded() -> None:
    """Lazy load vector store only when needed"""
    if "vs" not in st.session_state:
        # Only show spinner if we're actually loading
        if not st.session_state.get("vs_loading", False):
            st.session_state["vs_loading"] = True
            with st.spinner("Loading vector store..."):
                st.session_state.vs = _load_vectorstore()
            st.session_state["vs_loading"] = False
        else:
            # If already loading, just wait
            st.session_state.vs = _load_vectorstore()


def reset_quiz_state() -> None:
    """Reset only quiz-related state, not summary state"""
    st.session_state.quiz_started = False
    st.session_state.questions = []
    st.session_state.current_idx = 0
    st.session_state.answers = []
    st.session_state.response_ms = []
    st.session_state.feedbacks = []
    st.session_state.correct_count = 0
    st.session_state.q_start = None
    st.session_state.step = "question"
    st.session_state.last_feedback = None
    st.session_state.last_took_ms = 0
    st.session_state.last_q = None
    st.session_state.busy = False
    # Clear quiz-specific state
    st.session_state.pop("quiz_generating", None)
    st.session_state.pop("quiz_pending_start", None)


def ensure_quiz_state_initialized() -> None:
    """Initialize quiz-related session_state keys only if missing.
    Prevents accidental resets when navigating between tabs (Streamlit reruns).
    """
    defaults = {
        "quiz_started": False,
        "questions": [],
        "current_idx": 0,
        "answers": [],
        "response_ms": [],
        "feedbacks": [],
        "correct_count": 0,
        "q_start": None,
        "step": "question",
        "last_feedback": None,
        "last_took_ms": 0,
        "last_q": None,
        "busy": False,
        # Inputs
        "quiz_topic": "",
        "avoid_mode": "all",
        "feedback_mode": "immediate",
        "difficulty": None,
        "quiz_n": 4,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Debug log buffer for quiz flow
    if "quiz_debug_log" not in st.session_state:
        st.session_state["quiz_debug_log"] = []

def _quiz_debug(message: str) -> None:
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {message}"
    try:
        print(f"[QUIZ DEBUG] {line}")
    except Exception:
        pass
    try:
        # keep last 200 lines
        buf = st.session_state.get("quiz_debug_log", [])
        buf.append(line)
        st.session_state["quiz_debug_log"] = buf[-200:]
    except Exception:
        pass

def start_quiz(topic: str, n: int, avoid_mode: str, feedback_mode: str, question_mix_counts: dict | None = None, reference_file: str | None = None) -> None:
    ensure_vectorstore_loaded()
    memory = JsonMemory(str(PROGRESS_PATH))

    # Retrieve context (filter to selected reference file if provided)
    _quiz_debug(f"start_quiz: n={n}, avoid_mode={avoid_mode}, feedback_mode={feedback_mode}")
    _quiz_debug(f"start_quiz: reference_file={(reference_file or 'None')}")
    ctx = retrieve_context(st.session_state.vs, topic, k=8, source_path=reference_file)
    _quiz_debug(f"start_quiz: retrieved context length={len(ctx)} chars")
    # If no topic provided, infer pseudo-topic from context
    _topic = topic.strip()
    if not _topic:
        try:
            from src.auto_topic import one_liner_from_context
            _topic = one_liner_from_context(ctx)
            _quiz_debug(f'start_quiz: inferred pseudo-topic="{_topic}"')
        except Exception:
            _topic = "Key concepts and results"
            _quiz_debug("start_quiz: pseudo-topic inference failed; using fallback topic")
    topic = _topic
    if not ctx.strip():
        st.warning("No relevant context retrieved. Try uploading notes and rebuilding.")
        _quiz_debug("start_quiz: empty context; aborting")
        return

    # Exclusions & difficulty
    excluded = memory.get_excluded_prompts(mode=avoid_mode, topic=topic)
    _quiz_debug(f"start_quiz: excluded_prompts_count={len(excluded) if excluded else 0}")
    # Prefer user-chosen difficulty if provided, else adapt from history
    chosen = st.session_state.get("difficulty")
    difficulty = chosen if chosen in {"easy", "medium", "hard"} else memory.get_adaptive_difficulty(topic)
    _quiz_debug(f"start_quiz: chosen_difficulty={difficulty}")

    # Generate quiz
    quiz = generate_quiz(
        ctx,
        topic,
        n_questions=n,
        excluded_prompts=excluded,
        difficulty=difficulty,
        question_mix_counts=question_mix_counts,
    )
    _quiz_debug(f"start_quiz: quiz generation returned type={type(quiz).__name__}")
    questions = quiz.get("questions", []) if isinstance(quiz, dict) else []
    if not questions:
        st.warning("Quiz generation returned no questions. Try refining the topic or rebuilding the vector store.")
        _quiz_debug("start_quiz: no questions returned")
        return

    # Initialize quiz state
    st.session_state.quiz_started = True
    st.session_state.topic = topic
    st.session_state.avoid_mode = avoid_mode
    st.session_state.feedback_mode = feedback_mode
    st.session_state.difficulty = difficulty
    st.session_state.questions = questions
    st.session_state.current_idx = 0
    st.session_state.answers = [""] * len(questions)
    st.session_state.response_ms = [0] * len(questions)
    st.session_state.feedbacks = [None] * len(questions)
    st.session_state.correct_count = 0
    st.session_state.q_start = None
    st.session_state.step = "question"
    st.session_state.last_feedback = None
    st.session_state.last_took_ms = 0
    st.session_state.last_q = None
    st.session_state.busy = False


def render_question(i: int, q: Dict[str, Any]) -> str:
    st.write(f"Q{i+1}. {q['prompt']}")
    answer = st.session_state.answers[i]
    disabled = st.session_state.busy or (st.session_state.feedbacks[i] is not None)
    qtype = q.get("type")

    if qtype in ("mcq_single", "mcq"):  # legacy 'mcq' treated as single
        options = q.get("options", [])
        placeholder = "— Select an option —"
        choices = [placeholder] + options
        idx = choices.index(answer) if answer in options else 0
        selected = st.radio("Choose an option:", choices, index=idx, key=f"mcq_single_{i}", disabled=disabled)
        return "" if selected == placeholder else selected

    if qtype == "mcq_multi":
        options = q.get("options", [])
        preselected = [opt for opt in options if opt in (answer.split("; ") if isinstance(answer, str) else [])]
        selected = st.multiselect("Select all that apply:", options, default=preselected, key=f"mcq_multi_{i}", disabled=disabled)
        return "; ".join(selected)

    if qtype == "tf":
        choices = ["True", "False"]
        idx = choices.index(answer) if answer in choices else 0
        selected = st.radio("Pick one:", choices, index=idx, key=f"tf_{i}", disabled=disabled)
        return selected

    if qtype == "tf_justify":
        choices = ["True", "False"]
        base = answer.split(" — ")[0] if isinstance(answer, str) and " — " in answer else (answer if answer in choices else "True")
        idx = choices.index(base) if base in choices else 0
        selected = st.radio("Pick one:", choices, index=idx, key=f"tfj_{i}", disabled=disabled)
        just = st.text_area("Brief justification:", value=(answer.split(" — ")[-1] if " — " in str(answer) else ""), key=f"tfjjust_{i}", height=80, disabled=disabled)
        return f"{selected} — {just.strip()}" if just.strip() else selected

    if qtype in ("short", "explain"):
        return st.text_area("Your answer:", value=answer, key=f"short_{i}", height=120, disabled=disabled)

    if qtype == "numeric":
        return st.text_input("Enter a number (you may include units):", value=answer, key=f"num_{i}", disabled=disabled)

    if qtype == "matching":
        left = q.get("left", [])
        right = q.get("right", [])
        st.caption("Match each left item to one right choice.")
        selections: List[str] = []
        for idx_l, left_item in enumerate(left):
            sel = st.selectbox(f"{left_item}", options=["—"] + right, key=f"match_{i}_{idx_l}", disabled=disabled)
            if sel and sel != "—":
                selections.append(f"{left_item} -> {sel}")
        return "; ".join(selections)

    if qtype == "ordering":
        items = q.get("items", [])
        st.caption("Provide the order as comma-separated indices (e.g., 2,1,3).")
        for idx_o, it in enumerate(items, start=1):
            st.write(f"{idx_o}. {it}")
        return st.text_input("Order:", value=answer, key=f"ord_{i}", disabled=disabled)

    # Fallback
    return st.text_area("Your answer:", value=answer, key=f"generic_{i}", height=100, disabled=disabled)


def render_feedback(i: int, q: Dict[str, Any], result: Dict[str, Any], took_ms: int) -> None:
    is_correct = bool(result.get("correct"))
    status = "✅ Correct" if is_correct else "❌ Incorrect"
    feedback = result.get("feedback", "")
    st.markdown(f"**{status}**  •  {took_ms} ms")
    if feedback:
        st.write(feedback)
    # Show reference answer for objective types
    if q.get("type") in {"mcq_single", "mcq_multi", "mcq", "tf", "tf_justify", "numeric", "matching", "ordering"}:
        st.caption(f"Reference answer: {q.get('answer','')}")


def grade_and_log(i: int, q: Dict[str, Any], student_answer: str, took_ms: int) -> Dict[str, Any]:
    result = grade_answer(q['prompt'], q['answer'], student_answer)
    memory = JsonMemory(str(PROGRESS_PATH))
    memory.log_attempt(
        topic=st.session_state.topic,
        prompt=q['prompt'],
        student_answer=student_answer,
        correct=bool(result.get('correct')),
        response_ms=took_ms,
    )
    return result


def quiz_tab():
    st.subheader("Quiz")
    
    # Clear any summary-related state when entering quiz tab
    st.session_state.pop("summary_generating", None)
    st.session_state.pop("summary_pending", None)
    
    # Spinner below handles transient loading; avoid extra banner at top
    # Auto-recover quiz_started flag if questions persist (after tab switches)
    if (not st.session_state.get("quiz_started", False)) and st.session_state.get("questions"):
        st.session_state.quiz_started = True
    quiz_started = st.session_state.get("quiz_started", False)

    # Optional reference PDF/Notes preview (for context while answering)
    notes = list_notes()
    if notes:
        file_names = [note["file"] for note in notes]
        try:
            default_idx = file_names.index(st.session_state.get("quiz_reference_file", file_names[0]))
        except Exception:
            default_idx = 0
        reference_file = st.selectbox(
            "Reference PDF/Notes (optional)",
            file_names,
            index=default_idx,
            key="quiz_reference_file",
            help="This preview is for reference only and does not affect the quiz generation."
        )
        ref_path = NOTES_DIR / reference_file
        with st.expander("📄 PDF preview", expanded=False):
            try:
                if ref_path.suffix.lower() == ".pdf" and ref_path.exists():
                    # Cache PDF base64 to avoid repeated encoding
                    cache_key = f"pdf_preview_{ref_path.name}_{ref_path.stat().st_mtime}"
                    if cache_key not in st.session_state:
                        with open(ref_path, "rb") as f:
                            _bytes = f.read()
                        import base64 as _b64
                        b64 = _b64.b64encode(_bytes).decode()
                        st.session_state[cache_key] = b64
                    else:
                        b64 = st.session_state[cache_key]
                    
                    iframe = f"""
                    <iframe src=\"data:application/pdf;base64,{b64}\" 
                            width=\"100%\" height=\"1100px\" 
                            style=\"border: 1px solid #e0e0e0; border-radius: 6px;\"></iframe>
                    """
                    st.markdown(iframe, unsafe_allow_html=True)
                elif ref_path.exists():
                    try:
                        text = ref_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        text = "(Could not read file as text)"
                    st.code(text[:5000], language="markdown")
                else:
                    st.caption("Selected reference file not found.")
            except Exception as _e:
                st.caption(f"Preview unavailable: {_e}")

    # Reactive topic input: updates session_state on each keystroke
    # Use a separate widget key and persist to a stable state key to survive tab switches
    _persisted_topic = st.session_state.get("quiz_topic", "")
    st.text_input("Topic", value=_persisted_topic, key="quiz_topic_input", disabled=quiz_started)
    # Mirror widget value into persistent key every rerun (allowed: different key than widget)
    st.session_state["quiz_topic"] = st.session_state.get("quiz_topic_input", _persisted_topic)
    topic = st.session_state["quiz_topic"]
    MAX_Q = 20
    cols = st.columns(4)
    with cols[0]:
        n = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=MAX_Q,
            value=st.session_state.get("quiz_n", 4),
            step=1,
            disabled=quiz_started,
        )
        # Persist chosen total
        st.session_state["quiz_n"] = int(n)
    with cols[1]:
        difficulty = st.selectbox(
            "Difficulty",
            options=["auto", "easy", "medium", "hard"],
            index=["auto", "easy", "medium", "hard"].index(st.session_state.get("quiz_diff", "auto")),
            disabled=quiz_started,
            help="Choose difficulty or leave on auto to adapt based on history."
        )
        st.session_state["quiz_diff"] = difficulty
    with cols[2]:
        avoid_mode = st.selectbox("Avoid prompts", options=["all", "correct"], index=0, disabled=quiz_started)
    with cols[3]:
        feedback_mode = st.selectbox("Feedback mode", options=["immediate", "end"], index=0, disabled=quiz_started)

    # Question mix configuration (counts per type)
    with st.expander("⚙️ Question mix (choose counts per type)", expanded=False):
        mix_cols = st.columns(3)
        with mix_cols[0]:
            mcq_single = st.slider("MCQ (single correct)", 0, MAX_Q, value=st.session_state.get("mix_mcq_single", min(max(1, int(n)//2), MAX_Q)), key="mix_mcq_single", disabled=quiz_started)
            mcq_multi = st.slider("MCQ (multiple correct)", 0, MAX_Q, value=st.session_state.get("mix_mcq_multi", 0), key="mix_mcq_multi", disabled=quiz_started)
            tf_plain = st.slider("True/False", 0, MAX_Q, value=st.session_state.get("mix_true_false", 0), key="mix_true_false", disabled=quiz_started)
        with mix_cols[1]:
            tf_justify = st.slider("True/False + justification", 0, MAX_Q, value=st.session_state.get("mix_true_false_justify", 0), key="mix_true_false_justify", disabled=quiz_started)
            short_default = max(1, int(n) - st.session_state.get("mix_mcq_single", 0))
            short_ans = st.slider("Short answer", 0, MAX_Q, value=st.session_state.get("mix_short", short_default), key="mix_short", disabled=quiz_started)
            matching = st.slider("Matching", 0, MAX_Q, value=st.session_state.get("mix_matching", 0), key="mix_matching", disabled=quiz_started)
        with mix_cols[2]:
            ordering = st.slider("Ordering/Sequencing", 0, MAX_Q, value=st.session_state.get("mix_ordering", 0), key="mix_ordering", disabled=quiz_started)
            numeric = st.slider("Numeric calculation", 0, MAX_Q, value=st.session_state.get("mix_numeric", 0), key="mix_numeric", disabled=quiz_started)
            explain = st.slider("Explain why (conceptual)", 0, MAX_Q, value=st.session_state.get("mix_explain", 0), key="mix_explain", disabled=quiz_started)

        selected_total = sum([
            st.session_state.get("mix_mcq_single", 0),
            st.session_state.get("mix_mcq_multi", 0),
            st.session_state.get("mix_true_false", 0),
            st.session_state.get("mix_true_false_justify", 0),
            st.session_state.get("mix_short", 0),
            st.session_state.get("mix_matching", 0),
            st.session_state.get("mix_ordering", 0),
            st.session_state.get("mix_numeric", 0),
            st.session_state.get("mix_explain", 0),
        ])
        st.caption(f"Selected total: {selected_total} / {int(n)} (counts will be normalized to match total)")
        # Auto-update the total number if user adjusts the mix
        if not quiz_started and selected_total != int(n):
            new_total = max(1, min(MAX_Q, selected_total))
            st.session_state["quiz_n"] = int(new_total)
            # Force immediate UI sync
            st.rerun()

    if not quiz_started:
        # Do not require Topic text to start
        start_disabled = st.session_state.get("busy", False) or st.session_state.get("quiz_generating", False)
        cols_btn = st.columns(2)
        with cols_btn[0]:
            if st.button("Start Quiz", disabled=start_disabled, key="start_quiz"):
                # Two-phase start: set flags then rerun so Stop & Reset appears immediately
                st.session_state["quiz_generating"] = True
                # Snapshot current mix and normalize counts to N
                raw_mix = {
                    "mcq_single": int(st.session_state.get("mix_mcq_single", 0)),
                    "mcq_multi": int(st.session_state.get("mix_mcq_multi", 0)),
                    "true_false": int(st.session_state.get("mix_true_false", 0)),
                    "true_false_justify": int(st.session_state.get("mix_true_false_justify", 0)),
                    "short": int(st.session_state.get("mix_short", 0)),
                    "matching": int(st.session_state.get("mix_matching", 0)),
                    "ordering": int(st.session_state.get("mix_ordering", 0)),
                    "numeric": int(st.session_state.get("mix_numeric", 0)),
                    "explain": int(st.session_state.get("mix_explain", 0)),
                }
                total_req = sum(raw_mix.values())
                target_n = int(n)
                if total_req <= 0:
                    # Default to all MCQ single if nothing selected
                    norm_counts = {k: 0 for k in raw_mix}
                    norm_counts["mcq_single"] = target_n
                else:
                    # Largest Remainder method for fair rounding
                    proportions = {k: (v / total_req) * target_n for k, v in raw_mix.items()}
                    base_counts = {k: int(proportions[k]) for k in proportions}
                    remainder = target_n - sum(base_counts.values())
                    remainders = sorted(((proportions[k] - base_counts[k], k) for k in proportions), reverse=True)
                    norm_counts = dict(base_counts)
                    for _ in range(max(0, remainder)):
                        if not remainders:
                            break
                        _, key_to_inc = remainders.pop(0)
                        norm_counts[key_to_inc] = norm_counts.get(key_to_inc, 0) + 1
                st.session_state["quiz_pending_start"] = {
                    "n": int(target_n),
                    "avoid_mode": avoid_mode,
                    "feedback_mode": feedback_mode,
                    "topic": st.session_state.get("quiz_topic", "").strip(),
                    "difficulty": (None if st.session_state.get("quiz_diff") == "auto" else st.session_state.get("quiz_diff")),
                    "question_mix_raw": raw_mix,
                    "question_mix_counts": norm_counts,
                }
                st.rerun()
        with cols_btn[1]:
            # Stop & Reset available if there is anything to clear (quiz-specific only)
            can_reset = bool(
                st.session_state.get("quiz_generating") or st.session_state.get("quiz_started") or st.session_state.get("busy", False)
                or st.session_state.get("quiz_topic") or st.session_state.get("questions")
            )
            if st.button("⏹️ Stop & Reset", disabled=not can_reset, key="quiz_stop_reset_pre"):
                reset_quiz_state()
                for k in ["quiz_topic", "quiz_reference_file", "quiz_diff", "quiz_snapshot"]:
                    st.session_state.pop(k, None)
                st.rerun()
    else:
        # Show disabled Start and Stop & Reset only
        cols_btn = st.columns(2)
        with cols_btn[0]:
            st.button("Start Quiz", disabled=True, key="start_quiz_disabled")
        with cols_btn[1]:
            if st.button("⏹️ Stop & Reset", key="quiz_stop_reset_active"):
                reset_quiz_state()
                for k in ["quiz_topic", "quiz_reference_file", "quiz_diff", "quiz_snapshot"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # Debug output (optional, non-blocking) — disabled by default; set to True to re-enable
    if False and st.session_state.get("quiz_debug_log"):
        with st.expander("🔧 Debug (Quiz)", expanded=False):
            st.code("\n".join(st.session_state["quiz_debug_log"][-80:]))

    # Handle pending quiz start after we have rendered buttons (so Stop & Reset is visible immediately)
    pending = st.session_state.get("quiz_pending_start")
    if pending:
        st.session_state.busy = True
        with st.spinner("Preparing quiz..."):
            _quiz_debug("pending_start: detected; initializing")
            reset_quiz_state()
            # Store difficulty for engine
            st.session_state["difficulty"] = pending.get("difficulty")
            _quiz_debug(f"pending_start: settings n={pending.get('n')}, avoid={pending.get('avoid_mode')}, feedback={pending.get('feedback_mode')}")
            # Resolve absolute path of selected reference file if available
            ref_abs = None
            try:
                sel_ref = st.session_state.get("quiz_reference_file")
                if sel_ref:
                    ref_abs = str((NOTES_DIR / sel_ref).resolve())
            except Exception:
                ref_abs = None
            _quiz_debug(f"pending_start: resolved reference_file={(ref_abs or 'None')}")

            start_quiz(
                pending.get("topic", ""),
                pending.get("n", 4),
                pending.get("avoid_mode", "all"),
                pending.get("feedback_mode", "immediate"),
                question_mix_counts=pending.get("question_mix_counts"),
                reference_file=ref_abs,
            )
        _quiz_debug("pending_start: completed; clearing flags and rerunning")
        st.session_state.busy = False
        st.session_state["quiz_generating"] = False
        st.session_state.pop("quiz_pending_start", None)
        st.rerun()

    if not st.session_state.get("quiz_started"):
        return

    questions: List[Dict[str, Any]] = st.session_state.questions
    i = st.session_state.current_idx

    if st.session_state.feedback_mode == "immediate":
        # Guard: if we've reached the end, show summary
        if i >= len(questions):
            finish_quiz()
            return
        q = questions[i]

        # Render question and input
        if st.session_state.q_start is None and st.session_state.feedbacks[i] is None:
            st.session_state.q_start = time.perf_counter()
        student_answer = render_question(i, q)
        st.session_state.answers[i] = student_answer

        # Submit (only if no feedback yet)
        if st.session_state.feedbacks[i] is None:
            is_mcq = (q.get("type") == "mcq")
            submit_disabled = st.session_state.busy or (is_mcq and not student_answer)
            if st.button("Submit", disabled=submit_disabled, key=f"submit_{i}"):
                st.session_state.busy = True
                with st.spinner("Grading..."):
                    _quiz_debug(f"grading: question_index={i}")
                    took_ms = int(round((time.perf_counter() - st.session_state.q_start) * 1000))
                    st.session_state.response_ms[i] = took_ms
                    result = grade_and_log(i, q, student_answer, took_ms)
                    _quiz_debug(f"grading: correct={bool(result.get('correct'))}")
                    st.session_state.feedbacks[i] = result
                    if bool(result.get("correct")):
                        st.session_state.correct_count += 1
                st.session_state.busy = False
                st.rerun()

        # Inline feedback (if available)
        if st.session_state.feedbacks[i] is not None:
            render_feedback(i, q, st.session_state.feedbacks[i], st.session_state.response_ms[i])
            # Navigation under feedback
            cols_nav = st.columns(2)
            with cols_nav[0]:
                if i < len(questions) - 1:
                    if st.button("Next", disabled=st.session_state.busy, key=f"next_{i}"):
                        st.session_state.current_idx += 1
                        st.session_state.q_start = None
                        st.rerun()
                else:
                    st.write("")
            with cols_nav[1]:
                if i == len(questions) - 1:
                    if st.button("Finish", disabled=st.session_state.busy, key=f"finish_{i}"):
                        st.session_state.current_idx += 1
                        st.session_state.q_start = None
                        st.rerun()
    else:
        # End-mode: navigate through questions, grade at end
        # Guard: keep index in range
        if i >= len(questions):
            i = len(questions) - 1
            st.session_state.current_idx = i
        q = questions[i]
        if st.session_state.q_start is None:
            st.session_state.q_start = time.perf_counter()
        student_answer = render_question(i, q)
        st.session_state.answers[i] = student_answer
        cols_nav = st.columns(2)
        with cols_nav[0]:
            if st.button("Previous", disabled=i == 0 or st.session_state.busy, key=f"prev_{i}"):
                st.session_state.current_idx -= 1
                st.session_state.q_start = time.perf_counter()
                st.rerun()
        with cols_nav[1]:
            if i < len(questions) - 1:
                if st.button("Next", disabled=st.session_state.busy, key=f"next_end_{i}"):
                    took_ms = int(round((time.perf_counter() - st.session_state.q_start) * 1000))
                    st.session_state.response_ms[i] = took_ms
                    st.session_state.current_idx += 1
                    st.session_state.q_start = time.perf_counter()
                    st.rerun()
            else:
                if st.button("Submit All", disabled=st.session_state.busy, key=f"submit_all_{i}"):
                    st.session_state.busy = True
                    with st.spinner("Grading all answers..."):
                        _quiz_debug("grading_all: start")
                        # Record time for last question
                        took_ms = int(round((time.perf_counter() - st.session_state.q_start) * 1000))
                        st.session_state.response_ms[i] = took_ms
                        # Grade all now
                        for j, qj in enumerate(questions):
                            res = grade_and_log(j, qj, st.session_state.answers[j], st.session_state.response_ms[j])
                            st.session_state.feedbacks[j] = res
                            if bool(res.get("correct")):
                                st.session_state.correct_count += 1
                        _quiz_debug("grading_all: done")
                    st.session_state.busy = False
                    finish_quiz()
                    st.rerun()


def finish_quiz() -> None:
    total = len(st.session_state.questions)
    percent = (100 * st.session_state.correct_count / total) if total else 0.0
    st.success(f"Score: {st.session_state.correct_count}/{total} ({percent:.0f}%)")
    # Qualitative summary
    if percent < 50:
        st.info("Needs revision: focus on foundational concepts and definitions.")
    elif percent < 80:
        st.info("Fair progress: keep practicing and revisit tricky areas.")
    else:
        st.info("Good progress: you’re ready for more challenging, reasoning-based questions.")

    # Save session
    memory = JsonMemory(str(PROGRESS_PATH))
    try:
        memory.log_session(
            topic=st.session_state.topic,
            score=percent,
            details={
                "raw": f"{st.session_state.correct_count}/{total}",
                "avoid_mode": st.session_state.avoid_mode,
                "difficulty": st.session_state.difficulty,
                "feedback_mode": st.session_state.feedback_mode,
            },
        )
        st.caption("Progress saved to progress.json")
    except Exception as e:
        st.warning(f"Could not save progress: {e}")


def upload_tab():
    st.subheader("Upload Notes")
    st.write("Upload PDF, TXT, or MD files to include in your study notes.")
    
    # Clear any quiz/summary-related state when entering upload tab
    st.session_state.pop("quiz_generating", None)
    st.session_state.pop("quiz_pending_start", None)
    st.session_state.pop("summary_generating", None)
    st.session_state.pop("summary_pending", None)
    
    files = st.file_uploader("Upload files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if files:
        # Build a signature of current upload set (name + size) to avoid repeated rebuilds on rerun
        try:
            sig = tuple(sorted((f.name, len(f.getbuffer())) for f in files))
        except Exception:
            sig = tuple(sorted((f.name, 0) for f in files))
        if st.session_state.get("last_upload_sig") != sig:
            saved = save_uploaded_files(files)
            with st.spinner("Rebuilding vector store..."):
                docs = load_documents(str(NOTES_DIR))
                chunks = chunk_documents(docs)
                st.session_state.vs = build_or_load_vectorstore(chunks)
            st.session_state.last_upload_sig = sig
            st.session_state.vs_built_at = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            # Clear caches when vector store is rebuilt
            _list_notes_cached.clear()
            _load_vectorstore.clear()
            pdf_cache_keys = [k for k in st.session_state.keys() if k.startswith("pdf_preview_")]
            for key in pdf_cache_keys:
                st.session_state.pop(key, None)
            st.success(f"Uploaded {len(saved)} files. Vector store rebuilt.")
            st.balloons()
        else:
            st.caption("These uploads have already been processed.")

    # Optional: light caption style for meta rows
    st.markdown("""
        <style>.file-meta{font-size:12px; color:rgba(128,128,128,0.9);}</style>
    """, unsafe_allow_html=True)

    # Current PDFs on disk with preview and delete
    notes = list_notes()
    st.write("### Current PDFs")
    pdf_notes = [n for n in notes if str(n.get("file", "")).lower().endswith(".pdf")]
    if not pdf_notes:
        st.caption("No PDFs found yet in data/notes.")
    else:
        for i, n in enumerate(pdf_notes):
            rel = n.get("file", "")
            size_kb = n.get("size_kb", "?")
            modified = n.get("modified", "")
            path = NOTES_DIR / rel
            # Use native bordered container so all widgets are inside the same card
            with st.container(border=True):
                cols = st.columns([8, 1])
                with cols[0]:
                    st.markdown(f"**{rel}**")
                    st.markdown(f"<div class='file-meta'>Size: {size_kb} KB • Modified: {modified}</div>", unsafe_allow_html=True)
                with cols[1]:
                    if st.button("🗑️ Delete", key=f"del_pdf_{i}"):
                        try:
                            if path.exists() and path.is_file():
                                path.unlink()
                                # Clear caches
                                _list_notes_cached.clear()
                                _load_vectorstore.clear()
                                with st.spinner("Updating vector store..."):
                                    docs = load_documents(str(NOTES_DIR))
                                    chunks = chunk_documents(docs)
                                    st.session_state.vs = build_or_load_vectorstore(chunks)
                                st.success(f"Deleted {rel}.")
                                st.rerun()
                        except Exception as e:
                            st.warning(f"Could not delete {rel}: {e}")

                # PDF preview placed below the card
                with st.expander("📄 PDF preview", expanded=False):
                    try:
                        if path.exists():
                            with open(path, "rb") as f:
                                _bytes = f.read()
                            import base64 as _b64
                            b64 = _b64.b64encode(_bytes).decode()
                            iframe = f"""
                            <iframe src=\"data:application/pdf;base64,{b64}\" 
                                    width=\"100%\" height=\"1200px\" 
                                    style=\"border: 1px solid #e0e0e0; border-radius: 6px;\"></iframe>
                            """
                            st.markdown(iframe, unsafe_allow_html=True)
                        else:
                            st.caption("File not found.")
                    except Exception as _e:
                        st.caption(f"Preview unavailable: {_e}")

    # Show last vector store rebuild time if available
    if st.session_state.get("vs_built_at"):
        st.caption(f"Vector store last built at: {st.session_state.vs_built_at}")


@st.cache_data(ttl=60)  # Cache for 1 minute
def _get_progress_data():
    """Cached progress data loading"""
    memory = JsonMemory(str(PROGRESS_PATH))
    try:
        data = memory._read()
    except Exception:
        data = {"sessions": [], "attempts": [], "questions": {}}
    return data

def progress_tab():
    st.subheader("Progress")
    
    # Clear any quiz/summary-related state when entering progress tab
    st.session_state.pop("quiz_generating", None)
    st.session_state.pop("quiz_pending_start", None)
    st.session_state.pop("summary_generating", None)
    st.session_state.pop("summary_pending", None)
    
    data = _get_progress_data()
    sessions = data.get("sessions", [])

    st.write("### Sessions")
    if sessions:
        st.dataframe(sessions, use_container_width=True)
    else:
        st.caption("No sessions yet.")

    st.write("### Frequently Missed (by topic)")
    topics = sorted({s.get("topic", "") for s in sessions if s.get("topic")})
    topic = st.selectbox("Topic", options=topics or [""], index=0 if topics else 0)
    if topic:
        memory = JsonMemory(str(PROGRESS_PATH))
        missed = memory.get_frequently_missed(topic, min_attempts=1, limit=10)
        if missed:
            st.dataframe(missed, use_container_width=True)
        else:
            st.caption("No frequently missed questions yet for this topic.")


def summary_tab():
    st.subheader("📚 Professional PDF Summary Generator")
    st.write("""
    Generate beautiful, publication-quality PDF summaries of your course notes using advanced LaTeX formatting.
    Each summary includes your business branding and professional styling.
    """)
    
    # Clear any quiz-related state when entering summary tab
    st.session_state.pop("quiz_generating", None)
    st.session_state.pop("quiz_pending_start", None)
    
    # Ensure we're not in quiz mode when in summary tab
    if st.session_state.get("quiz_started", False):
        st.info("Please finish or reset your current quiz before generating summaries.")
        return
    
    # Check if vector store is loaded
    ensure_vectorstore_loaded()
    
    # Get list of available notes
    notes = list_notes()
    if not notes:
        st.warning("No notes found. Please upload some notes first in the Upload Notes tab.")
        return
    
    # Create file selection dropdown
    file_names = [note["file"] for note in notes]
    selected_file = st.selectbox("Select PDF/Notes to summarize:", file_names)
    # Preview the selected file (same preview feature as in Quiz)
    ref_path = NOTES_DIR / selected_file
    with st.expander("📄 PDF preview", expanded=False):
        try:
            if ref_path.suffix.lower() == ".pdf" and ref_path.exists():
                with open(ref_path, "rb") as f:
                    _bytes = f.read()
                import base64 as _b64
                b64 = _b64.b64encode(_bytes).decode()
                iframe = f"""
                <iframe src=\"data:application/pdf;base64,{b64}\" 
                        width=\"100%\" height=\"1200px\" 
                        style=\"border: 1px solid #e0e0e0; border-radius: 6px;\"></iframe>
                """
                st.markdown(iframe, unsafe_allow_html=True)
            elif ref_path.exists():
                try:
                    text = ref_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = "(Could not read file as text)"
                st.code(text[:5000], language="markdown")
            else:
                st.caption("Selected reference file not found.")
        except Exception as _e:
            st.caption(f"Preview unavailable: {_e}")
    
    # Focus area input
    focus = st.text_input("Focus Area/Topic (optional)", placeholder="Leave empty for a general summary of the selected PDF")
    
    # Summary type removed; default to Comprehensive
    summary_type = "Comprehensive"
    
    # Actions row: Generate and Stop & Reset side-by-side (uniform widths)
    st.markdown(
        """
        <style>
        #summary-actions .stButton > button { width: 100%; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div id='summary-actions'>", unsafe_allow_html=True)
    action_cols = st.columns([1, 1, 6])
    with action_cols[0]:
        generate_clicked = st.button("Generate Summary", disabled=not selected_file)
    with action_cols[1]:
        is_generating = st.session_state.get("summary_generating", False)
        has_summary = bool(st.session_state.get("last_summary_result")) or bool(st.session_state.get("last_pdf_bytes"))
        if st.button("⏹️ Stop & Reset", help="Cancel current generation and clear current preview", disabled=False if is_generating else not (is_generating or has_summary)):
            # Clear only the current preview/result; keep saved PDFs intact
            for k in [
                "last_summary_result",
                "last_pdf_bytes",
                "last_pdf_base64",
                "last_pdf_filename",
            ]:
                st.session_state.pop(k, None)
            st.session_state["summary_generating"] = False
            st.session_state.pop("summary_pending", None)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Generate summary flow
    if generate_clicked:
        st.session_state["summary_generating"] = True
        st.session_state["summary_pending"] = {
            "focus": focus,
            "selected_file": selected_file,
            "summary_type": summary_type,
        }
        st.rerun()

    pending_summary = st.session_state.get("summary_pending")
    if pending_summary:
        with st.spinner("Generating summary..."):
            try:
                # Retrieve context using focus if provided, otherwise use the filename as a broad query
                _focus = pending_summary.get("focus", "").strip()
                query = _focus if _focus else pending_summary.get("selected_file", "")
                context = retrieve_context(st.session_state.vs, query, k=8)
                
                if not context.strip():
                    st.warning("No relevant content found for this focus area. Try a different topic or check if the notes contain relevant information.")
                    st.session_state["summary_generating"] = False
                    st.session_state.pop("summary_pending", None)
                    # Keep message visible without immediate rerun
                    return
                
                # Import summary functions
                from src.summary_engine import generate_enhanced_summary_with_pdf
                
                # Generate summary with PDF
                # If no focus supplied, label the PDF as a general summary
                focus_for_pdf = _focus if _focus else "General Summary"
                result = generate_enhanced_summary_with_pdf(
                    context,
                    focus_for_pdf,
                    pending_summary.get("selected_file", ""),
                    "Comprehensive",
                )
                
                if result.get("status") == "success":
                    st.success("✅ Summary generated successfully!")
                    # Persist result and PDF so preview survives reruns (e.g., slider adjustments)
                    st.session_state["last_summary_result"] = result
                    try:
                        if result.get("pdf_path") and os.path.exists(result["pdf_path"]):
                            with open(result["pdf_path"], "rb") as pdf_file:
                                _bytes = pdf_file.read()
                            import base64
                            st.session_state["last_pdf_bytes"] = _bytes
                            st.session_state["last_pdf_base64"] = base64.b64encode(_bytes).decode()
                            st.session_state["last_pdf_filename"] = result.get("pdf_filename", "summary.pdf")
                        else:
                            st.session_state["last_pdf_bytes"] = None
                            st.session_state["last_pdf_base64"] = None
                            st.session_state["last_pdf_filename"] = None
                    except Exception:
                        st.session_state["last_pdf_bytes"] = None
                        st.session_state["last_pdf_base64"] = None
                        st.session_state["last_pdf_filename"] = None
                else:
                    ph = st.empty()
                    ph.error(f"Error generating summary: {result.get('error', 'Unknown error')}")
                    import time as _t
                    _t.sleep(10)
                    ph.empty()
            except Exception as e:
                ph = st.empty()
                ph.error(f"An error occurred: {str(e)}")
                st.exception(e)
                import time as _t
                _t.sleep(10)
                ph.empty()
                st.session_state["summary_generating"] = False
                st.session_state.pop("summary_pending", None)
                return
        st.session_state["summary_generating"] = False
        st.session_state.pop("summary_pending", None)
        st.rerun()
    
    # Persistent PDF preview (survives widget changes)
    persisted = st.session_state.get("last_summary_result")
    if persisted:
        st.markdown("### 📊 Summary Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Focus Area", persisted.get("focus", ""))
        with col2:
            st.metric("📄 Source Document", persisted.get("pdf_name", ""))
        with col3:
            st.metric("📏 Summary Length", f"{persisted.get('summary_length', 0)} chars")
                    
        st.markdown("### 📚 PDF Preview")
        # Fixed viewer height as requested
        viewer_height = 1200

        pdf_b64 = st.session_state.get("last_pdf_base64")
        # Fallback: if base64 missing but a path exists in last result, reload it
        if not pdf_b64 and persisted.get("pdf_path") and os.path.exists(persisted["pdf_path"]):
            try:
                with open(persisted["pdf_path"], "rb") as _f:
                    _bytes = _f.read()
                import base64
                pdf_b64 = base64.b64encode(_bytes).decode()
                st.session_state["last_pdf_base64"] = pdf_b64
                st.session_state["last_pdf_bytes"] = _bytes
                st.session_state["last_pdf_filename"] = os.path.basename(persisted["pdf_path"]) or "summary.pdf"
            except Exception:
                pdf_b64 = None

        if pdf_b64:
            pdf_display = f"""
                <iframe src=\"data:application/pdf;base64,{pdf_b64}\" 
                width=\"100%\" 
                height=\"{viewer_height}px\" 
                style=\"border: 2px solid #e0e0e0; border-radius: 8px;\">
                </iframe>"""
            st.markdown(pdf_display, unsafe_allow_html=True)
                            
            st.markdown("### 💾 Download & Save Options")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.session_state.get("last_pdf_bytes"):
                    save_clicked = st.button(
                        "💾 Save PDF to Session",
                        help="Keep this PDF accessible in the current session",
                        use_container_width=True,
                    )
                    if save_clicked:
                        # Save PDF to session state for persistent access (prevent duplicates by filename)
                        st.session_state["saved_pdfs"] = st.session_state.get("saved_pdfs", [])
                        current_name = st.session_state.get("last_pdf_filename", "summary.pdf")
                        already_saved = any(p.get("filename") == current_name for p in st.session_state["saved_pdfs"])
                        if already_saved:
                            # Show a toast that auto-dismisses (Streamlit >= 1.29)
                            try:
                                st.toast("This file has already been saved", icon="ℹ️")
                            except Exception:
                                # Fallback inline info (non-blocking, no sleep)
                                st.info("This file has already been saved")
                        else:
                            pdf_info = {
                                "filename": current_name,
                                "focus": persisted.get('focus', ''),
                                "source": persisted.get('pdf_name', ''),
                                "timestamp": persisted.get('timestamp', ''),
                                "pdf_bytes": st.session_state["last_pdf_bytes"],
                                "pdf_base64": st.session_state.get("last_pdf_base64", "")
                            }
                            st.session_state["saved_pdfs"].append(pdf_info)
                            st.success(f"✅ PDF saved! ({len(st.session_state['saved_pdfs'])} PDFs in session)")
                            st.rerun()

            with col2:
                if st.session_state.get("last_pdf_bytes"):
                    st.download_button(
                        label="📚 Download PDF",
                        data=st.session_state["last_pdf_bytes"],
                        file_name=st.session_state.get("last_pdf_filename", "summary.pdf"),
                        mime="application/pdf",
                        help="Download the professional LaTeX-generated PDF",
                        use_container_width=True,
                    )
            with col3:
                summary_text = f"# Course Notes Summary\n\n**Focus Area:** {persisted.get('focus','')}\n**Source:** {persisted.get('pdf_name','')}\n\n{persisted.get('summary','')}"
                st.download_button(
                    label="📄 Download Markdown",
                    data=summary_text,
                    file_name=f"summary_{persisted.get('focus','').replace(' ', '_')}.md",
                    mime="text/markdown",
                    help="Download as Markdown text (backup option)",
                    use_container_width=True,
                )
        else:
            st.warning("PDF preview unavailable. Try regenerating the summary.")

    # Tips (moved above Saved PDFs so it doesn't get buried below)
    with st.expander("💡 Tips for Better PDF Summaries", expanded=False):
        st.markdown("""
        - **Be specific**: Instead of 'math', try 'linear algebra' or 'calculus derivatives'
        - **Use key terms**: Include specific concepts, formulas, or theories you want to focus on
        - **Try different focus areas**: If one doesn't work, try rephrasing or being more specific
        - **Check your notes**: Make sure the selected file contains content related to your focus area
        """)

    # Display saved PDFs
    saved_pdfs = st.session_state.get("saved_pdfs", [])
    if saved_pdfs:
        st.markdown("### 📚 Saved PDFs")
        st.write(f"You have {len(saved_pdfs)} saved PDF(s) in this session:")
        
        for i, pdf_info in enumerate(saved_pdfs):
            with st.expander(f"📄 {pdf_info['filename']} - {pdf_info['focus']}"):
                col1, col3 = st.columns([3, 1])
                with col1:
                    st.write(f"**Focus:** {pdf_info['focus']}")
                    st.write(f"**Source:** {pdf_info['source']}")
                    st.write(f"**Generated:** {pdf_info['timestamp']}")
                with col3:
                    if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                        st.session_state["saved_pdfs"].pop(i)
                        st.success("PDF removed from session")
                        st.rerun()
                
                # Show PDF preview
                if pdf_info.get("pdf_base64"):
                    pdf_display = f"""
                    <iframe src="data:application/pdf;base64,{pdf_info['pdf_base64']}" 
                            width="100%" 
                            height="1200px" 
                            style="border: 1px solid #e0e0e0; border-radius: 4px;">
                    </iframe>
                    """
                    st.markdown(pdf_display, unsafe_allow_html=True)

    
    
    # Debug & Troubleshooting removed per request
def flashcards_tab():
    st.subheader("Flash Card Generator")
    
    # Clear any quiz/summary-related state when entering flashcards tab
    st.session_state.pop("quiz_generating", None)
    st.session_state.pop("quiz_pending_start", None)
    st.session_state.pop("summary_generating", None)
    st.session_state.pop("summary_pending", None)
    
    # Initialize session containers
    if "fc_deck_meta" not in st.session_state:
        st.session_state["fc_deck_meta"] = {"name": "", "subject": "", "default_tags": []}
    if "fc_options" not in st.session_state:
        st.session_state["fc_options"] = {
            "auto_mix": True,
            "counts": {"total": 12},
            "types": {"basic": 8, "cloze": 4},
            "content_mix": {"formula": 4, "definition": 4, "example": 2, "interpretation": 2},
            "difficulty": "medium",
            "front_style": "question",
            "back_style": "sentence",
            "back_bullets_max": 3,
            "cloze": {"allow": True, "max_clozes_per_card": 1, "scope": "formula"},
            "math": {"render_latex": True, "prefer_formula_cards": True, "min_formula_cards": 2},
            "tagging": {"auto": True, "mandatory": [], "max_per_card": 5},
            "diversity": {"ensure_terms": [], "per_term_min": 0, "dedupe_fronts": True},
            "quality": {"grammar_pass": False, "regen_weak": False},
            "lengths": {"front": "medium", "back": "medium"},
            "retries_max": 2,
        }

    # Ensure vector store is available
    ensure_vectorstore_loaded()
    notes = list_notes()
    if not notes:
        st.warning("No notes found. Upload a PDF first.")
        return
    files = [n["file"] for n in notes]
    try:
        default_idx = files.index(st.session_state.get("fc_ref", files[0]))
    except Exception:
        default_idx = 0
    ref = st.selectbox("PDF/Notes", files, index=default_idx, key="fc_ref")
    topic = st.text_input("Topic (optional)", key="fc_topic")
    # Deck name (moved out of overlay)
    st.session_state["fc_deck_meta"]["name"] = st.text_input(
        "Deck name (optional)",
        value=st.session_state["fc_deck_meta"].get("name", ""),
        help="Leave blank to auto‑name on save from the PDF context",
    )

    n_cards = st.number_input("Number of cards", 1, 50, st.session_state["fc_options"]["counts"]["total"], 1)
    st.session_state["fc_options"]["counts"]["total"] = int(n_cards)

    # Simplified mix: proportions only
    with st.expander("⚙️ Flashcard Options", expanded=False):
        st.session_state["fc_options"]["auto_mix"] = st.checkbox(
            "Auto determine the balance of card types from PDF/topic (recommended)",
            value=st.session_state["fc_options"].get("auto_mix", True),
            help="If enabled, the generator infers the right balance (e.g., more formula cards for formula-heavy PDFs).",
        )
        auto_mix = st.session_state["fc_options"]["auto_mix"]
        cm = st.session_state["fc_options"]["content_mix"]
        if auto_mix:
            st.info("Mix will be inferred automatically. Disable to set manual proportions.")
        cols_prop = st.columns(4)
        with cols_prop[0]:
            cm["formula"] = st.slider("Formula %", 0, 100, int(cm.get("formula", 25)), disabled=auto_mix)
        with cols_prop[1]:
            cm["definition"] = st.slider("Definition %", 0, 100, int(cm.get("definition", 25)), disabled=auto_mix)
        with cols_prop[2]:
            cm["example"] = st.slider("Example %", 0, 100, int(cm.get("example", 25)), disabled=auto_mix)
        with cols_prop[3]:
            cm["interpretation"] = st.slider("Interpretation %", 0, 100, int(cm.get("interpretation", 25)), disabled=auto_mix)
        total_pct = int(cm["formula"]) + int(cm["definition"]) + int(cm["example"]) + int(cm["interpretation"])
        if not auto_mix:
            if total_pct != 100:
                st.caption(f"Selected total: {total_pct}% (will be normalized to 100% on generation)")
            else:
                st.caption("Total: 100%")

        # Allowed card types as checkboxes
        st.markdown("**Card types**")
        type_options = ["term_def", "qa", "mcq", "tf", "cloze"]
        current_types = set(st.session_state["fc_options"].get("card_types", ["term_def","qa"]))
        cols_types = st.columns(5)
        chosen: list[str] = []
        for idx, t in enumerate(type_options):
            label = {
                "term_def": "Term/Definition",
                "qa": "Question → Answer",
                "mcq": "Multiple Choice (MCQ)",
                "tf": "True / False",
                "cloze": "Cloze (fill‑in‑the‑blank)",
            }[t]
            help_txt = {
                "term_def": "Front shows a term; back gives a concise definition.",
                "qa": "Standard question on front with a short, precise answer on back.",
                "mcq": "Question with options (A–D). Back includes the correct option.",
                "tf": "Statement to judge as True/False. Back may include a one‑line justification.",
                "cloze": "Fill‑in‑the‑blank using Anki cloze syntax {{c1::...}} (great for Anki).",
            }[t]
            with cols_types[idx]:
                if st.checkbox(label, value=(t in current_types), key=f"fc_ct_{t}", help=help_txt):
                    chosen.append(t)
        if not chosen:
            # Ensure at least one type is allowed
            chosen = ["term_def","qa"]
        st.session_state["fc_options"]["card_types"] = chosen

        # Length preferences
        st.markdown("**Card formatting**")
        len_cols = st.columns(2)
        with len_cols[0]:
            st.session_state["fc_options"]["lengths"]["front"] = st.selectbox(
                "Front length",
                ["Short", "Medium", "Long"],
                index=["Short","Medium","Long"].index(st.session_state["fc_options"]["lengths"].get("front", "medium").capitalize()),
            ).lower()
        with len_cols[1]:
            st.session_state["fc_options"]["lengths"]["back"] = st.selectbox(
                "Back length",
                ["Short", "Medium", "Long"],
                index=["Short","Medium","Long"].index(st.session_state["fc_options"]["lengths"].get("back", "medium").capitalize()),
            ).lower()

        # Back style (moved here from Advanced)
        st.session_state["fc_options"]["back_style"] = st.selectbox(
            "Back style",
            ["sentence", "bullets"],
            index=["sentence", "bullets"].index(st.session_state["fc_options"]["back_style"]),
            help="Choose how answers are formatted by default.",
        )
        if st.session_state["fc_options"]["back_style"] == "bullets":
            st.session_state["fc_options"]["back_bullets_max"] = st.slider(
                "Max bullets",
                1,
                5,
                st.session_state["fc_options"]["back_bullets_max"],
            )

        # Tagging controls (moved here)
        st.markdown("**Tagging**")
        tg = st.session_state["fc_options"]["tagging"]
        tg["auto"] = st.checkbox(
            "Auto‑tag from keywords",
            value=tg.get("auto", True),
            help="Automatically add tags based on frequent/key terms in the PDF.",
        )
        use_mand = st.checkbox(
            "Use mandatory tags",
            value=bool(tg.get("mandatory")),
            help="Always include these tags on every card.",
        )
        mand_str = ", ".join(tg.get("mandatory", [])) if use_mand else ""
        mand_edit = st.text_input(
            "Mandatory tags (comma‑separated)",
            value=mand_str,
            disabled=not use_mand,
        )
        tg["mandatory"] = ([t.strip() for t in mand_edit.split(",") if t.strip()] if use_mand else [])

    # Advanced options removed per request

    # Helper: reset current (unsaved) deck only
    def _fc_reset_current_deck():
        st.session_state.pop("fc_cards", None)
        st.session_state.pop("fc_pending", None)
        st.session_state.pop("fc_pseudo_topic", None)
        # Reset editable deck meta but keep saved decks and options
        st.session_state["fc_deck_meta"] = {"name": "", "subject": "", "default_tags": []}
        # Clear any per-card edit toggles
        for _k in list(st.session_state.keys()):
            if isinstance(_k, str) and _k.startswith("fc_edit_"):
                st.session_state.pop(_k, None)

    # Actions row: Generate and Stop & Reset (does not affect saved decks)
    fc_actions = st.columns(2)
    with fc_actions[0]:
        if st.button("Generate Cards"):
            # Two-phase flow so the spinner appears immediately on the next run
            st.session_state["fc_pending"] = {
                "ref": ref,
                "topic": topic.strip(),
                "n": int(n_cards),
                "options": st.session_state.get("fc_options", {}),
            }
            st.rerun()
    with fc_actions[1]:
        can_reset_fc = bool(st.session_state.get("fc_pending") or st.session_state.get("fc_cards"))
        if st.button("⏹️ Stop & Reset", disabled=not can_reset_fc, key="fc_stop_reset"):
            _fc_reset_current_deck()
            st.rerun()

    # Handle pending generation so spinner appears right away
    pending_fc = st.session_state.get("fc_pending")
    if pending_fc:
        with st.spinner("Generating cards..."):
            try:
                ref_name = pending_fc.get("ref") or st.session_state.get("fc_ref")
                ref_abs = str((NOTES_DIR / str(ref_name)).resolve())
                ctx = retrieve_context(st.session_state.vs, pending_fc.get("topic", ""), k=10, source_path=ref_abs)
            except Exception as e:
                st.warning(f"Could not retrieve content: {e}")
                st.session_state.pop("fc_pending", None)
                return
            if not ctx.strip():
                st.warning("No relevant content found in the selected PDF.")
                st.session_state.pop("fc_pending", None)
                return
            pseudo = pending_fc.get("topic", "").strip()
            if not pseudo:
                try:
                    from src.auto_topic import one_liner_from_context
                    pseudo = one_liner_from_context(ctx)
                except Exception:
                    pseudo = "Key concepts"
            opts = pending_fc.get("options", {})
            allow_cloze_flag = ("cloze" in (opts.get("card_types", ["term_def","qa","cloze"])) )
            out = generate_flashcards(
                ctx,
                pseudo,
                n=int(pending_fc.get("n", 12)),
                allow_cloze=allow_cloze_flag,
                options=opts,
            )
            cards = out.get("cards", [])
            if not cards:
                st.warning("No cards generated. Try adjusting topic or content.")
                st.session_state.pop("fc_pending", None)
                return
            st.session_state["fc_cards"] = cards
            st.session_state["fc_pseudo_topic"] = pseudo
        st.session_state.pop("fc_pending", None)
        st.rerun()

    cards = st.session_state.get("fc_cards", [])
    if cards:
        st.markdown("### ✏️ Your Flash Cards")
        # Helper to save deck with optional auto-generated name when empty
        def _fc_save_current_deck():
            deck_name = (st.session_state.get("fc_deck_meta", {}).get("name", "").strip())
            if not deck_name:
                try:
                    ref_name = st.session_state.get("fc_ref")
                    ref_abs = str((NOTES_DIR / str(ref_name)).resolve())
                    topic_hint = (st.session_state.get("fc_topic", "").strip() or st.session_state.get("fc_pseudo_topic", "").strip())
                    ctx = retrieve_context(st.session_state.vs, topic_hint, k=5, source_path=ref_abs)
                    try:
                        from src.auto_topic import one_liner_from_context
                        auto_name = one_liner_from_context(ctx).strip()
                    except Exception:
                        auto_name = "Study Deck"
                    deck_name = auto_name or "Study Deck"
                except Exception:
                    deck_name = "Study Deck"
                st.session_state["fc_deck_meta"]["name"] = deck_name
            deck_info = {
                "name": deck_name,
                "subject": st.session_state.get("fc_deck_meta", {}).get("subject", ""),
                "default_tags": st.session_state.get("fc_deck_meta", {}).get("default_tags", []),
                "cards": st.session_state.get("fc_cards", []),
                "options": st.session_state.get("fc_options", {}),
            }
            st.session_state.setdefault("saved_decks", []).append(deck_info)
            try:
                st.toast(f"Deck saved as '{deck_info['name']}'", icon="✅")
            except Exception:
                st.success(f"Deck saved as '{deck_info['name']}'")

        # Top Save Deck button (simple save)
        top_actions = st.columns([1, 6])
        with top_actions[0]:
            if st.button("💾 Save Deck", key="fc_save_deck_top"):
                _fc_save_current_deck()
        # Render cards with inline edit mode
        allowed_types = st.session_state.get("fc_options", {}).get("card_types", ["term_def","qa"]) or ["term_def","qa"]
        type_labels = {
            "term_def": "Term/Definition",
            "qa": "Question → Answer",
            "mcq": "Multiple Choice (MCQ)",
            "tf": "True / False",
            "cloze": "Cloze (fill‑in‑the‑blank)",
        }
        for i, c in enumerate(cards):
            with st.container(border=True):
                # Track edit mode per card
                edit_key = f"fc_edit_{i}"
                if edit_key not in st.session_state:
                    st.session_state[edit_key] = False

                hdr = st.columns([6, 1, 1])
                with hdr[0]:
                    st.markdown(f"**Card {i+1}** — {type_labels.get(c.get('type','qa'),'')} ")
                with hdr[1]:
                    if st.button("✏️ Edit" if not st.session_state[edit_key] else "✅ Done", key=f"fc_toggle_{i}"):
                        st.session_state[edit_key] = not st.session_state[edit_key]
                        st.rerun()
                with hdr[2]:
                    if st.button("🗑️ Delete", key=f"fc_del_{i}"):
                        st.session_state["fc_cards"].pop(i)
                        st.rerun()

                if not st.session_state[edit_key]:
                    # Read-only view
                    st.markdown(f"**Front:** {c.get('front','')}")
                    st.markdown(f"**Back:** {c.get('back','')}")
                    tags_disp = ", ".join(c.get("tags", [])) or "—"
                    st.caption(f"Tags: {tags_disp}")
                else:
                    # Editable view (auto-saves on Done)
                    options = [t for t in allowed_types if t in type_labels] or ["qa"]
                    labels = [type_labels[t] for t in options]
                    sel_idx = options.index(c.get("type","qa")) if c.get("type","qa") in options else 0
                    sel_label = st.selectbox("Type", labels, index=sel_idx, key=f"fc_type_{i}")
                    c["type"] = options[labels.index(sel_label)]
                    c["front"] = st.text_area("Front", value=c.get("front",""), key=f"fc_front_{i}", height=80)
                    c["back"] = st.text_area("Back", value=c.get("back",""), key=f"fc_back_{i}", height=120)
                    tags_str = ", ".join(c.get("tags", []))
                    tags_edit = st.text_input("Tags (comma‑separated)", value=tags_str, key=f"fc_tags_{i}")
                    c["tags"] = [t.strip() for t in tags_edit.split(",") if t.strip()]

        # Bottom Save Deck button
        bottom_actions = st.columns([1, 6])
        with bottom_actions[0]:
            if st.button("💾 Save Deck", key="fc_save_deck_bottom"):
                _fc_save_current_deck()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("⬇️ Download Anki TSV", to_tsv(cards), file_name="deck.tsv", mime="text/tab-separated-values")
            st.caption("Import in Anki: Text (tabs) → map Front, Back, Tags. Use Cloze model if fronts have {{c1::...}}.")
        with col2:
            st.download_button("⬇️ Download Quizlet CSV", to_csv_quizlet(cards), file_name="deck.csv", mime="text/csv")
        with col3:
            st.download_button("⬇️ Download Markdown", to_markdown(cards), file_name="deck.md", mime="text/markdown")

        # Modal flow removed

    # Saved decks list (session-persistent) — always visible
    saved = st.session_state.get("saved_decks", [])
    if saved:
        st.markdown("### 📚 Saved Decks (this session)")
        for i, dk in enumerate(saved):
            with st.container(border=True):
                # Unified single row: info | format | download | load | delete
                cols_sd = st.columns([6, 3, 2, 1, 1], gap="small")
                with cols_sd[0]:
                    deck_name = dk.get("name", "Untitled deck") or "Untitled deck"
                    subject = dk.get("subject", "")
                    num_cards = len(dk.get("cards", []))
                    st.markdown(f"**{deck_name}**")
                    meta_line = f"{num_cards} cards"
                    if subject:
                        meta_line += f" • {subject}"
                    st.caption(meta_line)

                    # Prepare export data (merge default tags with per-card tags)
                    cards_src = dk.get("cards", [])
                    default_tags = dk.get("default_tags", []) or []
                    export_cards = []
                    for c in cards_src:
                        tags = (c.get("tags", []) or []) + list(default_tags)
                        seen = set()
                        merged_tags = []
                        for t in tags:
                            tt = t.strip()
                            if tt and tt not in seen:
                                seen.add(tt)
                                merged_tags.append(tt)
                        export_cards.append({**c, "tags": merged_tags})
                with cols_sd[1]:
                    fmt = st.selectbox(
                        "Export format",
                        ["Anki TSV", "Quizlet CSV", "Markdown"],
                        index=0,
                        key=f"fc_export_fmt_{i}",
                        label_visibility="collapsed",
                    )
                with cols_sd[2]:
                    if fmt == "Anki TSV":
                        data = to_tsv(export_cards)
                        ext = "tsv"
                        mime = "text/tab-separated-values"
                    elif fmt == "Quizlet CSV":
                        data = to_csv_quizlet(export_cards)
                        ext = "csv"
                        mime = "text/csv"
                    else:
                        data = to_markdown(export_cards)
                        ext = "md"
                        mime = "text/markdown"

                    raw_name = dk.get("name", "deck").strip() or "deck"
                    safe_name = "".join(ch if (ch.isalnum() or ch in (" ", "-", "_")) else "_" for ch in raw_name).strip().replace(" ", "_")
                    file_name = f"{safe_name or 'deck'}.{ext}"

                    st.download_button(
                        label="⬇️ Download",
                        data=data,
                        file_name=file_name,
                        mime=mime,
                        key=f"fc_export_dl_{i}",
                        help="Export this saved deck in the chosen format.",
                        use_container_width=True,
                    )
                with cols_sd[3]:
                    if st.button("Load", key=f"fc_load_{i}", use_container_width=True):
                        st.session_state["fc_deck_meta"] = {
                            "name": dk.get("name", ""),
                            "subject": dk.get("subject", ""),
                            "default_tags": dk.get("default_tags", []),
                        }
                        st.session_state["fc_cards"] = dk.get("cards", [])
                        if dk.get("options"):
                            st.session_state["fc_options"] = dk.get("options")
                        st.rerun()
                with cols_sd[4]:
                    if st.button("🗑️ Delete", key=f"fc_del_deck_{i}", use_container_width=True):
                        st.session_state["saved_decks"].pop(i)
                        st.rerun()


def cleanup_session_state():
    """Clean up old session state to prevent memory leaks"""
    # Keep only essential keys and recent data
    essential_keys = {
        "vs", "quiz_started", "questions", "current_idx", "answers", "response_ms", 
        "feedbacks", "correct_count", "q_start", "step", "last_feedback", "last_took_ms", 
        "last_q", "busy", "quiz_topic", "avoid_mode", "feedback_mode", "difficulty", 
        "quiz_n", "nav", "notes_list_cache"
    }
    
    # Clean up old PDF preview caches (keep only last N)
    pdf_cache_keys = [k for k in st.session_state.keys() if k.startswith("pdf_preview_")]
    if len(pdf_cache_keys) > PDF_PREVIEW_CACHE_LIMIT:
        # Remove oldest caches
        for key in pdf_cache_keys[:-PDF_PREVIEW_CACHE_LIMIT]:
            st.session_state.pop(key, None)

def main():
    # Set page config with logo as page icon if available
    try:
        page_icon_path = "16.png" if os.path.exists("16.png") else None
        st.set_page_config(page_title="Edvance Study Coach", page_icon=page_icon_path, layout="wide")
    except Exception:
        # Fallback without page icon (Streamlit disallows multiple set_page_config calls across reruns)
        st.set_page_config(page_title="Edvance Study Coach", layout="wide")
    
    # Start background pre-loading of embeddings
    threading.Thread(target=_preload_embeddings, daemon=True).start()
    
    # Clean up session state periodically
    cleanup_session_state()

    # Hide Streamlit toolbar (Deploy), hamburger menu, and footer
    st.markdown(
        """
        <style>
        [data-testid="stToolbar"] { display: none !important; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        /* Reduce top padding before the first container to pull the header up */
        .block-container { padding-top: 8px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Brand header (compact flex layout)
    _render_brand_header()

    # Persistent navigation to prevent tab reset on rerun
    default_nav = st.session_state.get("nav", "Quiz")
    selected_nav = st.radio(
        "Navigation",
        ["Upload Notes", "Progress", "Quiz", "Summary", "Flash Cards"],
        index=["Upload Notes", "Progress", "Quiz", "Summary", "Flash Cards"].index(default_nav),
        horizontal=True,
        key="nav",
    )

    # Initialize state only when needed (lazy initialization)
    if selected_nav in ["Quiz", "Summary", "Flash Cards"]:
        ensure_quiz_state_initialized()
        ensure_vectorstore_loaded()
    elif selected_nav == "Progress":
        # Only initialize minimal state for progress tab
        if "nav" not in st.session_state:
            st.session_state["nav"] = selected_nav

    if selected_nav == "Upload Notes":
        upload_tab()
    elif selected_nav == "Progress":
        progress_tab()
    elif selected_nav == "Quiz":
        quiz_tab()
    elif selected_nav == "Summary":
        summary_tab()
    else:
        flashcards_tab()


if __name__ == "__main__":
    main() 