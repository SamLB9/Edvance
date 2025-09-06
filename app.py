import streamlit as st
import os
from pathlib import Path
import time
import base64
from typing import List, Dict, Any

from src.ingest import load_documents, chunk_documents
from src.retriever import build_or_load_vectorstore, retrieve_context
from src.quiz_engine import generate_quiz
from src.evaluation import grade_answer
from src.memory import JsonMemory

NOTES_DIR = Path("data/notes")
PROGRESS_PATH = Path("progress.json")


def ensure_notes_dir() -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_files(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Path]:
    saved = []
    ensure_notes_dir()
    for uf in files:
        dest = NOTES_DIR / uf.name
        dest.write_bytes(uf.getbuffer())
        saved.append(dest)
    return saved


def list_notes() -> List[Dict[str, Any]]:
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
        <div style="display:flex; align-items:center; gap:16px; padding:8px 0;">
          {logo_img_tag}
          <h1 style="margin:0;">Edvance Study Coach</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

def ensure_vectorstore_loaded() -> None:
    if "vs" not in st.session_state:
        # Load existing vectorstore if present
        st.session_state.vs = build_or_load_vectorstore([])


def reset_quiz_state() -> None:
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def start_quiz(topic: str, n: int, avoid_mode: str, feedback_mode: str) -> None:
    ensure_vectorstore_loaded()
    memory = JsonMemory(str(PROGRESS_PATH))

    # Retrieve context
    ctx = retrieve_context(st.session_state.vs, topic, k=6)
    if not ctx.strip():
        st.warning("No relevant context retrieved. Try uploading notes and rebuilding.")
        return

    # Exclusions & difficulty
    excluded = memory.get_excluded_prompts(mode=avoid_mode, topic=topic)
    # Prefer user-chosen difficulty if provided, else adapt from history
    chosen = st.session_state.get("difficulty")
    difficulty = chosen if chosen in {"easy", "medium", "hard"} else memory.get_adaptive_difficulty(topic)

    # Generate quiz
    quiz = generate_quiz(ctx, topic, n_questions=n, excluded_prompts=excluded, difficulty=difficulty)
    questions = quiz.get("questions", []) if isinstance(quiz, dict) else []
    if not questions:
        st.warning("Quiz generation returned no questions. Try refining the topic or rebuilding the vector store.")
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
    # Disable inputs if busy or feedback already exists for this question
    disabled = st.session_state.busy or (st.session_state.feedbacks[i] is not None)
    if q.get("type") == "mcq":
        options = q.get("options", [])
        placeholder = "— Select an option —"
        choices = [placeholder] + options
        if answer in options:
            idx = choices.index(answer)
        else:
            idx = 0
        selected = st.radio("Choose an option:", choices, index=idx, key=f"mcq_{i}", disabled=disabled)
        return "" if selected == placeholder else selected
    else:
        return st.text_area("Your answer:", value=answer, key=f"short_{i}", height=100, disabled=disabled)


def render_feedback(i: int, q: Dict[str, Any], result: Dict[str, Any], took_ms: int) -> None:
    is_correct = bool(result.get("correct"))
    status = "✅ Correct" if is_correct else "❌ Incorrect"
    feedback = result.get("feedback", "")
    st.markdown(f"**{status}**  •  {took_ms} ms")
    if feedback:
        st.write(feedback)
    # Show correct option/answer for MCQ
    if q.get("type") == "mcq":
        st.caption(f"Correct answer: {q.get('answer','')}")


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
                    with open(ref_path, "rb") as f:
                        _bytes = f.read()
                    import base64 as _b64
                    b64 = _b64.b64encode(_bytes).decode()
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
    cols = st.columns(4)
    with cols[0]:
        n = st.number_input("Number of questions", min_value=1, max_value=10, value=4, step=1, disabled=quiz_started)
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

    if not quiz_started:
        # Do not require Topic text to start
        start_disabled = st.session_state.get("busy", False)
        cols_btn = st.columns(2)
        with cols_btn[0]:
            if st.button("Start Quiz", disabled=start_disabled, key="start_quiz"):
                # Two-phase start: set flags then rerun so Stop & Reset appears immediately
                st.session_state["quiz_generating"] = True
                st.session_state["quiz_pending_start"] = {
                    "n": int(n),
                    "avoid_mode": avoid_mode,
                    "feedback_mode": feedback_mode,
                    "topic": st.session_state.get("quiz_topic", "").strip(),
                    "difficulty": (None if st.session_state.get("quiz_diff") == "auto" else st.session_state.get("quiz_diff")),
                }
                st.rerun()
        with cols_btn[1]:
            # Stop & Reset available if there is anything to clear
            can_reset = bool(
                st.session_state.get("quiz_generating") or st.session_state.get("quiz_started") or st.session_state.get("busy", False)
                or st.session_state.get("quiz_topic") or st.session_state.get("questions")
            )
            if st.button("⏹️ Stop & Reset", disabled=not can_reset, key="quiz_stop_reset_pre"):
                reset_quiz_state()
                for k in ["quiz_topic", "quiz_reference_file", "quiz_diff", "quiz_snapshot"]:
                    st.session_state.pop(k, None)
                st.session_state["quiz_generating"] = False
                st.session_state.pop("quiz_pending_start", None)
                st.rerun()
    else:
        # Show disabled Start, Stop, and Stop & Reset
        cols_btn = st.columns(3)
        with cols_btn[0]:
            st.button("Start Quiz", disabled=True, key="start_quiz_disabled")
        with cols_btn[1]:
            if st.button("Stop", disabled=st.session_state.get("busy", False), key="stop_quiz"):
                reset_quiz_state()
                st.info("Quiz stopped. You can modify the settings and start again.")
                _snapshot_quiz_state()
                st.rerun()
        with cols_btn[2]:
            if st.button("⏹️ Stop & Reset", key="quiz_stop_reset_active"):
                reset_quiz_state()
                for k in ["quiz_topic", "quiz_reference_file", "quiz_diff", "quiz_snapshot"]:
                    st.session_state.pop(k, None)
                st.session_state["quiz_generating"] = False
                st.session_state.pop("quiz_pending_start", None)
                st.rerun()

    # Handle pending quiz start after we have rendered buttons (so Stop & Reset is visible immediately)
    pending = st.session_state.get("quiz_pending_start")
    if pending:
        st.session_state.busy = True
        with st.spinner("Preparing quiz..."):
            reset_quiz_state()
            # Store difficulty for engine
            st.session_state["difficulty"] = pending.get("difficulty")
            start_quiz(pending.get("topic", ""), pending.get("n", 4), pending.get("avoid_mode", "all"), pending.get("feedback_mode", "immediate"))
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
                    took_ms = int(round((time.perf_counter() - st.session_state.q_start) * 1000))
                    st.session_state.response_ms[i] = took_ms
                    result = grade_and_log(i, q, student_answer, took_ms)
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
                        # Record time for last question
                        took_ms = int(round((time.perf_counter() - st.session_state.q_start) * 1000))
                        st.session_state.response_ms[i] = took_ms
                        # Grade all now
                        for j, qj in enumerate(questions):
                            res = grade_and_log(j, qj, st.session_state.answers[j], st.session_state.response_ms[j])
                            st.session_state.feedbacks[j] = res
                            if bool(res.get("correct")):
                                st.session_state.correct_count += 1
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


def progress_tab():
    st.subheader("Progress")
    memory = JsonMemory(str(PROGRESS_PATH))
    try:
        data = memory._read()
    except Exception:
        data = {"sessions": [], "attempts": [], "questions": {}}

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

    # Actions row will be shown next to Generate button (added below)
    
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
    focus = st.text_input("Focus Area/Topic", placeholder="e.g., Bayes theorem, neural networks, calculus derivatives")
    
    # Summary type selection
    summary_type = st.selectbox("Summary Type", ["Comprehensive", "Structured"], 
                               help="Comprehensive: General summary, Structured: Organized with specific sections")
    
    # Actions row: Generate and Stop & Reset side-by-side
    action_cols = st.columns([1, 1, 6])
    with action_cols[0]:
        generate_clicked = st.button("Generate Summary", disabled=not focus.strip() or not selected_file)
    with action_cols[1]:
        is_generating = st.session_state.get("summary_generating", False)
        has_summary = bool(st.session_state.get("last_summary_result")) or bool(st.session_state.get("last_pdf_bytes"))
        if st.button("⏹️ Stop & Reset", help="Cancel current generation and clear summary state", disabled=False if is_generating else not (is_generating or has_summary)):
            for k in [
                "last_summary_result",
                "last_pdf_bytes",
                "last_pdf_base64",
                "last_pdf_filename",
            ]:
                st.session_state.pop(k, None)
            st.session_state["saved_pdfs"] = []
            st.session_state["summary_generating"] = False
            st.session_state.pop("summary_pending", None)
            st.rerun()

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
                # Retrieve context for the selected file and focus
                context = retrieve_context(st.session_state.vs, pending_summary.get("focus", ""), k=8)
                
                if not context.strip():
                    st.warning("No relevant content found for this focus area. Try a different topic or check if the notes contain relevant information.")
                    st.session_state["summary_generating"] = False
                    st.session_state.pop("summary_pending", None)
                    st.rerun()
                
                # Import summary functions
                from src.summary_engine import generate_enhanced_summary_with_pdf
                
                # Generate summary with PDF
                result = generate_enhanced_summary_with_pdf(
                    context,
                    pending_summary.get("focus", ""),
                    pending_summary.get("selected_file", ""),
                    pending_summary.get("summary_type", "Comprehensive"),
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
                    st.error(f"Error generating summary: {result.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
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
            </iframe>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)

            st.markdown("### 💾 Download & Save Options")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.session_state.get("last_pdf_bytes"):
                    st.download_button(
                        label="📚 Download PDF",
                        data=st.session_state["last_pdf_bytes"],
                        file_name=st.session_state.get("last_pdf_filename", "summary.pdf"),
                        mime="application/pdf",
                        help="Download the professional LaTeX-generated PDF"
                    )
            with col2:
                summary_text = f"# Course Notes Summary\n\n**Focus Area:** {persisted.get('focus','')}\n**Source:** {persisted.get('pdf_name','')}\n\n{persisted.get('summary','')}"
                st.download_button(
                    label="📄 Download Markdown",
                    data=summary_text,
                    file_name=f"summary_{persisted.get('focus','').replace(' ', '_')}.md",
                    mime="text/markdown",
                    help="Download as Markdown text (backup option)"
                )
            with col3:
                if st.session_state.get("last_pdf_bytes"):
                    if st.button("💾 Save PDF to Session", help="Keep this PDF accessible in the current session"):
                        # Save PDF to session state for persistent access
                        st.session_state["saved_pdfs"] = st.session_state.get("saved_pdfs", [])
                        pdf_info = {
                            "filename": st.session_state.get("last_pdf_filename", "summary.pdf"),
                            "focus": persisted.get('focus', ''),
                            "source": persisted.get('pdf_name', ''),
                            "timestamp": persisted.get('timestamp', ''),
                            "pdf_bytes": st.session_state["last_pdf_bytes"],
                            "pdf_base64": st.session_state.get("last_pdf_base64", "")
                        }
                        st.session_state["saved_pdfs"].append(pdf_info)
                        st.success(f"✅ PDF saved! ({len(st.session_state['saved_pdfs'])} PDFs in session)")
                        st.rerun()
        else:
            st.warning("PDF preview unavailable. Try regenerating the summary.")

    # Display saved PDFs
    saved_pdfs = st.session_state.get("saved_pdfs", [])
    if saved_pdfs:
        st.markdown("### 📚 Saved PDFs")
        st.write(f"You have {len(saved_pdfs)} saved PDF(s) in this session:")
        
        for i, pdf_info in enumerate(saved_pdfs):
            with st.expander(f"📄 {pdf_info['filename']} - {pdf_info['focus']}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**Focus:** {pdf_info['focus']}")
                    st.write(f"**Source:** {pdf_info['source']}")
                    st.write(f"**Generated:** {pdf_info['timestamp']}")
                with col2:
                    if st.button(f"👁️ View", key=f"view_{i}"):
                        st.session_state["last_pdf_bytes"] = pdf_info["pdf_bytes"]
                        st.session_state["last_pdf_base64"] = pdf_info["pdf_base64"]
                        st.session_state["last_pdf_filename"] = pdf_info["filename"]
                        st.rerun()
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
                            height="600px" 
                            style="border: 1px solid #e0e0e0; border-radius: 4px;">
                    </iframe>
                    """
                    st.markdown(pdf_display, unsafe_allow_html=True)

    # Show some tips
    with st.expander("💡 Tips for Better PDF Summaries"):
        st.markdown("""
        - **Be specific**: Instead of 'math', try 'linear algebra' or 'calculus derivatives'
        - **Use key terms**: Include specific concepts, formulas, or theories you want to focus on
        - **Check your notes**: Make sure the selected file contains content related to your focus area
        - **Try different focus areas**: If one doesn't work, try rephrasing or being more specific
        - **PDF Quality**: Each summary generates a professional LaTeX PDF with your business branding
        - **Preview First**: Check the PDF preview before downloading to ensure it meets your needs
        """)
    
    # Debug section for troubleshooting
    with st.expander("🔧 Debug & Troubleshooting"):
        if st.button("Test LaTeX Compilation"):
            from src.summary_engine import test_latex_compilation
            result = test_latex_compilation()
            if result["success"]:
                st.success("✅ LaTeX compilation test passed!")
                st.json(result)
            else:
                st.error("❌ LaTeX compilation test failed!")
                st.json(result)
        
        st.markdown("""
        **Common PDF Generation Issues:**
        - **LaTeX not installed**: Install TeX Live or MiKTeX
        - **Permission errors**: Check write access to output directory
        - **Compilation errors**: Usually content-related, check LaTeX syntax
        - **File cleanup issues**: Temporary files not being removed properly
        """)


def main():
    # Set page config with logo as page icon if available
    try:
        page_icon_path = "16.png" if os.path.exists("16.png") else None
        st.set_page_config(page_title="Edvance Study Coach", page_icon=page_icon_path, layout="wide")
    except Exception:
        # Fallback without page icon (Streamlit disallows multiple set_page_config calls across reruns)
        st.set_page_config(page_title="Edvance Study Coach", layout="wide")

    # Hide Streamlit toolbar (Deploy), hamburger menu, and footer
    st.markdown(
        """
        <style>
        [data-testid="stToolbar"] { display: none !important; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize state to persist across reruns/tab switches
    ensure_quiz_state_initialized()

    # Brand header (compact flex layout)
    _render_brand_header()

    # Persistent navigation to prevent tab reset on rerun
    default_nav = st.session_state.get("nav", "Quiz")
    selected_nav = st.radio("", ["Upload Notes", "Quiz", "Progress", "Summary"], index=["Upload Notes", "Quiz", "Progress", "Summary"].index(default_nav), horizontal=True, key="nav")

    if selected_nav == "Upload Notes":
        upload_tab()
    elif selected_nav == "Quiz":
        quiz_tab()
    elif selected_nav == "Progress":
        progress_tab()
    else:
        summary_tab()


if __name__ == "__main__":
    main() 