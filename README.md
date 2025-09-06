## 🧑‍🏫 Edvance Study Coach — Streamlit Q&A & PDF Summaries

An LLM-powered RAG study coach that:
- ingests your notes (PDF/TXT/MD),
- generates adaptive quizzes with feedback,
- tracks progress,
- and creates professional LaTeX → PDF summaries with a branded title page.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com)

---

## 🔑 Key features

### 1) Upload Notes
- Upload PDF/TXT/MD files (stored in `data/notes/`).
- A Chroma vector store is (re)built so content is searchable.
- Clean “file card” list:
  - PDF preview expander (1200px)
  - Delete per file (auto rebuilds vector store)

### 2) Quiz
- Topic (optional), Number of questions, Avoid prompts, Feedback mode (immediate/end).
- Difficulty combobox: `auto`, `easy`, `medium`, `hard`.
  - If `auto`, the engine uses adaptive difficulty (based on past performance).
- Non-blocking UI:
  - “Start Quiz” shows a spinner and immediately enables “Stop & Reset”.
  - State persists across tab switches.
- Per-question feedback (immediate) or batch feedback (end).
- Progress and attempts are logged.

### 3) Summary (Professional PDF)
- Select a note and optionally set a Focus (leave empty to summarize the whole PDF).
- PDF preview expander for the source note.
- Generate Summary:
  - The app sends cleaned content to the LLM for LaTeX body generation.
  - A standardized LaTeX preamble/title page is used (consistent branding, colors, logo).
  - Compiles with `pdflatex` (2 passes).
- Filename policy:
  - `summary_{pdf_name}_{focus}.pdf`
  - If exists, de-dupes as `..._1.pdf`, `..._2.pdf`, etc.
- PDF metadata (pdftitle) is set so viewers show a friendly title.
- Download & Save options:
  - Save PDF to Session (prevent duplicate saves; shows auto-dismissing toast)
  - Download PDF
  - Download Markdown (backup)
- Saved PDFs:
  - Listed with metadata (focus/source/generated) and inline 1200px preview.
  - Remove any saved PDF.
- Error handling:
  - Errors/warnings appear in the app and auto-dismiss after a short time (without freezing the UI).
  - “Stop & Reset” clears only the current preview/generation state (saved PDFs remain).

---

## ⚙️ How it works

### Ingestion & Retrieval
- `src/ingest.py`: loads/chunks documents.
- `src/retriever.py`: builds/loads a Chroma vector store (`vectorstore/`) and retrieves top-k relevant chunks.

### Quiz Engine
- `src/quiz_engine.py` generates structured JSON for MCQs/short answers.
- Difficulty is either user-selected (`easy|medium|hard`) or auto (based on past accuracy).
- `src/evaluation.py` grades with `{correct: bool, feedback: str}`.

### Memory & Adaptivity
- `src/memory.py` logs:
  - sessions (topic, score, timestamp, difficulty, feedback mode),
  - per-attempt data (prompt, answer, correct, response time),
  - aggregates for frequently missed items.

### Summary Engine
- `src/summary_engine.py` orchestrates:
  - LLM call → LaTeX body,
  - standard preamble/title page (logo, colors, header/footer),
  - file naming (de-dupe),
  - `pdflatex` compilation with logs,
  - error surfacing in the app.
- `src/dynamic_latex_generator.py`:
  - prompts the LLM to produce LaTeX body only,
  - wraps it with a consistent preamble/title page,
  - sets PDF metadata title (`pdftitle`) to match the real filename.

---

## 🗂️ Project structure

```text
Student_Coach_Q-A/
  app.py                         # Streamlit UI (Upload Notes, Quiz, Progress, Summary)
  requirements.txt
  .gitignore
  16.png                         # App/logo (used in PDFs)
  data/notes/                    # Uploaded notes (ignored by git)
  generated_summaries/           # Generated PDFs & debug (ignored by git, except 16.png if present)
  vectorstore/                   # Chroma DB files (ignored by git)
  progress.json                  # Session/progress log
  src/
    config.py                    # Model settings
    ingest.py                    # Load + chunk documents
    retriever.py                 # Build/load vector store, retrieve context
    quiz_engine.py               # Generate quiz JSON (difficulty-aware)
    evaluation.py                # Grade answers with feedback
    memory.py                    # JSON memory for sessions/attempts/aggregates
    summary_engine.py            # PDF summary orchestration (naming, compile, errors)
    dynamic_latex_generator.py   # LLM → LaTeX body, standardized preamble/title page
```

---

## ⚡ Quickstart

### Requirements
- Python 3.10+
- OpenAI API key

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project root:
```env
OPENAI_API_KEY=sk-your-key-here
```

### Run the app
```bash
streamlit run app.py
```

Tabs:
- Upload Notes → add PDFs/TXT/MD (vector store rebuilds)
- Quiz → configure and start; difficulty auto or fixed
- Progress → see sessions + frequently missed
- Summary → generate professional PDF (focus optional), preview/save/download

---

## 🛠️ Troubleshooting

- PDF name at viewer top looks odd:
  - We set PDF metadata (`pdftitle`) to `summary_{pdf_name}_{focus}` so browsers show a friendly name.
- “Stop & Reset”:
  - In Summary, only clears the current preview and pending generation; it does not delete Saved PDFs.
- Duplicate save message:
  - Shows non-blocking toast; auto-dismisses shortly (no UI freeze).
- Generated outputs & uploads in git:
  - The repo ignores `generated_summaries/*` (except `16.png`) and `data/notes/*`.

---

## 💻 CLI (optional)

Run a quiz from terminal:
```bash
python -m src.main --topic "Bayes theorem" --n 4 --avoid all --feedback immediate
```

Flags:
- `--topic`: topic
- `--n`: number of questions (default 4)
- `--avoid`: `all` or `correct`
- `--feedback`: `immediate` | `end`
- `--docs`: notes folder (default `data/notes`)
- `--rebuild`: rebuild vector store
