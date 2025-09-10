from typing import List, Optional
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from .config import EMBEDDING_MODEL


def build_or_load_vectorstore(chunks: List[Document], persist_dir: str = "vectorstore"):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    if chunks:
        vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
        try:
            vs.persist()
        except Exception:
            pass
        return vs
    # Fallback: load existing
    return Chroma(embedding_function=embeddings, persist_directory=persist_dir)


def retrieve_context(vs, topic: str, k: int = 6, source_path: Optional[str] = None) -> str:
    query = topic.strip() or "core concepts"
    try:
        print(f"[QUIZ DEBUG] retriever: query='{query}', k={k}, source_path={(source_path or 'None')}")
    except Exception:
        pass

    # Build candidate source keys to match how metadata['source'] might have been stored
    candidates: List[str] = []
    if source_path:
        try:
            abs_path = os.path.abspath(source_path)
            candidates.append(abs_path)
            # Common relative forms used by loaders
            notes_dir_abs = os.path.abspath("data/notes")
            rel_notes = os.path.relpath(abs_path, start=notes_dir_abs)  # e.g., 'BayesTheorem.pdf' or subdir/file
            # 1) path relative to notes root
            candidates.append(rel_notes)
            # 2) typical project-relative path
            candidates.append(os.path.join("data", "notes", rel_notes))
            # 3) include project folder prefix (e.g., 'Student_Coach_Q-A/data/notes/file.pdf')
            project_dir = os.path.basename(os.path.abspath(os.path.join(notes_dir_abs, os.pardir, os.pardir)))
            candidates.append(os.path.join(project_dir, "data", "notes", rel_notes))
            # 4) include parent + project folder (e.g., '../Student_Coach_Q-A/data/notes/file.pdf')
            candidates.append(os.path.join("..", project_dir, "data", "notes", rel_notes))
            # 5) Basename as last resort
            candidates.append(os.path.basename(abs_path))
        except Exception:
            candidates.append(source_path)

    def _try_filtered(_k: int) -> List:
        # Try chroma filter for each candidate; stop at first with hits
        if not candidates:
            return []
        for cand in candidates:
            try:
                hits = vs.similarity_search(query, k=_k, filter={"source": cand})
                if hits:
                    try:
                        print(f"[QUIZ DEBUG] retriever: filter hit with cand='{cand}' -> {len(hits)} docs")
                    except Exception:
                        pass
                    return hits
            except TypeError:
                # Filter unsupported by this chroma version
                break
            except Exception:
                continue
        return []

    # 1) Try exact/relative filtered search first
    results = _try_filtered(k)

    # 2) If still empty, do a broader search then post-filter by metadata
    if not results:
        try:
            broader = vs.similarity_search(query, k=max(k * 3, k + 10))
        except Exception:
            broader = []
        if candidates and broader:
            filtered = []
            for d in broader:
                src = (d.metadata or {}).get("source")
                if not src:
                    continue
                for cand in candidates:
                    try:
                        if os.path.isabs(cand) and os.path.isabs(src):
                            if os.path.normpath(src) == os.path.normpath(cand):
                                filtered.append(d)
                                break
                        else:
                            if src.endswith(cand):
                                filtered.append(d)
                                break
                    except Exception:
                        if str(src).endswith(str(cand)):
                            filtered.append(d)
                            break
            results = filtered[:k] if filtered else []
            try:
                print(f"[QUIZ DEBUG] retriever: post-filtered broader -> {len(results)} docs")
            except Exception:
                pass

    # 3) Fallback: unfiltered search
    if not results:
        try:
            results = vs.similarity_search(query, k=k)
            print(f"[QUIZ DEBUG] retriever: fallback unfiltered -> {len(results)} docs")
        except Exception:
            results = []

    text = "\n\n".join([r.page_content for r in results])
    try:
        print(f"[QUIZ DEBUG] retriever: results={len(results)}, text_len={len(text)}")
    except Exception:
        pass
    return text