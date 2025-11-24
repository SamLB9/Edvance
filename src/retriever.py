from typing import List, Optional
import os
import warnings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .config import EMBEDDING_MODEL

# Suppress deprecation warnings for better performance
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*Chroma.*deprecated.*", category=DeprecationWarning)


# Global embeddings instance to avoid repeated initialization
_embeddings_cache = None

def _get_embeddings():
    """Get cached embeddings instance"""
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings_cache

def build_or_load_vectorstore(chunks: List[Document], persist_dir: str = "vectorstore"):
    embeddings = _get_embeddings()
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

    # Simplified approach: try filtered search first, then fallback to unfiltered
    results = []
    
    if source_path:
        # Build simplified candidate list
        candidates = [source_path, os.path.basename(source_path)]
        if os.path.isabs(source_path):
            try:
                notes_dir_abs = os.path.abspath("data/notes")
                rel_notes = os.path.relpath(source_path, start=notes_dir_abs)
                candidates.extend([rel_notes, os.path.join("data", "notes", rel_notes)])
            except Exception:
                pass
        
        # Try filtered search with candidates
        for cand in candidates:
            try:
                results = vs.similarity_search(query, k=k, filter={"source": cand})
                if results:
                    try:
                        print(f"[QUIZ DEBUG] retriever: filter hit with cand='{cand}' -> {len(results)} docs")
                    except Exception:
                        pass
                    break
            except (TypeError, Exception):
                # Filter unsupported or other error, continue to next candidate
                continue
    
    # Fallback: unfiltered search
    if not results:
        try:
            results = vs.similarity_search(query, k=k)
            try:
                print(f"[QUIZ DEBUG] retriever: fallback unfiltered -> {len(results)} docs")
            except Exception:
                pass
        except Exception:
            results = []

    text = "\n\n".join([r.page_content for r in results])
    try:
        print(f"[QUIZ DEBUG] retriever: results={len(results)}, text_len={len(text)}")
    except Exception:
        pass
    return text