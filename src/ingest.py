from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

def load_documents(notes_dir: str) -> List[Document]:
    docs = []
    for p in Path(notes_dir).glob("**/*"):
        if p.suffix.lower() in [".pdf"]:
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower() in [".txt", ".md"]:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
    return docs

def chunk_documents(docs: List[Document], chunk_size=1200, chunk_overlap=150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)