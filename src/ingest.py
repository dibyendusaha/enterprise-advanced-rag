import pickle
from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.config import (
    RAW_DOCS_DIR,
    CHROMA_DIR,
    BM25_INDEX_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)


def _load_pdf(file_path: Path) -> list[Document]:
    reader = PdfReader(str(file_path))
    docs = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "file_type": "pdf",
                        "page": page_idx,
                    },
                )
            )
    return docs


def _load_docx(file_path: Path) -> list[Document]:
    doc = DocxDocument(str(file_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)
    if not full_text:
        return []
    return [
        Document(
            page_content=full_text,
            metadata={
                "source": file_path.name,
                "file_type": "docx",
                "page": 1,
            },
        )
    ]


def _load_text(file_path: Path) -> list[Document]:
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "file_type": file_path.suffix.lstrip(".").lower(),
                "page": 1,
            },
        )
    ]


def load_documents(docs_dir: Path = RAW_DOCS_DIR) -> list[Document]:
    """Loads .pdf, .docx, .md, and .txt files using standalone native libraries."""
    documents: list[Document] = []
    for file_path in sorted(docs_dir.iterdir()):
        if file_path.is_dir():
            continue

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            documents.extend(_load_pdf(file_path))
        elif ext == ".docx":
            documents.extend(_load_docx(file_path))
        elif ext in {".md", ".txt"}:
            documents.extend(_load_text(file_path))
        else:
            print(f"Skipping unsupported file format: {file_path.name}")

    return documents


def run_ingestion():
    print(f"Loading documents from {RAW_DOCS_DIR}...")
    raw_docs = load_documents()
    if not raw_docs:
        raise RuntimeError(
            f"No supported documents found in {RAW_DOCS_DIR}. Run `python -m src.generate_data` first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}::chunk_{idx}"

    print(f"Split {len(raw_docs)} document pages/sections into {len(chunks)} chunks.")

    # 1. Persist Dense Vector Store (Standalone langchain-chroma + langchain-openai)
    CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="enterprise_policies",
    )
    print(f"ChromaDB persisted at {CHROMA_DIR} (count={vectorstore._collection.count()})")

    # 2. Persist Chunk Corpus for Standalone rank_bm25 Retrieval
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"BM25 chunk corpus saved at {BM25_INDEX_PATH}")


if __name__ == "__main__":
    run_ingestion()