from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = PROJECT_ROOT / "data"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "vectorstore"
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

PROMPT_TEMPLATE = """You are a helpful retrieval-augmented assistant.
Answer the user's question using only the provided context.
If the answer is not supported by the context, say you do not know based on the indexed documents.

Context:
{context}

Question: {question}
"""


class ConfigurationError(RuntimeError):
    """Raised when the runtime is missing required configuration."""


@dataclass(slots=True)
class AnswerResult:
    answer: str
    sources: list[str]


@dataclass(slots=True)
class RAGPipeline:
    retriever: Any
    generation_chain: Any
    chat_model: str
    embedding_model: str
    source_count: int
    chunk_count: int

    def ask(self, question: str) -> AnswerResult:
        documents = self.retriever.invoke(question)
        answer = self.generation_chain.invoke(
            {
                "question": question,
                "context": format_context(documents),
            }
        )

        seen: set[str] = set()
        sources: list[str] = []
        for document in documents:
            source = describe_source(document)
            if source not in seen:
                seen.add(source)
                sources.append(source)

        return AnswerResult(answer=answer, sources=sources)


def load_environment() -> None:
    load_dotenv()


def ensure_openai_api_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    raise ConfigurationError(
        "OPENAI_API_KEY is not set. Add it locally in .env or as a Hugging Face Space secret."
    )


def count_source_files(docs_dir: Path) -> int:
    if not docs_dir.exists():
        raise ConfigurationError(f"Document directory does not exist: {docs_dir}")

    patterns = ("*.txt", "*.md", "*.pdf")
    return sum(1 for pattern in patterns for _ in docs_dir.rglob(pattern))


def load_documents(docs_dir: Path) -> list[Document]:
    if not docs_dir.exists():
        raise ConfigurationError(f"Document directory does not exist: {docs_dir}")

    loaders = [
        DirectoryLoader(
            str(docs_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            recursive=True,
            silent_errors=True,
        ),
        DirectoryLoader(
            str(docs_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            recursive=True,
            silent_errors=True,
        ),
        DirectoryLoader(
            str(docs_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            recursive=True,
            silent_errors=True,
        ),
    ]

    documents: list[Document] = []
    for loader in loaders:
        documents.extend(loader.load())

    if not documents:
        raise ConfigurationError(
            f"No supported documents found in {docs_dir}. Add .txt, .md, or .pdf files before starting the app."
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def load_or_build_vectorstore(
    docs_dir: Path,
    index_dir: Path,
    embeddings: OpenAIEmbeddings,
    rebuild: bool,
) -> tuple[FAISS, int, int]:
    faiss_file = index_dir / "index.faiss"
    pickle_file = index_dir / "index.pkl"
    source_count = count_source_files(docs_dir)

    if source_count == 0:
        raise ConfigurationError(
            f"No supported documents found in {docs_dir}. Add .txt, .md, or .pdf files before starting the app."
        )

    if not rebuild and faiss_file.exists() and pickle_file.exists():
        vectorstore = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore, source_count, int(vectorstore.index.ntotal)

    documents = load_documents(docs_dir)
    chunks = split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore, source_count, len(chunks)


def describe_source(document: Document) -> str:
    source = document.metadata.get("source", "unknown")
    label = Path(source).name if source else "unknown"
    page = document.metadata.get("page")
    if isinstance(page, int):
        return f"{label} (page {page + 1})"
    return label


def format_context(documents: Iterable[Document]) -> str:
    parts: list[str] = []
    for document in documents:
        parts.append(f"Source: {describe_source(document)}\n{document.page_content.strip()}")
    return "\n\n".join(parts)


def build_generation_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt | llm | StrOutputParser()


def create_pipeline(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
    rebuild: bool = False,
) -> RAGPipeline:
    load_environment()
    ensure_openai_api_key()

    chat_model = os.getenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL)
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    embeddings = OpenAIEmbeddings(model=embedding_model)
    llm = ChatOpenAI(model=chat_model, temperature=0)
    vectorstore, source_count, chunk_count = load_or_build_vectorstore(
        docs_dir=docs_dir,
        index_dir=index_dir,
        embeddings=embeddings,
        rebuild=rebuild,
    )

    return RAGPipeline(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        generation_chain=build_generation_chain(llm),
        chat_model=chat_model,
        embedding_model=embedding_model,
        source_count=source_count,
        chunk_count=chunk_count,
    )


@lru_cache(maxsize=1)
def get_cached_pipeline() -> RAGPipeline:
    return create_pipeline()
