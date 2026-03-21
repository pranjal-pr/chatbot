from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = PROJECT_ROOT / "data"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "vectorstore"
DEFAULT_CHAT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LOCAL_CHAT_MODEL = "google/flan-t5-small"
DEFAULT_LOCAL_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
DEFAULT_PROVIDER = "groq"

PROMPT_TEMPLATE = """You are a helpful retrieval-augmented assistant.
Answer the user's question using only the provided context.
If the answer is not supported by the context, say you do not know based on the indexed documents.

Context:
{context}

Question: {question}
"""

CHAT_PROMPT_TEMPLATE = """You are a concise, helpful assistant.
Answer the user's question directly and clearly.
Do not repeat the question back to the user.

Question: {question}

Answer:
"""

_LOCAL_FALLBACK_REASON: str | None = None


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
    provider: str
    note: str | None = None

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


@dataclass(slots=True)
class ChatRuntime:
    generation_chain: Any
    chat_model: str
    provider: str
    note: str | None = None

    def ask(self, question: str) -> str:
        return self.generation_chain.invoke({"question": question})


def load_environment() -> None:
    load_dotenv()


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


def sanitize_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()


def build_index_dir(index_root: Path, provider: str, embedding_model: str) -> Path:
    return index_root / sanitize_slug(provider) / sanitize_slug(embedding_model)


def load_or_build_vectorstore(
    docs_dir: Path,
    index_root: Path,
    embeddings: Any,
    rebuild: bool,
    provider: str,
    embedding_model: str,
) -> tuple[FAISS, int, int]:
    index_dir = build_index_dir(index_root, provider=provider, embedding_model=embedding_model)
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


def build_generation_chain(llm: Any, template: str = PROMPT_TEMPLATE, use_chat_prompt: bool = True):
    prompt = ChatPromptTemplate.from_template(template) if use_chat_prompt else PromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()


def groq_api_key_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def summarize_exception(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return message[:280] if len(message) > 280 else message


def humanize_groq_issue(exc: Exception) -> str:
    message = summarize_exception(exc).lower()
    if "insufficient_quota" in message or "quota" in message or "billing" in message:
        return "Groq quota is unavailable"
    if (
        "invalid_api_key" in message
        or "incorrect api key" in message
        or "authentication" in message
        or "unauthorized" in message
        or "api key" in message
    ):
        return "Groq authentication failed"
    if "permission" in message or "forbidden" in message or "403" in message:
        return "Groq model access failed"
    if "rate limit" in message or "error code: 429" in message:
        return "Groq rate limit was reached"
    return "Groq is unavailable"


def should_fallback_to_local(exc: Exception) -> bool:
    message = summarize_exception(exc).lower()
    markers = (
        "insufficient_quota",
        "error code: 429",
        "rate limit",
        "quota",
        "billing",
        "incorrect api key",
        "authentication",
        "invalid_api_key",
        "unauthorized",
        "forbidden",
        "permission",
        "groq",
    )
    return any(marker in message for marker in markers)


def prefer_local_runtime(reason: str) -> None:
    global _LOCAL_FALLBACK_REASON
    _LOCAL_FALLBACK_REASON = reason


def clear_local_runtime_preference() -> None:
    global _LOCAL_FALLBACK_REASON
    _LOCAL_FALLBACK_REASON = None


def build_groq_llm(chat_model: str) -> Any:
    return ChatGroq(model=chat_model, temperature=0)


@lru_cache(maxsize=2)
def load_local_model_components(chat_model: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(chat_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(chat_model)
    model.eval()
    return tokenizer, model


def build_local_llm(chat_model: str) -> Any:
    tokenizer, model = load_local_model_components(chat_model)

    def local_generate(prompt_value: Any) -> str:
        prompt_text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=196,
                do_sample=False,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return RunnableLambda(local_generate)


def build_groq_pipeline(docs_dir: Path, index_root: Path, rebuild: bool) -> RAGPipeline:
    chat_model = os.getenv("GROQ_MODEL", DEFAULT_CHAT_MODEL)
    embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
    )
    llm = build_groq_llm(chat_model)
    vectorstore, source_count, chunk_count = load_or_build_vectorstore(
        docs_dir=docs_dir,
        index_root=index_root,
        embeddings=embeddings,
        rebuild=rebuild,
        provider="groq",
        embedding_model=embedding_model,
    )

    return RAGPipeline(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        generation_chain=build_generation_chain(llm),
        chat_model=chat_model,
        embedding_model=embedding_model,
        source_count=source_count,
        chunk_count=chunk_count,
        provider="groq",
    )


def build_local_pipeline(docs_dir: Path, index_root: Path, rebuild: bool, note: str | None = None) -> RAGPipeline:
    chat_model = os.getenv("LOCAL_CHAT_MODEL", DEFAULT_LOCAL_CHAT_MODEL)
    embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
    )
    llm = build_local_llm(chat_model)

    vectorstore, source_count, chunk_count = load_or_build_vectorstore(
        docs_dir=docs_dir,
        index_root=index_root,
        embeddings=embeddings,
        rebuild=rebuild,
        provider="local",
        embedding_model=embedding_model,
    )

    return RAGPipeline(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        generation_chain=build_generation_chain(llm, use_chat_prompt=False),
        chat_model=chat_model,
        embedding_model=embedding_model,
        source_count=source_count,
        chunk_count=chunk_count,
        provider="local",
        note=note,
    )


def build_groq_chat_runtime() -> ChatRuntime:
    chat_model = os.getenv("GROQ_MODEL", DEFAULT_CHAT_MODEL)
    llm = build_groq_llm(chat_model)
    return ChatRuntime(
        generation_chain=build_generation_chain(llm, CHAT_PROMPT_TEMPLATE),
        chat_model=chat_model,
        provider="groq",
    )


def build_local_chat_runtime(note: str | None = None) -> ChatRuntime:
    chat_model = os.getenv("LOCAL_CHAT_MODEL", DEFAULT_LOCAL_CHAT_MODEL)
    llm = build_local_llm(chat_model)
    return ChatRuntime(
        generation_chain=build_generation_chain(llm, CHAT_PROMPT_TEMPLATE, use_chat_prompt=False),
        chat_model=chat_model,
        provider="local",
        note=note,
    )


def create_pipeline(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
    rebuild: bool = False,
) -> RAGPipeline:
    load_environment()
    provider = os.getenv("RAG_PROVIDER", DEFAULT_PROVIDER).strip().lower()

    if _LOCAL_FALLBACK_REASON and provider in {"groq", "auto"}:
        return build_local_pipeline(docs_dir=docs_dir, index_root=index_dir, rebuild=rebuild, note=_LOCAL_FALLBACK_REASON)

    if provider == "local":
        return build_local_pipeline(
            docs_dir=docs_dir,
            index_root=index_dir,
            rebuild=rebuild,
            note="Local runtime selected via RAG_PROVIDER=local.",
        )

    if provider not in {"groq", "auto"}:
        raise ConfigurationError("RAG_PROVIDER must be one of: groq, auto, local.")

    if not groq_api_key_available():
        return build_local_pipeline(
            docs_dir=docs_dir,
            index_root=index_dir,
            rebuild=rebuild,
            note="GROQ_API_KEY is not set. Using local fallback models.",
        )

    try:
        clear_local_runtime_preference()
        return build_groq_pipeline(docs_dir=docs_dir, index_root=index_dir, rebuild=rebuild)
    except Exception as exc:
        if not should_fallback_to_local(exc):
            raise

        reason = f"{humanize_groq_issue(exc)}. Using local fallback models."
        prefer_local_runtime(reason)
        return build_local_pipeline(docs_dir=docs_dir, index_root=index_dir, rebuild=rebuild, note=reason)


@lru_cache(maxsize=1)
def get_cached_pipeline() -> RAGPipeline:
    return create_pipeline()


def reset_cached_pipeline() -> None:
    get_cached_pipeline.cache_clear()


def create_chat_runtime() -> ChatRuntime:
    load_environment()
    provider = os.getenv("RAG_PROVIDER", DEFAULT_PROVIDER).strip().lower()

    if _LOCAL_FALLBACK_REASON and provider in {"groq", "auto"}:
        return build_local_chat_runtime(_LOCAL_FALLBACK_REASON)

    if provider == "local":
        return build_local_chat_runtime("Local runtime selected via RAG_PROVIDER=local.")

    if provider not in {"groq", "auto"}:
        raise ConfigurationError("RAG_PROVIDER must be one of: groq, auto, local.")

    if not groq_api_key_available():
        return build_local_chat_runtime("GROQ_API_KEY is not set. Using local fallback models.")

    try:
        clear_local_runtime_preference()
        return build_groq_chat_runtime()
    except Exception as exc:
        if not should_fallback_to_local(exc):
            raise

        reason = f"{humanize_groq_issue(exc)}. Using local fallback models."
        prefer_local_runtime(reason)
        return build_local_chat_runtime(reason)


@lru_cache(maxsize=1)
def get_cached_chat_runtime() -> ChatRuntime:
    return create_chat_runtime()


def reset_cached_chat_runtime() -> None:
    get_cached_chat_runtime.cache_clear()


def enable_local_fallback_from_exception(exc: Exception) -> bool:
    if not should_fallback_to_local(exc):
        return False

    prefer_local_runtime(f"{humanize_groq_issue(exc)}. Using local fallback models.")
    reset_cached_pipeline()
    reset_cached_chat_runtime()
    return True
