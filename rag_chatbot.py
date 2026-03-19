from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_backend import (
    DEFAULT_DOCS_DIR,
    DEFAULT_INDEX_DIR,
    ConfigurationError,
    RAGPipeline,
    create_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple RAG chatbot with LangChain, FAISS, and OpenAI.")
    parser.add_argument(
        "--docs",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help="Directory containing source documents (.txt, .md, .pdf).",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory where the FAISS index will be stored.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the FAISS index from the source documents.",
    )
    return parser.parse_args()


def chat_loop(pipeline: RAGPipeline) -> None:
    print("\nRAG chatbot ready.")
    print("Type a question, or enter 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit"}:
            print("Exiting.")
            break

        result = pipeline.ask(question)

        print(f"\nBot: {result.answer}\n")
        print("Sources:")
        for index, source in enumerate(result.sources, start=1):
            print(f"{index}. {source}")
        print()


def main() -> None:
    args = parse_args()
    try:
        pipeline = create_pipeline(
            docs_dir=args.docs,
            index_dir=args.index,
            rebuild=args.rebuild,
        )
    except ConfigurationError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc

    chat_loop(pipeline)


if __name__ == "__main__":
    main()
