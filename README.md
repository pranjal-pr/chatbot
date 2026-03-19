---
title: RAG Chatbot
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.11"
app_file: app.py
fullWidth: true
short_description: Grounded Q&A over the documents stored in this repository.
tags:
  - rag
  - chatbot
  - openai
  - langchain
---

# RAG Chatbot

This project is a minimal Retrieval-Augmented Generation (RAG) chatbot in Python using:

- LangChain for document loading, chunking, and orchestration
- OpenAI for embeddings and answer generation
- FAISS for local vector search
- Gradio for the Hugging Face Spaces web UI

## What it does

1. Loads `.txt`, `.md`, and `.pdf` files from `data/`
2. Splits them into chunks of 1000 characters with 200-character overlap
3. Builds a local FAISS vector index
4. Retrieves the top 3 most relevant chunks for each question
5. Passes that context to an OpenAI chat model and answers in either a terminal app or a Hugging Face Space

## Deploy to Hugging Face Spaces

1. Create a new Hugging Face Space and choose the `Gradio` SDK.
2. Push this repository to GitHub.
3. In the GitHub repo, add an Actions secret named `HF_TOKEN` with write access to the target Space.
4. In the GitHub repo, add an Actions variable named `HF_SPACE_REPO` with the value `username/space-name`.
5. In the Space `Settings` page, add `OPENAI_API_KEY` as a secret.
6. Commit your knowledge base files under `data/`.
7. Push to the `main` branch on GitHub. After CI passes, the workflow in `.github/workflows/deploy-space.yml` will force-push that branch to the Hugging Face Space.

The Space reads its configuration from environment variables, which matches Hugging Face Spaces secrets behavior.

## GitHub CI/CD

This repository includes two GitHub Actions workflows:

- `.github/workflows/ci.yml` installs dependencies, compiles the Python files, imports the Gradio app, and checks pull requests for files larger than 10 MB.
- `.github/workflows/deploy-space.yml` deploys the latest successful `main` build to the Hugging Face Space and also supports manual runs from the Actions tab.

If any tracked file exceeds 10 MB, Hugging Face recommends Git LFS for Spaces synchronization. For large PDFs or other binary assets, track them with Git LFS before pushing.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your OpenAI API key to `.env`, then place your source documents in `data/`.

## Run Locally

```powershell
python rag_chatbot.py --rebuild
```

After the first run, the FAISS index is stored locally in `vectorstore/`. Subsequent runs can reuse it:

```powershell
python rag_chatbot.py
```

## Notes

- Use `--rebuild` whenever you add or change documents.
- The script defaults to `gpt-4.1-mini` for chat and `text-embedding-3-small` for embeddings.
- Supported document types are `.txt`, `.md`, and `.pdf`.
- The Space UI is defined in `app.py`, while the shared RAG runtime lives in `rag_backend.py`.
