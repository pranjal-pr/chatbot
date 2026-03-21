---
title: RAG Chatbot
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
fullWidth: true
short_description: Grounded Q&A over the documents stored in this repository.
tags:
  - rag
  - chatbot
  - groq
  - langchain
---

# RAG Chatbot

This project is a minimal Retrieval-Augmented Generation (RAG) chatbot in Python using:

- LangChain for document loading, chunking, and orchestration
- Groq for answer generation
- Sentence Transformers for local embeddings
- FAISS for local vector search
- Gradio for the web UI, packaged inside a Docker Space

## What it does

1. Loads `.txt`, `.md`, and `.pdf` files from `data/`
2. Splits them into chunks of 1000 characters with 200-character overlap
3. Builds local sentence-transformer embeddings and stores them in a FAISS index
4. Retrieves the top 3 most relevant chunks for each question
5. Passes that context to a Groq chat model and answers in either a terminal app or a Docker-based Hugging Face Space

## Deploy to Hugging Face Spaces

1. Create a new Hugging Face Space and choose the `Docker` SDK.
2. Push this repository to GitHub.
3. In the GitHub repo, add an Actions secret named `HF_TOKEN` with write access to the target Space.
4. In the GitHub repo, add an Actions variable named `HF_SPACE_REPO` with the value `username/space-name`.
5. In the Space `Settings` page, add `GROQ_API_KEY` as a secret.
6. Commit your knowledge base files under `data/`.
7. Push to the `main` branch on GitHub. After CI passes, the workflow in `.github/workflows/deploy-space.yml` will force-push that branch to the Hugging Face Space.

The Space reads its configuration from environment variables, which matches Hugging Face Spaces secrets behavior. The Docker container serves the Gradio app on port `7860`, which matches the `app_port` configured above.

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

Add your Groq API key to `.env`, then place your source documents in `data/`.

## Run Locally

```powershell
python rag_chatbot.py --rebuild
```

After the first run, the FAISS index is stored locally in `vectorstore/`. Subsequent runs can reuse it:

```powershell
python rag_chatbot.py
```

## Run Locally with Docker

```powershell
docker build -t rag-chatbot .
docker run --rm -p 7860:7860 --env-file .env rag-chatbot
```

## Notes

- Use `--rebuild` whenever you add or change documents.
- The script defaults to Groq `llama-3.3-70b-versatile` for generation and `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- Supported document types are `.txt`, `.md`, and `.pdf`.
- The Space UI is defined in `app.py`, while the shared RAG runtime lives in `rag_backend.py`.
- `Dockerfile` is the Hugging Face Space entrypoint for the Docker deployment path.
