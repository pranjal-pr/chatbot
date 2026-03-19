# Chatbot Knowledge Base

This repository hosts a Retrieval-Augmented Generation chatbot.

## Purpose

The chatbot answers questions by retrieving relevant chunks from files stored in the `data` directory and then using an OpenAI model to generate a grounded response.

## Supported Documents

The system indexes these file types:

- `.txt`
- `.md`
- `.pdf`

## Architecture

The app follows a standard RAG pipeline:

1. Load documents from the `data` folder.
2. Split documents into chunks of 1000 characters with 200 characters of overlap.
3. Create embeddings with the OpenAI embeddings API.
4. Store and search vectors locally with FAISS.
5. Retrieve the top 3 relevant chunks for each question.
6. Generate an answer using the retrieved context.

## Deployment

The project is deployed as a Docker-based Hugging Face Space.

- GitHub Actions runs CI on pushes to `main`.
- After CI passes, a deployment workflow pushes the latest code to the Hugging Face Space.
- The Hugging Face Space uses the `OPENAI_API_KEY` secret for embeddings and answer generation.

## Example Questions

Users can ask questions like:

- What file types does the chatbot support?
- How does the retrieval pipeline work?
- Where is the app deployed?
- Which vector database is used in this project?

## Note

This is a starter knowledge file added so the deployed Space has something to index. You can replace or extend it with your own business documents, notes, PDFs, or product content.
