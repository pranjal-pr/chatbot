from __future__ import annotations

import os

import gradio as gr

from rag_backend import ConfigurationError, enable_local_fallback_from_exception, get_cached_pipeline

TITLE = "RAG Chatbot"
SUBTITLE = "Ask grounded questions over the documents stored in this Space's data folder."
THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="cyan")

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

body, .gradio-container {
    font-family: 'Space Grotesk', sans-serif !important;
    background:
        radial-gradient(circle at top left, rgba(18, 120, 120, 0.22), transparent 28%),
        radial-gradient(circle at bottom right, rgba(19, 94, 160, 0.24), transparent 32%),
        linear-gradient(135deg, #07131a 0%, #0f2330 48%, #112c2f 100%);
}

.gradio-container {
    max-width: 980px !important;
}

.hero {
    padding: 1.5rem 1.75rem 0.5rem 1.75rem;
    border: 1px solid rgba(181, 229, 230, 0.16);
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(12, 31, 39, 0.9), rgba(8, 21, 27, 0.92));
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
}

.hero h1 {
    margin: 0;
    font-size: 2.4rem;
    letter-spacing: -0.04em;
}

.eyebrow {
    margin: 0 0 0.5rem 0;
    font-size: 0.82rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #8ad7d6;
}

.hero p:last-child {
    margin-bottom: 0.3rem;
    color: #d5e8eb;
}

.status-card {
    margin-top: 1rem;
    padding: 1rem 1.1rem;
    border-radius: 18px;
    background: rgba(10, 25, 32, 0.78);
    border: 1px solid rgba(154, 227, 214, 0.14);
}
"""


def format_status() -> str:
    try:
        pipeline = get_cached_pipeline()
        status = [
            "### Status",
            f"Ready. Indexed **{pipeline.source_count}** files into **{pipeline.chunk_count}** chunks.",
            f"Runtime: `{pipeline.provider}`",
        ]
        if pipeline.note:
            status.append(pipeline.note)
        status.append(f"Chat model: `{pipeline.chat_model}`")
        status.append(f"Embedding model: `{pipeline.embedding_model}`")
        return "\n\n".join(status)
    except ConfigurationError as exc:
        return (
            "### Status\n"
            f"Blocked: {exc}\n\n"
            "Add supported documents under `data/`, or set `OPENAI_API_KEY`, or let the local fallback models initialize."
        )
    except Exception as exc:
        return f"### Status\nInitialization failed: `{exc}`"


def respond(message: str, history) -> str:
    del history
    if not message.strip():
        return "Ask a question about the indexed documents."

    try:
        result = get_cached_pipeline().ask(message)
    except ConfigurationError as exc:
        return (
            "The Space is not configured yet.\n\n"
            f"Reason: {exc}\n\n"
            "Add at least one `.txt`, `.md`, or `.pdf` file to `data/`, or provide a working `OPENAI_API_KEY`."
        )
    except Exception as exc:
        if enable_local_fallback_from_exception(exc):
            try:
                result = get_cached_pipeline().ask(message)
            except Exception as retry_exc:
                return f"Unexpected runtime error after local fallback: {retry_exc}"
        else:
            return f"Unexpected runtime error: {exc}"

    sources = "\n".join(f"- {source}" for source in result.sources) or "- No sources retrieved"
    return f"{result.answer}\n\n**Sources**\n{sources}"


with gr.Blocks(title=TITLE, fill_width=True) as demo:
    gr.Markdown(
        f"""
<div class="hero">
  <p class="eyebrow">Retrieval Augmented Generation</p>
  <h1>{TITLE}</h1>
  <p>{SUBTITLE}</p>
</div>
"""
    )

    status = gr.Markdown(elem_classes=["status-card"])

    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=460, layout="bubble", buttons=["copy"]),
        textbox=gr.Textbox(
            placeholder="Ask about the knowledge base...",
            container=False,
            scale=7,
        ),
    )

    gr.Markdown(
        "This Space reads documents from the repository `data/` folder. It uses OpenAI when available and automatically falls back to local Hugging Face models when OpenAI quota is unavailable."
    )

    demo.load(fn=format_status, outputs=status)


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=THEME,
        css=CUSTOM_CSS,
    )
