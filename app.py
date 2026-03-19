from __future__ import annotations

import os

import gradio as gr

from rag_backend import ConfigurationError, enable_local_fallback_from_exception, get_cached_pipeline

TITLE = "RAG Chatbot"
SUBTITLE = "Ask grounded questions over the documents stored in this Space's data folder."
THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="cyan")
EXAMPLE_QUESTIONS = [
    "What file types does this chatbot support?",
    "How does the retrieval pipeline work in this project?",
    "What should I add to the data folder to improve answers?",
]

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
    color-scheme: dark;
}

body, .gradio-container {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #ebf5f7;
    background:
        radial-gradient(circle at top left, rgba(51, 174, 169, 0.20), transparent 24%),
        radial-gradient(circle at 78% 12%, rgba(38, 99, 191, 0.22), transparent 25%),
        linear-gradient(140deg, #061118 0%, #0c1b24 46%, #12262c 100%);
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 0 1.4rem 2.2rem 1.4rem !important;
    min-height: 100vh;
    box-sizing: border-box;
}

.app-shell {
    width: min(1680px, 100%);
    margin: 0 auto;
    padding: 1.15rem 0 1.8rem 0;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 2rem;
    border: 1px solid rgba(181, 229, 230, 0.16);
    border-radius: 30px;
    background:
        linear-gradient(180deg, rgba(11, 27, 36, 0.96), rgba(7, 18, 25, 0.92)),
        linear-gradient(135deg, rgba(67, 162, 143, 0.18), rgba(42, 86, 182, 0.12));
    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.34);
}

.hero::after {
    content: "";
    position: absolute;
    inset: auto -8% -35% auto;
    width: 420px;
    height: 420px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(73, 198, 194, 0.28), rgba(73, 198, 194, 0));
    pointer-events: none;
}

.hero-grid {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: minmax(0, 1.55fr) minmax(300px, 0.95fr);
    gap: 1rem;
    align-items: stretch;
}

.hero h1 {
    margin: 0;
    font-size: clamp(2.8rem, 5vw, 4.9rem);
    letter-spacing: -0.04em;
    line-height: 0.95;
    color: #f5fcfd;
}

.eyebrow {
    margin: 0 0 0.5rem 0;
    font-size: 0.8rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #99ebe8;
}

.hero-subtitle {
    max-width: 58ch;
    margin: 1rem 0 0 0;
    font-size: 1.08rem;
    line-height: 1.7;
    color: #d4e8eb;
}

.hero-metrics {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.9rem;
}

.metric-card,
.panel {
    border: 1px solid rgba(182, 231, 233, 0.15);
    background: linear-gradient(180deg, rgba(13, 28, 36, 0.94), rgba(8, 19, 26, 0.90));
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.24);
    backdrop-filter: blur(12px);
}

.metric-card {
    min-height: 132px;
    padding: 1rem 1.05rem;
    border-radius: 22px;
}

.metric-label {
    display: block;
    font-size: 0.74rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8bd7d6;
}

.metric-value {
    display: block;
    margin-top: 0.55rem;
    font-size: 1.2rem;
    font-weight: 700;
    color: #f2fbfc;
}

.metric-copy {
    margin: 0.55rem 0 0 0;
    line-height: 1.6;
    color: #bfd7db;
}

.main-grid {
    align-items: stretch;
    gap: 1rem;
    margin-top: 1rem;
}

.panel {
    border-radius: 24px;
}

.stack-panel,
.status-panel,
.chat-frame {
    padding: 1.2rem 1.25rem;
}

.status-panel {
    min-height: 236px;
}

.stack-panel h3,
.chat-frame h3 {
    margin: 0 0 0.7rem 0;
    font-size: 1rem;
    color: #f1fbfc;
}

.stack-panel p,
.stack-panel li,
.status-panel p,
.status-panel li,
.footnote {
    color: #c9dee2;
    line-height: 1.65;
}

.stack-panel ul {
    margin: 0.7rem 0 0 0;
    padding-left: 1.1rem;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.8rem;
}

.mini-card {
    padding: 0.95rem 1rem;
    border-radius: 18px;
    background: rgba(18, 40, 50, 0.66);
    border: 1px solid rgba(154, 227, 214, 0.12);
}

.mini-card strong {
    display: block;
    color: #f3fcfd;
    margin-bottom: 0.35rem;
}

.chat-frame {
    min-height: 760px;
}

.chat-frame .gradio-container,
.chat-frame .wrap {
    padding: 0 !important;
}

.chat-frame .gr-chatbot,
.chat-frame .bubble-wrap,
.chat-frame [data-testid="chatbot"] {
    border-radius: 22px !important;
}

.chat-frame textarea,
.chat-frame input,
.chat-frame .gr-textbox,
.chat-frame [data-testid="textbox"] textarea {
    font-size: 1rem !important;
}

.chat-frame button {
    border-radius: 14px !important;
}

.footer-note {
    margin-top: 1rem;
    padding: 0.85rem 0.35rem 0 0.35rem;
}

footer {
    display: none !important;
}

@media (max-width: 1180px) {
    .hero-grid,
    .info-grid {
        grid-template-columns: 1fr;
    }

    .chat-frame {
        min-height: 680px;
    }
}

@media (max-width: 780px) {
    .gradio-container {
        padding: 0 0.85rem 1.5rem 0.85rem !important;
    }

    .app-shell {
        padding-top: 0.75rem;
    }

    .hero {
        padding: 1.35rem;
        border-radius: 24px;
    }

    .hero-metrics {
        grid-template-columns: 1fr;
    }

    .chat-frame {
        min-height: 620px;
        padding: 1rem;
    }
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


with gr.Blocks(
    title=TITLE,
    fill_width=True,
    fill_height=True,
) as demo:
    with gr.Column(elem_classes=["app-shell"]):
        gr.HTML(
            f"""
<section class="hero">
  <div class="hero-grid">
    <div class="hero-copy">
      <p class="eyebrow">Retrieval Augmented Generation</p>
      <h1>{TITLE}</h1>
      <p class="hero-subtitle">{SUBTITLE}</p>
    </div>
    <div class="hero-metrics">
      <div class="metric-card">
        <span class="metric-label">Mode</span>
        <span class="metric-value">Grounded answers</span>
        <p class="metric-copy">Responses are built from the indexed documents instead of relying on raw model memory alone.</p>
      </div>
      <div class="metric-card">
        <span class="metric-label">Sources</span>
        <span class="metric-value">Repository-backed</span>
        <p class="metric-copy">Drop `.txt`, `.md`, or `.pdf` files into `data/` and the Space will rebuild around your content.</p>
      </div>
    </div>
  </div>
</section>
"""
        )

        with gr.Row(elem_classes=["main-grid"], equal_height=True):
            with gr.Column(scale=4, min_width=320):
                status = gr.Markdown(elem_classes=["panel", "status-panel"])

                gr.HTML(
                    """
<section class="panel stack-panel">
  <h3>How To Use It</h3>
  <div class="info-grid">
    <div class="mini-card">
      <strong>Ask specific questions</strong>
      Use concrete prompts about your documents to get tighter retrieval and better answers.
    </div>
    <div class="mini-card">
      <strong>Bring real knowledge</strong>
      Replace the starter file with your own docs in `data/` for meaningful results.
    </div>
    <div class="mini-card">
      <strong>Watch the sources</strong>
      Every answer includes retrieved files so you can verify where the response came from.
    </div>
    <div class="mini-card">
      <strong>Fallback is automatic</strong>
      If OpenAI is unavailable, the app switches to local Hugging Face models instead of breaking.
    </div>
  </div>
</section>
"""
                )

            with gr.Column(scale=8, min_width=600):
                with gr.Group(elem_classes=["panel", "chat-frame"]):
                    gr.Markdown("### Ask The Knowledge Base")
                    gr.ChatInterface(
                        fn=respond,
                        chatbot=gr.Chatbot(height=560, layout="bubble", buttons=["copy"]),
                        textbox=gr.Textbox(
                            placeholder="Ask about the indexed documents...",
                            container=False,
                            scale=7,
                        ),
                        examples=EXAMPLE_QUESTIONS,
                        fill_width=True,
                        fill_height=True,
                    )

        gr.Markdown(
            "The Space reads documents from the repository `data/` folder, builds a FAISS index, and answers with retrieved context. OpenAI is used when available; otherwise the app falls back to local Hugging Face models.",
            elem_classes=["footer-note", "footnote"],
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
