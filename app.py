from __future__ import annotations

import os

import gradio as gr

from rag_backend import ConfigurationError, enable_local_fallback_from_exception, get_cached_pipeline

TITLE = "RAG Chatbot"
SUBTITLE = "Ask grounded questions over the documents stored in this Space's data folder."
THEME = gr.themes.Soft(primary_hue="fuchsia", secondary_hue="purple")
PROMPT_CHIPS = [
    ("Create Image", "Summarize the indexed documents visually and suggest image directions."),
    ("Brainstorm", "Brainstorm the strongest ideas from the indexed documents."),
    ("Make a plan", "Make a clear plan using only the indexed documents as context."),
]

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
    color-scheme: dark;
    --page-lilac: #d8b3dc;
    --page-lilac-deep: #c99bcd;
    --panel-black: #060607;
    --panel-black-soft: #121014;
    --stage-top: #221026;
    --stage-bottom: #120811;
    --text-main: #fff8ff;
    --text-soft: #c8b8cb;
    --text-dim: #9f93a4;
    --line: rgba(242, 204, 255, 0.16);
    --accent: #b649ff;
    --accent-deep: #8228f2;
}

html, body {
    min-height: 100%;
}

body,
.gradio-container {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-main);
    background: linear-gradient(180deg, var(--page-lilac) 0%, #ddb9e3 100%) !important;
}

.gradio-container {
    position: relative;
    z-index: 1;
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 1.25rem 1.25rem 2rem 1.25rem !important;
    box-sizing: border-box;
}

body::before,
body::after {
    content: "";
    position: fixed;
    pointer-events: none;
    z-index: 0;
}

body::before {
    top: -18rem;
    left: 12%;
    width: 82rem;
    height: 34rem;
    border-radius: 50%;
    background: radial-gradient(circle at center, rgba(92, 19, 91, 0.96) 0%, rgba(92, 19, 91, 0.88) 42%, rgba(92, 19, 91, 0) 74%);
    transform: rotate(-7deg);
}

body::after {
    right: -14rem;
    bottom: -10rem;
    width: 34rem;
    height: 34rem;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(126, 42, 160, 0.24), rgba(126, 42, 160, 0));
}

.app-shell {
    width: min(1780px, 100%);
    margin: 0 auto;
}

.workspace-grid {
    align-items: start;
    gap: 1.1rem;
}

.feature-column,
.sidebar-column {
    align-self: start;
}

.feature-stack,
.sidebar-shell,
.status-shell,
.stage-shell,
.chat-history {
    border: 1px solid var(--line);
    box-shadow: 0 24px 80px rgba(14, 5, 17, 0.24);
}

.feature-stack {
    position: sticky;
    top: 1rem;
    display: grid;
    gap: 1rem;
}

.rail-card {
    min-height: 11.5rem;
    padding: 1.1rem;
    border-radius: 1.85rem;
    background: linear-gradient(180deg, rgba(7, 7, 8, 0.98), rgba(11, 10, 13, 0.98));
}

.rail-card h3 {
    margin: 2.55rem 0 0.7rem 0;
    font-size: 1.15rem;
}

.rail-card p {
    margin: 0;
    line-height: 1.65;
    color: var(--text-soft);
}

.rail-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 7rem;
    padding: 0.45rem 0.85rem;
    border-radius: 999px;
    background: #2a2730;
    color: #f4edf6;
    font-size: 0.82rem;
}

.sidebar-column {
    position: sticky;
    top: 1rem;
}

.sidebar-shell {
    padding: 1.2rem;
    border-radius: 2rem;
    background: linear-gradient(180deg, rgba(5, 5, 6, 0.99), rgba(10, 9, 11, 0.98));
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1rem;
}

.brand-mark {
    width: 2.35rem;
    height: 2.35rem;
    border-radius: 999px;
    background: radial-gradient(circle at 35% 35%, #ffffff, #b79ac4 32%, #3b3244 33%, #090909 58%);
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.08);
}

.brand-name {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-main);
}

.brand-copy {
    margin: 0.18rem 0 0 0;
    color: var(--text-dim);
    line-height: 1.5;
}

.new-chat-btn button {
    width: 100%;
    justify-content: flex-start !important;
    border-radius: 1rem !important;
    background: #2c2930 !important;
    color: #f1ecf3 !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    box-shadow: none !important;
}

.sidebar-label {
    margin: 1.15rem 0 0.65rem 0;
    font-size: 0.83rem;
    color: var(--text-dim);
    letter-spacing: 0.02em;
}

.sidebar-list {
    display: grid;
    gap: 0.35rem;
}

.sidebar-item {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.45rem 0;
    color: var(--text-soft);
}

.sidebar-divider {
    margin: 1rem 0 0.6rem 0;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.status-shell {
    margin-top: 1rem;
    padding: 1rem 1.05rem;
    border-radius: 1.7rem;
    background: linear-gradient(180deg, rgba(9, 9, 10, 0.98), rgba(14, 11, 16, 0.98));
}

.status-shell p,
.status-shell li {
    color: var(--text-soft);
    line-height: 1.6;
}

.stage-shell {
    position: relative;
    overflow: hidden;
    padding: 1.1rem;
    border-radius: 2.3rem;
    background: radial-gradient(circle at 50% 36%, rgba(144, 67, 182, 0.18), rgba(144, 67, 182, 0) 25%), linear-gradient(180deg, var(--stage-top) 0%, var(--stage-bottom) 100%);
}

.stage-shell::before {
    content: "";
    position: absolute;
    inset: auto -6rem -7rem auto;
    width: 18rem;
    height: 18rem;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(142, 59, 178, 0.14), rgba(142, 59, 178, 0));
}

.stage-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
}

.stage-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.55rem 1.05rem;
    border-radius: 999px;
    background: rgba(3, 3, 4, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.06);
    color: var(--text-main);
    font-size: 0.92rem;
}

.stage-hero {
    padding: 3.2rem 0 1.65rem 0;
    text-align: center;
}

.orb {
    width: 5.3rem;
    height: 5.3rem;
    margin: 0 auto 1.25rem auto;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 28%, rgba(255, 215, 255, 0.95), rgba(255, 215, 255, 0.12) 24%, rgba(150, 98, 255, 0.88) 43%, rgba(88, 22, 135, 0.95) 68%, rgba(27, 11, 48, 0.92) 100%);
    box-shadow: 0 0 0 10px rgba(180, 85, 255, 0.05), 0 0 42px rgba(198, 103, 255, 0.34);
}

.stage-title {
    margin: 0;
    font-size: clamp(2rem, 4vw, 3.3rem);
    line-height: 1.08;
    letter-spacing: -0.04em;
    color: var(--text-main);
}

.stage-subtitle {
    max-width: 44rem;
    margin: 0.85rem auto 0 auto;
    color: var(--text-soft);
    font-size: 1.02rem;
    line-height: 1.7;
}

.composer-shell {
    width: min(100%, 56rem);
    margin: 0 auto;
    padding: 1rem;
    border-radius: 1.9rem;
    background: linear-gradient(180deg, rgba(35, 25, 38, 0.96), rgba(25, 18, 28, 0.96));
    border: 1px solid rgba(246, 198, 255, 0.20);
}

.chip-row {
    gap: 0.7rem;
    margin-bottom: 0.85rem;
}

.chip-btn button {
    border-radius: 999px !important;
    background: rgba(7, 7, 8, 0.82) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    color: var(--text-main) !important;
    box-shadow: none !important;
}

.composer-row {
    align-items: end;
    gap: 0.8rem;
}

.prompt-shell {
    flex: 1 1 auto;
    padding: 0.1rem 0.35rem 0.35rem 0.35rem;
    border-radius: 1.5rem;
    background: rgba(26, 19, 29, 0.98);
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.composer-box textarea {
    min-height: 7rem !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-main) !important;
    font-size: 1.15rem !important;
    line-height: 1.55 !important;
}

.composer-box textarea::placeholder {
    color: #9e91a4 !important;
}

.send-btn button {
    width: 4rem;
    min-width: 4rem !important;
    height: 4rem;
    border-radius: 999px !important;
    background: linear-gradient(180deg, var(--accent) 0%, var(--accent-deep) 100%) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    box-shadow: 0 12px 30px rgba(158, 65, 246, 0.32) !important;
}

.utility-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1.2rem;
    margin-top: 0.95rem;
    padding-top: 0.85rem;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    color: var(--text-dim);
    font-size: 0.93rem;
}

.chat-history {
    margin-top: 1rem;
    border-radius: 1.9rem;
    background: linear-gradient(180deg, rgba(12, 10, 14, 0.96), rgba(9, 8, 11, 0.98));
}

.chat-history [data-testid="chatbot"] {
    border-radius: 1.6rem !important;
    background: transparent !important;
}

.footer-note {
    margin-top: 0.9rem;
    text-align: center;
    color: #6b4f6c;
    font-size: 0.93rem;
}

footer {
    display: none !important;
}

@media (max-width: 1500px) {
    .feature-column {
        display: none !important;
    }
}

@media (max-width: 1120px) {
    .sidebar-column {
        position: static;
    }
}

@media (max-width: 860px) {
    .gradio-container {
        padding: 0.85rem 0.85rem 1.5rem 0.85rem !important;
    }

    .stage-hero {
        padding-top: 2rem;
    }

    .stage-title {
        font-size: 2rem;
    }

    .composer-shell {
        padding: 0.8rem;
    }

    .composer-row {
        flex-direction: column;
        align-items: stretch;
    }

    .send-btn button {
        width: 100%;
        min-width: 100% !important;
        height: 3.5rem;
        border-radius: 1rem !important;
    }
}
"""

def format_status() -> str:
    try:
        pipeline = get_cached_pipeline()
        status = [
            "### Runtime Status",
            f"Ready. Indexed **{pipeline.source_count}** files into **{pipeline.chunk_count}** chunks.",
            f"Provider: `{pipeline.provider}`",
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
        return f"### Runtime Status\nInitialization failed: `{exc}`"


def answer_question(message: str) -> str:
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


def submit_chat(message: str, history: list[dict] | None) -> tuple[list[dict], str]:
    history = history or []
    if not message.strip():
        return history, ""

    answer = answer_question(message)
    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return updated_history, ""


def clear_chat() -> tuple[list[dict], str]:
    return [], ""


with gr.Blocks(
    title=TITLE,
    fill_width=True,
    fill_height=True,
) as demo:
    with gr.Column(elem_classes=["app-shell"]):
        with gr.Row(elem_classes=["workspace-grid"], equal_height=False):
            with gr.Column(scale=2, min_width=240, elem_classes=["feature-column"]):
                gr.HTML(
                    """
<div class="feature-stack">
  <div class="rail-card">
    <span class="rail-pill">Create Image</span>
    <h3>Image Generator</h3>
    <p>Turn document context into visual concepts, references, and directions.</p>
  </div>
  <div class="rail-card">
    <span class="rail-pill">Make Slides</span>
    <h3>Presentation</h3>
    <p>Condense your indexed material into clean narrative structure and talking points.</p>
  </div>
  <div class="rail-card">
    <span class="rail-pill">Generate Code</span>
    <h3>Dev Assistant</h3>
    <p>Use the same knowledge base to answer implementation and architecture questions.</p>
  </div>
</div>
"""
                )

            with gr.Column(scale=3, min_width=300, elem_classes=["sidebar-column"]):
                with gr.Group(elem_classes=["sidebar-shell"]):
                    gr.HTML(
                        """
<div class="brand-row">
  <div class="brand-mark"></div>
  <div>
    <p class="brand-name">RAG Studio</p>
    <p class="brand-copy">A dark AI workspace grounded on the documents inside this Space.</p>
  </div>
</div>

<p class="sidebar-label">Features</p>
<div class="sidebar-list">
  <div class="sidebar-item">Chat</div>
  <div class="sidebar-item">Archived</div>
  <div class="sidebar-item">Library</div>
</div>

<div class="sidebar-divider"></div>

<p class="sidebar-label">Workspaces</p>
<div class="sidebar-list">
  <div class="sidebar-item">New Project</div>
  <div class="sidebar-item">Image</div>
  <div class="sidebar-item">Presentation</div>
  <div class="sidebar-item">Research</div>
</div>
"""
                    )

                    new_chat = gr.Button("+ New Chat", elem_classes=["new-chat-btn"], variant="secondary")

                status = gr.Markdown(elem_classes=["status-shell"])

            with gr.Column(scale=7, min_width=760):
                with gr.Group(elem_classes=["stage-shell"]):
                    gr.HTML(
                        f"""
<div class="stage-topbar">
  <div class="stage-pill">ChatGPT v4.0</div>
</div>

<div class="stage-hero">
  <div class="orb"></div>
  <h1 class="stage-title">Ready to Create Something New?</h1>
  <p class="stage-subtitle">{SUBTITLE}</p>
</div>
"""
                    )

                    with gr.Group(elem_classes=["composer-shell"]):
                        with gr.Row(elem_classes=["chip-row"], equal_height=False):
                            chip_buttons = []
                            for label, _ in PROMPT_CHIPS:
                                chip_buttons.append(
                                    gr.Button(label, elem_classes=["chip-btn"], size="sm", variant="secondary")
                                )

                        with gr.Row(elem_classes=["composer-row"], equal_height=False):
                            with gr.Group(elem_classes=["prompt-shell"]):
                                message_box = gr.Textbox(
                                    placeholder="Ask Anything...",
                                    lines=3,
                                    max_lines=6,
                                    container=False,
                                    autofocus=True,
                                    submit_btn=False,
                                    stop_btn=False,
                                    elem_classes=["composer-box"],
                                )
                            send_button = gr.Button("Send", elem_classes=["send-btn"], variant="primary")

                        gr.HTML(
                            """
<div class="utility-row">
  <span>Attach</span>
  <span>Settings</span>
  <span>Options</span>
</div>
"""
                        )

                    chatbot = gr.Chatbot(
                        [],
                        height=360,
                        layout="bubble",
                        buttons=["copy"],
                        placeholder="No messages yet. Ask about the indexed documents.",
                        elem_classes=["chat-history"],
                    )

        gr.Markdown(
            "This interface is styled after the shared reference while still running the same RAG pipeline over your repository documents.",
            elem_classes=["footer-note"],
        )

        for button, (_, prompt_text) in zip(chip_buttons, PROMPT_CHIPS):
            button.click(lambda value=prompt_text: value, outputs=message_box)

        send_button.click(submit_chat, inputs=[message_box, chatbot], outputs=[chatbot, message_box])
        message_box.submit(submit_chat, inputs=[message_box, chatbot], outputs=[chatbot, message_box])
        new_chat.click(clear_chat, outputs=[chatbot, message_box])
        demo.load(fn=format_status, outputs=status)


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=THEME,
        css=CUSTOM_CSS,
    )
