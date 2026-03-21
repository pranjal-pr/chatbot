from __future__ import annotations

import os
from html import escape
from pathlib import Path

import gradio as gr

from rag_backend import ConfigurationError, enable_local_fallback_from_exception, get_cached_pipeline

TITLE = "RAG Chatbot"
SUBTITLE = "Ask grounded questions over the documents stored in this Space's data folder."
THEME = gr.themes.Base(primary_hue="violet", secondary_hue="slate", neutral_hue="slate").set(
    body_background_fill="#050507",
    body_text_color="#f5f7fb",
    body_text_color_subdued="#aeb5c2",
    background_fill_primary="#0f1013",
    background_fill_secondary="#14161a",
    block_background_fill="#0f1013",
    block_border_color="#242833",
    block_label_text_color="#f5f7fb",
    input_background_fill="#13151a",
    input_border_color="#242833",
    input_border_color_focus="#8f7cff",
    button_primary_background_fill="#8f7cff",
    button_primary_text_color="#050507",
    button_secondary_background_fill="#181b22",
    button_secondary_text_color="#f5f7fb",
    button_secondary_border_color="#242833",
)
EXAMPLE_ACTIONS = [
    ("Overview", "Give me a concise overview of the indexed documents."),
    ("Summarize", "Summarize the most important points in the indexed documents."),
    ("Plan", "Create an action plan based on the indexed documents."),
]
DATA_DIR = Path(__file__).resolve().parent / "data"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');

:root {
    color-scheme: dark;
    --bg: #050507;
    --panel: #0f1013;
    --panel-soft: #14161a;
    --border: rgba(255, 255, 255, 0.08);
    --text: #f5f7fb;
    --muted: #9ea5b3;
    --accent: #8f7cff;
    --accent-soft: rgba(143, 124, 255, 0.14);
}

html, body {
    min-height: 100%;
}

body, .gradio-container {
    font-family: "Manrope", sans-serif !important;
    color: var(--text);
    background:
        radial-gradient(circle at 20% 10%, rgba(143, 124, 255, 0.12), transparent 22%),
        radial-gradient(circle at 80% 25%, rgba(143, 124, 255, 0.08), transparent 18%),
        linear-gradient(180deg, #060607 0%, #09090b 100%) !important;
}

.gradio-container,
.gradio-container * {
    color: inherit;
}

.gradio-container {
    max-width: none !important;
    margin: 0 !important;
    padding: 1rem !important;
}

.gradio-container .gr-block,
.gradio-container .block,
.gradio-container .gr-box,
.gradio-container .gr-group,
.gradio-container .form,
.gradio-container .wrap,
.gradio-container .contain {
    background: transparent !important;
    box-shadow: none !important;
}

.app-shell {
    width: min(1480px, 100%);
    margin: 0 auto;
}

.workspace-grid {
    display: grid !important;
    grid-template-columns: 320px minmax(0, 1fr);
    gap: 1.25rem;
    align-items: stretch;
}

.sidebar-col, .main-col {
    min-width: 0 !important;
    width: auto !important;
    flex: none !important;
}

.sidebar {
    position: sticky;
    top: 1rem;
    min-height: calc(100vh - 2rem);
    padding: 1.25rem;
    border-radius: 28px;
    background: rgba(12, 13, 16, 0.96);
    border: 1px solid var(--border);
}

.sidebar,
.sidebar > div,
.sidebar .gr-block,
.sidebar .gr-box,
.sidebar .gr-group {
    background: rgba(12, 13, 16, 0.96) !important;
}

.brand {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 1rem;
}

.brand-mark {
    width: 48px;
    height: 48px;
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(143, 124, 255, 0.4), rgba(143, 124, 255, 0.08));
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.brand h1 {
    margin: 0;
    font-size: 1.6rem;
}

.brand p, .sidebar-note, .status-card p, .status-card li {
    color: var(--muted);
    line-height: 1.65;
}

.brand h1,
.sidebar h1,
.sidebar h2,
.sidebar h3,
.sidebar strong,
.main-canvas h2 {
    color: var(--text) !important;
}

.sidebar-note {
    margin: 1rem 0 1.1rem 0;
    font-size: 0.95rem;
}

.sidebar-section {
    margin-top: 1rem;
}

.sidebar-section h3 {
    margin: 0 0 0.7rem 0;
    font-size: 0.95rem;
}

.status-card {
    margin-top: 1rem;
    padding: 1rem;
    border-radius: 20px;
    background: var(--panel-soft);
    border: 1px solid var(--border);
}

.status-card,
.status-card > div,
.status-card .gr-block,
.status-card .gr-box {
    background: var(--panel-soft) !important;
}

.status-card .prose,
.status-card .prose *,
.status-card .markdown,
.status-card .markdown * {
    color: var(--text) !important;
}

.status-card code {
    background: rgba(143, 124, 255, 0.12) !important;
    color: #d8d1ff !important;
}

.new-chat-btn button,
.quick-btn button {
    border-radius: 999px !important;
    background: #1a1d23 !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    box-shadow: none !important;
}

.new-chat-btn button {
    width: 100%;
    justify-content: center !important;
}

.main-canvas {
    min-height: calc(100vh - 2rem);
    padding: 1.35rem;
    border-radius: 32px;
    background: linear-gradient(180deg, rgba(14, 15, 18, 0.98), rgba(8, 9, 11, 0.98));
    border: 1px solid var(--border);
}

.main-canvas,
.main-canvas > div,
.main-canvas .gr-block,
.main-canvas .gr-box,
.main-canvas .gr-group {
    background: linear-gradient(180deg, rgba(14, 15, 18, 0.98), rgba(8, 9, 11, 0.98)) !important;
}

.hero {
    padding: 2.8rem 1rem 1.4rem 1rem;
    text-align: center;
}

.orb {
    width: 72px;
    height: 72px;
    margin: 0 auto 1rem auto;
    border-radius: 999px;
    background: radial-gradient(circle at 35% 30%, #fff, #c8beff 28%, #8f7cff 54%, #2f2850 78%, #14151a 100%);
    box-shadow: 0 0 36px rgba(143, 124, 255, 0.3);
}

.hero h2 {
    margin: 0;
    font-size: clamp(2rem, 4vw, 3.6rem);
    letter-spacing: -0.04em;
}

.hero p {
    max-width: 680px;
    margin: 0.85rem auto 0 auto;
    color: var(--muted);
    line-height: 1.7;
}

.header-pills {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin-bottom: 1.25rem;
}

.status-pill {
    padding: 0.55rem 0.9rem;
    border-radius: 999px;
    background: #171920;
    border: 1px solid var(--border);
    color: var(--text);
    font-size: 0.9rem;
}

.chat-card, .composer {
    border-radius: 24px;
    background: rgba(9, 10, 13, 0.94);
    border: 1px solid var(--border);
}

.chat-card {
    overflow: hidden;
}

.chat-card,
.chat-card > div,
.chat-card .gr-block,
.chat-card .gr-box,
.chat-card .wrap,
.chat-card .bubble-wrap,
.chat-card [data-testid="chatbot"] {
    background: rgba(9, 10, 13, 0.94) !important;
}

.chat-history [data-testid="chatbot"] {
    background: transparent !important;
}

.chat-history .message-wrap,
.chat-history .message,
.chat-history .bot,
.chat-history .user {
    color: var(--text) !important;
}

.chat-history .placeholder,
.chat-history [data-testid="chatbot"] .placeholder {
    color: var(--muted) !important;
}

.composer {
    margin-top: 1rem;
    padding: 1rem;
}

.composer,
.composer > div,
.composer .gr-block,
.composer .gr-box,
.composer .gr-group {
    background: rgba(9, 10, 13, 0.94) !important;
}

.prompt-row {
    align-items: end;
    gap: 0.85rem;
}

.composer-input textarea {
    min-height: 84px !important;
    background: #13151a !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    color: var(--text) !important;
    box-shadow: none !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

.composer-input,
.composer-input > div,
.composer-input .gr-block,
.composer-input .gr-box {
    background: transparent !important;
}

.composer-input textarea::placeholder {
    color: var(--muted) !important;
}

.send-btn button {
    height: 56px;
    min-width: 120px !important;
    border-radius: 18px !important;
    background: var(--accent) !important;
    border: none !important;
    color: #0a0a0c !important;
    font-weight: 800 !important;
    box-shadow: none !important;
}

footer {
    display: none !important;
}

@media (max-width: 980px) {
    .workspace-grid {
        grid-template-columns: 1fr;
    }

    .sidebar {
        position: static;
        min-height: auto;
    }
}
"""


def format_status() -> str:
    files = []
    if DATA_DIR.exists():
        for path in DATA_DIR.iterdir():
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)

    file_count = len(files)
    file_line = f"Files: **{file_count}**" if file_count else "Files: **0**"
    return "\n\n".join(
        [
            "### Status",
            "Ready.",
            file_line,
            "Supported files: `.txt`, `.md`, `.pdf`",
        ]
    )


def format_header_pills() -> str:
    files = 0
    if DATA_DIR.exists():
        files = sum(
            1
            for path in DATA_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    pills = [
        "Grounded answers",
        f"Files: {files}",
        "Local fallback",
    ]

    items = "".join(f"<span class='status-pill'>{pill}</span>" for pill in pills)
    return f"<div class='header-pills'>{items}</div>"


def answer_question(message: str) -> str:
    if not message.strip():
        return "Ask a question about the indexed documents."

    try:
        result = get_cached_pipeline().ask(message)
    except ConfigurationError as exc:
        return f"The Space is not configured yet.\n\nReason: {exc}"
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
    updated = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return updated, ""


def clear_chat() -> tuple[list[dict], str]:
    return [], ""


with gr.Blocks(title=TITLE, fill_width=True, fill_height=True) as demo:
    with gr.Column(elem_classes=["app-shell"]):
        with gr.Row(elem_classes=["workspace-grid"], equal_height=False):
            with gr.Column(scale=3, min_width=300, elem_classes=["sidebar-col"]):
                with gr.Group(elem_classes=["sidebar"]):
                    gr.HTML(
                        """
<div class="brand">
  <div class="brand-mark"></div>
  <div>
    <h1>RAG Chatbot</h1>
    <p>Minimal grounded chat over your repository documents.</p>
  </div>
</div>
"""
                    )
                    new_chat = gr.Button("+ New chat", elem_classes=["new-chat-btn"], variant="secondary")
                    gr.HTML("<p class='sidebar-note'>Use this chat when you want answers anchored in your `data/` folder instead of generic model memory.</p>")
                    gr.HTML("<div class='sidebar-section'><h3>Quick actions</h3></div>")
                    with gr.Row():
                        quick_buttons = []
                        for label, _ in EXAMPLE_ACTIONS:
                            quick_buttons.append(gr.Button(label, elem_classes=["quick-btn"], size="sm"))
                    status = gr.Markdown(elem_classes=["status-card"])

            with gr.Column(scale=8, min_width=700, elem_classes=["main-col"]):
                with gr.Group(elem_classes=["main-canvas"]):
                    gr.HTML(
                        f"""
<div class="hero">
  <div class="orb"></div>
  <h2>{TITLE}</h2>
  <p>{SUBTITLE}</p>
</div>
"""
                    )
                    header_pills = gr.HTML()
                    chatbot = gr.Chatbot(
                        [],
                        height=500,
                        layout="bubble",
                        buttons=["copy"],
                        placeholder="Start a conversation with your indexed documents.",
                        elem_classes=["chat-card", "chat-history"],
                    )
                    with gr.Group(elem_classes=["composer"]):
                        with gr.Row(elem_classes=["prompt-row"], equal_height=False):
                            message_box = gr.Textbox(
                                placeholder="Ask anything about your documents...",
                                lines=3,
                                max_lines=6,
                                container=False,
                                autofocus=True,
                                submit_btn=False,
                                stop_btn=False,
                                elem_classes=["composer-input"],
                            )
                            send_button = gr.Button("Send", elem_classes=["send-btn"], variant="secondary")

        for button, (_, prompt_text) in zip(quick_buttons, EXAMPLE_ACTIONS):
            button.click(lambda value=prompt_text: value, outputs=message_box)

        send_button.click(submit_chat, inputs=[message_box, chatbot], outputs=[chatbot, message_box])
        message_box.submit(submit_chat, inputs=[message_box, chatbot], outputs=[chatbot, message_box])
        new_chat.click(clear_chat, outputs=[chatbot, message_box])
        demo.load(fn=format_status, outputs=status)
        demo.load(fn=format_header_pills, outputs=header_pills)


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=THEME,
        css=CUSTOM_CSS,
    )
