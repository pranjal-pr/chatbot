from __future__ import annotations

import os
import re
from html import escape
from pathlib import Path

import gradio as gr

from rag_backend import (
    ConfigurationError,
    enable_local_fallback_from_exception,
    get_cached_chat_runtime,
    get_cached_pipeline,
)

TITLE = "RAG Chatbot"
SUBTITLE = "Switch between direct chat and grounded retrieval."
MODE_OPTIONS = ["Auto", "Chat only", "RAG only"]
THEME = gr.themes.Base(primary_hue="violet", secondary_hue="slate", neutral_hue="slate").set(
    body_background_fill="#050507",
    body_text_color="#f5f7fb",
    body_text_color_subdued="#9ea5b3",
    background_fill_primary="#101218",
    background_fill_secondary="#14161a",
    block_background_fill="#101218",
    block_border_color="#232833",
    input_background_fill="#101218",
    input_border_color="#232833",
    input_border_color_focus="#8f7cff",
    button_primary_background_fill="#8f7cff",
    button_primary_text_color="#0b0c10",
    button_secondary_background_fill="#181b22",
    button_secondary_text_color="#f5f7fb",
    button_secondary_border_color="#232833",
)
DATA_DIR = Path(__file__).resolve().parent / "data"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');

:root {
    color-scheme: dark;
    --bg: #050507;
    --panel: #0e1015;
    --panel-soft: #171a20;
    --border: rgba(255, 255, 255, 0.08);
    --text: #f5f7fb;
    --muted: #9ea5b3;
    --accent: #8f7cff;
}

html, body {
    min-height: 100%;
}

body, .gradio-container {
    font-family: "Manrope", sans-serif !important;
    color: var(--text);
    background:
        radial-gradient(circle at 25% 8%, rgba(143, 124, 255, 0.10), transparent 18%),
        radial-gradient(circle at 80% 22%, rgba(143, 124, 255, 0.08), transparent 16%),
        linear-gradient(180deg, #050507 0%, #08090d 100%) !important;
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
.gradio-container .wrap {
    background: transparent !important;
    box-shadow: none !important;
}

.app-shell {
    width: min(1500px, 100%);
    margin: 0 auto;
}

.workspace-grid {
    display: grid !important;
    grid-template-columns: 310px minmax(0, 1fr);
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
    padding: 1.1rem;
    border-radius: 28px;
    background: rgba(13, 14, 18, 0.98) !important;
    border: 1px solid var(--border);
}

.brand {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 1rem;
}

.brand-mark {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: radial-gradient(circle at 35% 30%, #fff, #c8beff 32%, #8f7cff 58%, #2f2850 84%, #14151a 100%);
    box-shadow: 0 0 24px rgba(143, 124, 255, 0.28);
}

.brand h1 {
    margin: 0;
    font-size: 1.55rem;
}

.brand p, .sidebar-meta, .sidebar-meta * {
    color: var(--muted) !important;
    line-height: 1.65;
}

.new-chat-btn button {
    width: 100%;
    border-radius: 16px !important;
    background: #181b22 !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    box-shadow: none !important;
}

.mode-switch {
    margin-top: 1rem;
    padding: 0.95rem 1rem !important;
    border-radius: 20px;
    background: var(--panel-soft) !important;
    border: 1px solid var(--border) !important;
}

.mode-switch label,
.mode-switch span,
.mode-switch p {
    color: var(--text) !important;
}

.mode-switch [data-testid="block-label"] {
    margin-bottom: 0.5rem !important;
    font-size: 0.95rem !important;
}

.mode-switch .wrap {
    gap: 0.55rem !important;
}

.mode-switch .wrap label {
    border-radius: 999px !important;
    background: #12151b !important;
    border: 1px solid var(--border) !important;
    padding: 0.55rem 0.9rem !important;
}

.mode-switch .wrap label:has(input:checked) {
    background: rgba(143, 124, 255, 0.16) !important;
    border-color: rgba(143, 124, 255, 0.55) !important;
}

.sidebar-meta {
    margin-top: 1rem;
    padding: 0.95rem 1rem;
    border-radius: 20px;
    background: var(--panel-soft) !important;
    border: 1px solid var(--border);
}

.sidebar-meta-line {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.sidebar-meta-line:last-child {
    border-bottom: none;
}

.sidebar-meta strong {
    color: var(--text) !important;
}

.main-canvas {
    min-height: calc(100vh - 2rem);
    padding: 1.2rem;
    border-radius: 32px;
    background: linear-gradient(180deg, rgba(14, 15, 18, 0.98), rgba(8, 9, 11, 0.98)) !important;
    border: 1px solid var(--border);
    display: flex;
    flex-direction: column;
}

.hero-card {
    width: min(820px, 100%);
    margin: 0 auto;
    padding: 2rem 1rem 1.4rem 1rem;
    border-radius: 28px;
    background: rgba(18, 20, 26, 0.96) !important;
    border: 1px solid var(--border);
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

.hero-card h2 {
    margin: 0;
    font-size: clamp(2.2rem, 4vw, 3.7rem);
    letter-spacing: -0.04em;
}

.hero-card p {
    margin: 0.75rem auto 0 auto;
    max-width: 620px;
    color: var(--muted);
    line-height: 1.6;
}

.header-pills {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin: 1rem 0 0 0;
}

.status-pill {
    padding: 0.55rem 0.9rem;
    border-radius: 999px;
    background: #161920;
    border: 1px solid var(--border);
    color: var(--text);
    font-size: 0.9rem;
}

.chat-shell {
    margin-top: 1rem;
    border-radius: 24px;
    border: 1px solid var(--border);
    background: rgba(9, 10, 13, 0.96) !important;
    overflow: hidden;
}

.chat-shell,
.chat-shell > div,
.chat-shell .gr-block,
.chat-shell .gr-box,
.chat-shell [data-testid="chatbot"] {
    background: rgba(9, 10, 13, 0.96) !important;
}

.chat-history [data-testid="chatbot"] .placeholder {
    color: var(--muted) !important;
}

.composer {
    margin-top: auto;
    padding-top: 1.2rem;
}

.input-bar {
    padding: 0.7rem;
    border-radius: 22px;
    background: rgba(11, 12, 16, 0.98) !important;
    border: 1px solid var(--border);
}

.prompt-row {
    align-items: center;
    gap: 0.8rem;
}

.composer-input textarea {
    min-height: 24px !important;
    max-height: 24px !important;
    padding-top: 0.95rem !important;
    padding-bottom: 0.95rem !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text) !important;
    font-size: 1rem !important;
}

.composer-input textarea::placeholder {
    color: var(--muted) !important;
}

.send-btn button {
    width: 56px;
    min-width: 56px !important;
    height: 56px;
    border-radius: 999px !important;
    background: #1c2028 !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-weight: 700 !important;
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


def _indexed_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return [
        path
        for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def format_status() -> str:
    files = _indexed_files()
    return "\n\n".join(
        [
            "### Status",
            "Ready.",
            f"Files: **{len(files)}**",
            "Supported files: `.txt`, `.md`, `.pdf`",
        ]
    )


def normalize_mode(mode: str | None) -> str:
    return mode if mode in MODE_OPTIONS else "Auto"


def mode_summary(mode: str) -> str:
    labels = {
        "Auto": "Adaptive",
        "Chat only": "Direct",
        "RAG only": "Grounded",
    }
    return labels[normalize_mode(mode)]


def format_sidebar_meta(mode: str) -> str:
    files = _indexed_files()
    rows = [
        ("Mode", mode_summary(mode)),
        ("Files", str(len(files))),
        ("Types", ".txt .md .pdf"),
    ]
    lines = "".join(
        f"<div class='sidebar-meta-line'><span>{escape(label)}</span><strong>{escape(value)}</strong></div>"
        for label, value in rows
    )
    return f"<div class='sidebar-meta'>{lines}</div>"


def format_header_pills(mode: str) -> str:
    files = len(_indexed_files())
    mode = normalize_mode(mode)
    if mode == "Chat only":
        pills = ["Chat only", "Direct answers"]
    elif mode == "RAG only":
        pills = ["RAG only", f"Files: {files}"]
    else:
        pills = ["Auto mode", f"Files: {files}", "Chat + RAG"]
    items = "".join(f"<span class='status-pill'>{escape(pill)}</span>" for pill in pills)
    return f"<div class='header-pills'>{items}</div>"


def refresh_mode_ui(mode: str) -> tuple[str, str]:
    mode = normalize_mode(mode)
    return format_sidebar_meta(mode), format_header_pills(mode)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def should_use_rag_auto(message: str) -> bool:
    files = _indexed_files()
    if not files:
        return False

    text = message.lower()
    keywords = (
        "document",
        "documents",
        "doc",
        "docs",
        "repository",
        "repo",
        "file",
        "files",
        "data",
        "dataset",
        "source",
        "sources",
        "indexed",
        "knowledge",
        "context",
        "grounded",
        "readme",
        ".pdf",
        ".md",
        ".txt",
        "supported",
        "in the file",
        "based on",
    )
    if any(keyword in text for keyword in keywords):
        return True

    return any(path.name.lower() in text or path.stem.lower() in text for path in files)


def should_bypass_local_chat_model(message: str) -> bool:
    text = normalize_text(message)
    patterns = (
        r"^(hi|hello|hey|yo|good morning|good afternoon|good evening)\b",
        r"^(i am|i'm|my name is)\b",
        r"\b(student|study|studying|exam|homework|assignment)\b",
        r"\b(what can you do|who are you|tell me about (yourself|this chatbot|this app))\b",
        r"\b(can you help me|help me)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def is_low_quality_local_chat_answer(message: str, answer: str) -> bool:
    normalized_message = normalize_text(message)
    normalized_answer = normalize_text(answer)

    if not normalized_answer:
        return True
    if normalized_answer == normalized_message:
        return True
    if normalized_answer in {"yes", "no", "maybe", "...", ".", ".."}:
        return True
    if "answer the user's question" in normalized_answer:
        return True
    if len(normalized_answer.split()) <= 3 and not any(char.isdigit() for char in normalized_answer):
        return True
    if normalized_answer.endswith("?") and not normalized_message.endswith("?"):
        return True
    return False


def build_local_chat_fallback(message: str) -> str:
    text = normalize_text(message)

    if re.search(r"^(hi|hello|hey|yo|good morning|good afternoon|good evening)\b", text):
        return "Hi. What do you want help with?"

    if re.search(r"^(i am|i'm|my name is)\b", text):
        if "student" in text:
            return "Got it. What do you want help with as a student? I can explain concepts, summarize material, or help you study."
        return "Got it. What would you like help with?"

    if re.search(r"\b(student|study|studying|exam|homework|assignment)\b", text):
        return "Yes. I can help you study, explain concepts, summarize notes, and break down problems step by step. What subject are you working on?"

    if re.search(r"\b(what can you do|who are you|tell me about (yourself|this chatbot|this app))\b", text):
        return (
            "I can work in three modes: `Auto`, `Chat only`, and `RAG only`. "
            "`Chat only` gives direct answers, `RAG only` answers from indexed documents with sources, "
            "and `Auto` switches between them based on your prompt."
        )

    if re.search(r"\b(can you help me|help me)\b", text):
        return "Yes. Tell me the task, question, or topic, and I’ll help directly or use the indexed documents when needed."

    return "Tell me what you want to work on, and I’ll answer directly or use the indexed documents when grounding is needed."


def answer_with_rag(message: str) -> str:
    if not _indexed_files():
        return "No indexed documents are available for RAG yet."

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


def answer_with_chat(message: str) -> str:
    runtime = get_cached_chat_runtime()

    if runtime.provider == "local" and should_bypass_local_chat_model(message):
        return build_local_chat_fallback(message)

    try:
        answer = runtime.ask(message)
    except ConfigurationError as exc:
        return f"The chat runtime is not configured yet.\n\nReason: {exc}"
    except Exception as exc:
        if enable_local_fallback_from_exception(exc):
            try:
                runtime = get_cached_chat_runtime()
                if runtime.provider == "local" and should_bypass_local_chat_model(message):
                    return build_local_chat_fallback(message)
                answer = runtime.ask(message)
            except Exception as retry_exc:
                return f"Unexpected runtime error after local fallback: {retry_exc}"
        else:
            return f"Unexpected runtime error: {exc}"

    if runtime.provider == "local" and is_low_quality_local_chat_answer(message, answer):
        return build_local_chat_fallback(message)
    return answer


def answer_question(message: str, mode: str) -> str:
    if not message.strip():
        return "Ask something to start the conversation."

    mode = normalize_mode(mode)
    if mode == "Chat only":
        return answer_with_chat(message)
    if mode == "RAG only":
        return answer_with_rag(message)
    if should_use_rag_auto(message):
        return answer_with_rag(message)
    return answer_with_chat(message)


def submit_chat(message: str, history: list[dict] | None, mode: str) -> tuple[list[dict], str, gr.update]:
    history = history or []
    if not message.strip():
        return history, "", gr.update(visible=bool(history))

    answer = answer_question(message, mode)
    updated = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return updated, "", gr.update(visible=True)


def clear_chat() -> tuple[list[dict], str, gr.update]:
    return [], "", gr.update(visible=False)


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
    <p>Direct chat or grounded retrieval.</p>
  </div>
</div>
"""
                    )
                    new_chat = gr.Button("+ New chat", elem_classes=["new-chat-btn"], variant="secondary")
                    mode_selector = gr.Radio(
                        MODE_OPTIONS,
                        value="Auto",
                        label="Mode",
                        container=True,
                        elem_classes=["mode-switch"],
                    )
                    sidebar_meta = gr.HTML()

            with gr.Column(scale=8, min_width=700, elem_classes=["main-col"]):
                with gr.Group(elem_classes=["main-canvas"]):
                    gr.HTML(
                        f"""
<div class="hero-card">
  <div class="orb"></div>
  <h2>{TITLE}</h2>
  <p>{SUBTITLE}</p>
</div>
"""
                    )
                    header_pills = gr.HTML()
                    with gr.Group(visible=False, elem_classes=["chat-shell"]) as chat_shell:
                        chatbot = gr.Chatbot(
                            [],
                            height=430,
                            layout="bubble",
                            buttons=["copy"],
                            placeholder="Start a conversation with your indexed documents.",
                            elem_classes=["chat-history"],
                        )
                    with gr.Group(elem_classes=["composer"]):
                        with gr.Group(elem_classes=["input-bar"]):
                            with gr.Row(elem_classes=["prompt-row"], equal_height=False):
                                message_box = gr.Textbox(
                                    placeholder="Ask anything...",
                                    lines=1,
                                    max_lines=1,
                                    container=False,
                                    autofocus=True,
                                    submit_btn=False,
                                    stop_btn=False,
                                    elem_classes=["composer-input"],
                                )
                                send_button = gr.Button(">", elem_classes=["send-btn"], variant="secondary")

        send_button.click(
            submit_chat,
            inputs=[message_box, chatbot, mode_selector],
            outputs=[chatbot, message_box, chat_shell],
        )
        message_box.submit(
            submit_chat,
            inputs=[message_box, chatbot, mode_selector],
            outputs=[chatbot, message_box, chat_shell],
        )
        new_chat.click(clear_chat, outputs=[chatbot, message_box, chat_shell])
        demo.load(lambda: refresh_mode_ui("Auto"), outputs=[sidebar_meta, header_pills])
        mode_selector.change(refresh_mode_ui, inputs=mode_selector, outputs=[sidebar_meta, header_pills])


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=THEME,
        css=CUSTOM_CSS,
    )
