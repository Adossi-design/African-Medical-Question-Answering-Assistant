"""This module provides the Streamlit chat interface, while all language processing takes place in src/assistant.py."""
import os
from pathlib import Path

import streamlit as st
from huggingface_hub import snapshot_download

from src.assistant import MedicalAssistant

# Notebook 04 exports the model to this folder, and the hosted app downloads the same files from Hugging Face instead.
MODEL_DIR = os.environ.get("MODEL_DIR", "models/final")
MODEL_REPO = "Fred-William/african-medical-qa-assistant"

# Setting this to True during development shows the retrieved question, the similarity score and the source.
DEBUG_MODE = False

TITLE = "AfriMed QA Assistant"
SUBTITLE = "Your medical information companion for Africa"
ABOUT = "AfriMed QA Assistant provides educational medical information using knowledge from the AfriMed-QA v2.5 dataset."
DISCLAIMER = "For educational purposes only. It does not replace professional medical advice, diagnosis, or treatment."
INPUT_NOTE = ("Educational information only. Important health decisions should be discussed with a qualified "
              "healthcare professional.")
SHORT_INPUT_NOTE = "Educational information only. Check health decisions with a professional."
FOLLOW_UP = " Please consult a qualified healthcare professional for further guidance."
TRADITIONAL_NOTE = ("This answer describes a traditional practice reported by dataset contributors. "
                    "Its safety and effectiveness may not be scientifically established.")
FILES_NOT_READ = "I can only read typed questions, so attached files are not used. Please type your medical question."
EXAMPLES = [
    "How is uncomplicated malaria treated in Africa?",
    "How does cholera spread?",
    "How do people usually catch Ebola in Africa?",
    "Is vertigo a brain problem?",
]

BLUE, GREEN, DARK_GREEN, ORANGE, INK = "#1f73d8", "#2e8b6a", "#1d5b48", "#d8891f", "#1f2937"


def svg(body: str, viewbox: str = "0 0 24 24") -> str:
    """Return a CSS image made from the inner markup of a small SVG icon."""
    body = body.replace("#", "%23")
    return f"url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='{viewbox}'>{body}</svg>\")"


def filled(path: str, colour: str) -> str:
    """Return a CSS image of a filled icon drawn from one path in the given colour."""
    return svg(f"<path fill='{colour}' d='{path}'/>")


def outlined(paths: str, colour: str) -> str:
    """Return a CSS image of an outlined icon drawn from one or more path elements in the given colour."""
    return svg(f"<g fill='none' stroke='{colour}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
               f"{paths}</g>")


PERSON = filled("M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66"
                "-5.33-4-8-4z", "#ffffff")
ADD_CIRCLE = filled("M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm5 11h-4v4h-2v-4H7v-2h4V7h2v4h4"
                    "v2z", "#2f3542")
INFO = filled("M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z",
              "#3b4452")
SEND = filled("M21 3 3 10.53v.98l6.84 2.65L12.48 21h.98L21 3z", "#ffffff")
MENU = filled("M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z", INK)
CLOSE = filled("M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z",
               INK)
BUBBLE = outlined("<path d='M7.9 20A9 9 0 1 0 4 16.1L2 22Z'/>", "#374151")
SHIELD = outlined("<path d='M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5"
                  "-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z'/>", ORANGE)
PAPERCLIP = outlined("<path d='m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2"
                     " 0 0 1-2.83-2.83l8.49-8.48'/>", "#6b7280")

# The style sheet reproduces the approved design for desktop, tablet and phone on top of the Streamlit components.
STYLE = f"""
<style>
/* Shared icon styles let small SVG images and Material Symbols sit inline with text. */
.icon {{ font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1;
         display: inline-block; vertical-align: middle; }}
.svg-icon {{ display: inline-block; flex: 0 0 auto; background: center / contain no-repeat; }}
.icon-info {{ background-image: {INFO}; }}
.icon-shield {{ background-image: {SHIELD}; }}

/* The conversation column stays centred with a comfortable reading width. */
[data-testid="stMainBlockContainer"], [data-testid="stBottomBlockContainer"] {{ max-width: 62rem; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* Chat rows place an avatar beside a rounded bubble, on the right for the user and on the left for the assistant. */
[data-testid="stChatMessage"] {{ background: transparent; padding: 0; margin: 0.8rem 0; gap: 0.85rem;
                                 align-items: flex-start; }}
[data-testid="stChatMessageContent"] {{ flex: 0 1 auto; width: fit-content; max-width: 70%; margin: 0 !important;
                                         border-radius: 1rem; padding: 0.7rem 1.15rem; color: {INK}; }}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessageContent"] p:last-child {{ margin-bottom: 0; }}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {{ flex-direction: row-reverse; }}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] {{
    background: #e9effb; }}
[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) [data-testid="stChatMessageContent"] {{
    background: #f1f2f4; }}
[data-testid="stChatMessageAvatarCustom"] {{ width: 2.45rem; height: 2.45rem; border-radius: 50%; border: none;
                                             display: flex; align-items: center; justify-content: center; }}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageAvatarCustom"] {{
    background: {BLUE}; }}
[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) [data-testid="stChatMessageAvatarCustom"] {{
    background: {DARK_GREEN}; }}
[data-testid="stChatMessageAvatarCustom"] [data-testid="stIconMaterial"] {{ color: #ffffff !important; font-size: 1.35rem; }}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stIconMaterial"] {{
    color: transparent !important; width: 1.5rem; height: 1.5rem; background: {PERSON} center / contain no-repeat; }}

/* The message composer is a rounded bar with a paperclip on the left and a round send button on the right. */
[data-testid="stChatInput"] {{ border: none; background: transparent; }}
[data-testid="stChatInput"] > div {{ border: 1px solid {BLUE} !important; border-radius: 2rem !important;
                                     background: #ffffff; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05); }}
[data-testid="stChatInputTextArea"] {{ background: #ffffff; }}
[data-testid="stBottomBlockContainer"] {{ padding-bottom: 1.9rem; }}

/* Streamlit Community Cloud adds Fork and GitHub buttons to the header, which the design does not include. */
[data-testid="stToolbarActions"] {{ display: none; }}
[data-testid="stChatInputSubmitButton"] {{ width: 2.75rem; height: 2.75rem; border-radius: 50%;
    background: {BLUE} {SEND} center / 1.2rem no-repeat !important; }}
[data-testid="stChatInputSubmitButton"] svg {{ visibility: hidden; }}
[data-testid="stChatInputFileUploadButton"] {{ background: {PAPERCLIP} center / 1.2rem no-repeat !important; }}
[data-testid="stChatInputFileUploadButton"] svg, [data-testid="stChatInputFileUploadButton"] span {{ visibility: hidden; }}
[data-testid="stBottomBlockContainer"]::after {{ content: "{INPUT_NOTE}"; display: block; text-align: center;
    font-size: 0.75rem; color: #6b7280; padding-top: 0.55rem; }}

/* The sidebar holds the brand, the new chat card, the example cards and the two information boxes. */
[data-testid="stSidebarUserContent"] {{ padding-top: 1.4rem; }}
[data-testid="stSidebarUserContent"] [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {{ gap: 0.55rem; min-height: calc(100vh - 3rem); }}
.brand {{ display: flex; gap: 0.6rem; align-items: flex-start; margin-bottom: 0.6rem; }}
.brand .icon {{ color: {GREEN}; font-size: 2rem; margin-top: 0.05rem; }}
.brand-title {{ font-size: 1.3rem; font-weight: 700; color: {INK}; line-height: 1.25; }}
.brand-subtitle {{ font-size: 0.85rem; color: #6b7280; line-height: 1.35; margin-top: 0.1rem; }}
[data-testid="stSidebar"] button[kind="secondary"] {{ background: #ffffff; border: 1px solid #e3e6ea; color: {INK};
    border-radius: 0.6rem; justify-content: flex-start; text-align: left; padding: 0.6rem 0.9rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05); }}
[data-testid="stSidebar"] button[kind="secondary"] > div {{ justify-content: flex-start !important; gap: 0.7rem; }}
[data-testid="stSidebar"] button[kind="secondary"] p {{ text-align: left; font-size: 0.87rem; }}
.st-key-new_chat button {{ padding: 0.7rem 0.9rem !important; }}
.st-key-new_chat button p {{ font-weight: 700; font-size: 0.95rem !important; }}
.st-key-new_chat [data-testid="stIconMaterial"] {{ color: transparent !important; width: 1.3rem; height: 1.3rem;
    background: {ADD_CIRCLE} center / contain no-repeat; }}
[class*="st-key-example_"] [data-testid="stIconMaterial"] {{ color: transparent !important; width: 1rem; height: 1rem;
    background: {BUBBLE} center / contain no-repeat; }}
.section-label {{ font-size: 0.85rem; color: #6b7280; margin: 0.5rem 0 0.1rem; }}
[data-testid="stLayoutWrapper"]:has(> .st-key-sidebar_footer) {{ margin-top: auto; }}
.info-box {{ border-radius: 0.6rem; padding: 0.8rem 0.9rem; display: flex; gap: 0.65rem; margin-top: 0.25rem; }}
.info-box.about {{ background: #edf2fb; border: 1px solid #dbe4f3; }}
.info-box.disclaimer {{ background: #fff6e8; border: 1px solid #f5e0bf; }}
.info-box .svg-icon {{ width: 1.1rem; height: 1.1rem; margin-top: 0.1rem; }}
.info-title {{ font-weight: 700; font-size: 0.9rem; color: {INK}; }}
.info-text {{ font-size: 0.82rem; color: #4b5563; line-height: 1.45; margin-top: 0.15rem; }}
.topbar {{ display: none; }}

/* On wide screens the sidebar is permanent, so its collapse control is hidden. */
@media (min-width: 768px) {{
    [data-testid="stSidebarHeader"] {{ display: none; }}
    [data-testid="stSidebar"] {{ width: 21rem !important; min-width: 21rem !important; }}
    [data-testid="stMainBlockContainer"] {{ padding-top: 2rem; }}
}}

/* On narrow screens a top bar replaces the sidebar, and the sidebar opens as a drawer over the chat. */
@media (max-width: 767px) {{
    .topbar {{ display: flex; position: fixed; top: 0; left: 0; right: 0; height: 3.75rem; z-index: 999989;
               align-items: center; justify-content: center; gap: 0.5rem; background: #f5f6f8;
               border-bottom: 1px solid #e5e7eb; font-size: 1.15rem; font-weight: 700; color: {INK}; }}
    .topbar .icon {{ color: {GREEN}; font-size: 1.7rem; }}
    .topbar-new {{ position: absolute; right: 1.1rem; width: 1.6rem; height: 1.6rem;
                   background: {ADD_CIRCLE} center / contain no-repeat; }}
    [data-testid="stHeader"] {{ pointer-events: none; z-index: 999990; }}
    [data-testid="stExpandSidebarButton"] {{ pointer-events: auto; width: 1.8rem; height: 1.8rem;
        background: {MENU} center / contain no-repeat !important; }}
    [data-testid="stExpandSidebarButton"] * {{ visibility: hidden; }}
    [data-testid="stSidebar"][aria-expanded="true"] {{ box-shadow: 0 0 0 100vmax rgba(17, 24, 39, 0.45); }}
    [data-testid="stSidebarHeader"] {{ position: absolute; top: 0.9rem; right: 0.8rem; z-index: 2; padding: 0; }}
    [data-testid="stSidebarCollapseButton"] {{ visibility: visible !important; width: 1.8rem; height: 1.8rem;
        background: {CLOSE} center / contain no-repeat !important; }}
    [data-testid="stSidebarCollapseButton"] * {{ visibility: hidden; }}
    [data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {{ min-height: 0; }}
    .brand {{ padding-right: 2.5rem; }}
    .section-label, [class*="st-key-example_"] {{ display: none !important; }}
    [data-testid="stLayoutWrapper"]:has(> .st-key-sidebar_footer) {{ margin-top: 0.4rem; }}
    [data-testid="stMainBlockContainer"] {{ padding-top: 5rem; }}
}}

/* On phones the user avatar is hidden and the note under the composer is shortened. */
@media (max-width: 600px) {{
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageAvatarCustom"] {{
        display: none; }}
    [data-testid="stChatMessageContent"] {{ max-width: 85%; }}
    [data-testid="stBottomBlockContainer"]::after {{ content: "{SHORT_INPUT_NOTE}"; }}
    [data-testid="stBottomBlockContainer"] {{ padding-bottom: 3.4rem; }}
}}
</style>
"""


@st.cache_resource(show_spinner="Loading the model...")
def load_assistant() -> MedicalAssistant:
    """Load the assistant once from the local model folder, or from Hugging Face when that folder is missing."""
    folder = MODEL_DIR if Path(MODEL_DIR, "assistant_config.json").exists() else snapshot_download(MODEL_REPO)
    return MedicalAssistant(folder)


def respond(assistant: MedicalAssistant, question: str) -> dict:
    """Return the chat message for one question, keeping the retrieval details only for debug mode."""
    result = assistant.ask(question)
    thresholds = f"answer threshold {assistant.config['answer_threshold']:.2f}"
    if result["decision"] == "answer":
        return {"role": "assistant",
                "content": result["answer"].replace("\n", "  \n"),  # Keep the line breaks of list-style answers.
                "notes": [TRADITIONAL_NOTE] if result["traditional_remedy"] else [],
                "debug": (f"Closest question in the knowledge base: {result['matched_question']}  \n"
                          f"Similarity: {result['score']:.2f} ({thresholds}) · Source: {result['source']}")}
    content = result["message"] + (FOLLOW_UP if result["decision"] == "not_enough_information" else "")
    debug = f"Best similarity found: {result['score']:.2f} ({thresholds})" if "score" in result else None
    return {"role": "assistant", "content": content, "notes": [], "debug": debug}


def show(message: dict):
    """Draw one chat message, adding the retrieval details only when debug mode is switched on."""
    avatar = ":material/person:" if message["role"] == "user" else ":material/stethoscope:"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        for note in message.get("notes", []):
            st.caption(f":material/info: {note}")
        if DEBUG_MODE and message.get("debug"):
            st.caption(message["debug"])


def ask_example(question: str):
    """Queue an example question so that it is answered like a typed question."""
    st.session_state.pending = question


def new_chat():
    """Clear the conversation so that a new chat starts."""
    st.session_state.messages = []
    st.session_state.pop("pending", None)


st.set_page_config(page_title=TITLE, page_icon=":material/stethoscope:", layout="centered")
st.markdown(STYLE, unsafe_allow_html=True)
assistant = load_assistant()
if "messages" not in st.session_state:
    st.session_state.messages = []

# The top bar only appears on narrow screens, where reloading the page starts a new chat.
st.markdown(f"<div class='topbar'><span class='icon'>stethoscope</span>{TITLE}"
            "<a class='topbar-new' href='./' target='_self' title='New chat'></a></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"<div class='brand'><span class='icon'>stethoscope</span><div><div class='brand-title'>{TITLE}</div>"
                f"<div class='brand-subtitle'>{SUBTITLE}</div></div></div>", unsafe_allow_html=True)
    st.button("New chat", icon=":material/add_circle:", on_click=new_chat, width="stretch", key="new_chat")
    st.markdown("<div class='section-label'>Example questions</div>", unsafe_allow_html=True)
    for i, example in enumerate(EXAMPLES):
        st.button(example, icon=":material/chat_bubble:", on_click=ask_example, args=(example,), width="stretch",
                  key=f"example_{i}")
    with st.container(key="sidebar_footer"):
        st.markdown(f"<div class='info-box about'><span class='svg-icon icon-info'></span>"
                    f"<div><div class='info-title'>About</div><div class='info-text'>{ABOUT}</div></div></div>"
                    f"<div class='info-box disclaimer'><span class='svg-icon icon-shield'>"
                    f"</span><div><div class='info-title'>Disclaimer</div><div class='info-text'>{DISCLAIMER}</div>"
                    "</div></div>", unsafe_allow_html=True)

submitted = st.chat_input("Ask a medical question", accept_file=True)
question, files = st.session_state.pop("pending", None), []
if submitted is not None:
    question, files = (submitted.text or "").strip(), list(submitted.files or [])

if not st.session_state.messages and not (question or files):
    st.markdown("<h2 style='text-align: center; margin-top: 30vh; font-weight: 700;'>How can I help you today?</h2>",
                unsafe_allow_html=True)
for message in st.session_state.messages:
    show(message)
if question or files:
    shown = question or "Attached file: " + ", ".join(file.name for file in files)
    user_message = {"role": "user", "content": shown}
    st.session_state.messages.append(user_message)
    show(user_message)
    if question:
        answer = respond(assistant, question)
        if files:
            answer["notes"].append("Only your typed question was used, because attached files are not read.")
    else:
        answer = {"role": "assistant", "content": FILES_NOT_READ, "notes": [], "debug": None}
    st.session_state.messages.append(answer)
    show(answer)
