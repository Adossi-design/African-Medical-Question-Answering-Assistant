"""Gradio web interface for the African Medical Question Answering Assistant.

Run `python app.py` from the project root and open http://127.0.0.1:7860. The interface only formats the input and
the output, while all of the NLP work happens in src/assistant.py.
"""
import os

import gradio as gr

from src.assistant import MedicalAssistant

MODEL_DIR = os.environ.get("MODEL_DIR", "models/final")
assistant = MedicalAssistant(MODEL_DIR)

TITLE = "African Medical Question Answering Assistant"
INTRO = (
    "Ask a medical question in everyday English and receive educational information from the assistant's "
    f"knowledge base of {len(assistant.kb):,} human-written answers from the AfriMed-QA v2.5 dataset, which were "
    "written by contributors from Nigeria, Kenya and Malawi, including medical experts. A fine-tuned Transformer "
    "model finds the answer whose meaning is closest to your question, and if no answer is close enough, "
    "the assistant says so instead of guessing."
)
DISCLAIMER = (
    "**Disclaimer:** This system provides educational information and does not replace professional medical "
    "advice, diagnosis, or treatment. It cannot diagnose you or prescribe treatment. Its knowledge is limited to "
    "the topics in its dataset. In an emergency, contact a health facility immediately."
)
TRADITIONAL_NOTE = (
    "> **Caution:** This answer describes a traditional practice reported by dataset contributors. "
    "Its safety and effectiveness may not be scientifically established."
)
EXAMPLES = [
    "How is uncomplicated malaria treated in Africa?",
    "How does cholera spread?",
    "How do people usually catch Ebola in Africa?",
    "Is vertigo a brain problem?",          # A medical topic that the dataset does not cover.
    "How do I repair my laptop?",           # A question outside the medical domain.
]


def respond(question: str) -> str:
    result = assistant.ask(question)
    if result["decision"] != "answer":
        score = f"\n\n*Best similarity found: {result['score']:.2f}*" if "score" in result else ""
        return f"**{result['message']}**{score}"
    answer = result["answer"].replace("\n", "  \n")  # Keep the line breaks of list-style answers.
    parts = [f"### Answer\n{answer}"]
    if result["traditional_remedy"]:
        parts.append(TRADITIONAL_NOTE)
    parts.append(f"---\n**Closest question in the knowledge base:** {result['matched_question']}  \n"
                 f"**Similarity (confidence):** {result['score']:.2f}  \n**Source:** {result['source']}")
    return "\n\n".join(parts)


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{INTRO}")
    question = gr.Textbox(label="Your medical question", lines=2,
                          placeholder="e.g. How does cholera usually spread between people in Africa?")
    ask_button = gr.Button("Ask", variant="primary")
    output = gr.Markdown()
    gr.Examples(examples=EXAMPLES, inputs=question)
    gr.Markdown(DISCLAIMER)
    ask_button.click(respond, inputs=question, outputs=output)
    question.submit(respond, inputs=question, outputs=output)

if __name__ == "__main__":
    demo.launch()
