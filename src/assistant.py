"""The question answering assistant used by the web app.

A user's question is cleaned, turned into an embedding and compared with every answer in the knowledge base. The
best score then goes through the confidence check in `decide`, which notebook 04 also uses, and the assistant
returns either the human-written answer or a refusal message. Everything the assistant needs is stored in one
folder, `models/final` by default: the fine-tuned model files, `assistant_config.json` with the thresholds, and
`knowledge_base.csv` with the cleaned AfriMed-QA questions and answers.
"""
import json
import re
from pathlib import Path

import pandas as pd

from src.retrieval import EmbeddingRetriever
from src.text_cleaning import clean_question

OUTSIDE_SCOPE = ("This question is outside the scope of this medical assistant. "
                 "I can only answer health and medical questions.")
NOT_ENOUGH_INFO = "I do not have enough reliable information about this topic in my medical knowledge base."
TRADITIONAL_REMEDY = re.compile(r"traditional|herb|indigenous|remed|healer|local practice", re.IGNORECASE)


def decide(score: float, answer_threshold: float, scope_threshold: float) -> str:
    """Apply the two confidence thresholds to the similarity of the best answer."""
    if score >= answer_threshold:
        return "answer"
    return "not_enough_information" if score >= scope_threshold else "outside_scope"


class MedicalAssistant:
    """Loads the final model, thresholds and knowledge base, and answers one question at a time."""

    def __init__(self, folder: str | Path):
        folder = Path(folder)
        with open(folder / "assistant_config.json", encoding="utf-8") as f:
            self.config = json.load(f)
        self.kb = pd.read_csv(folder / "knowledge_base.csv")
        kb_questions = self.kb.question.tolist() if self.config["index"] == "both" else None
        self.retriever = EmbeddingRetriever(str(folder), self.kb.answer.tolist(), kb_questions=kb_questions)

    def ask(self, question: str) -> dict:
        question = clean_question(question or "")
        if not question:
            return {"decision": "empty", "message": "Please type a medical question."}
        scores = self.retriever.scores([question])[0]
        best = int(scores.argmax())
        score = float(scores[best])
        decision = decide(score, self.config["answer_threshold"], self.config["scope_threshold"])
        if decision == "outside_scope":
            return {"decision": decision, "message": OUTSIDE_SCOPE, "score": score}
        if decision == "not_enough_information":
            return {"decision": decision, "message": NOT_ENOUGH_INFO, "score": score}
        record = self.kb.iloc[best]
        return {
            "decision": "answer",
            "answer": record.answer,
            "matched_question": record.question,
            "score": score,
            "source": f"AfriMed-QA v2.5 ({record.tier}, {record.country})",
            "traditional_remedy": bool(TRADITIONAL_REMEDY.search(record.question + " " + record.answer)),
        }
