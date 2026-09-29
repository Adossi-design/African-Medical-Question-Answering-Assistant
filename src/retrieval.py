"""This module represents texts as TF-IDF vectors or sentence embeddings and ranks answers by cosine similarity."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import PROCESSED_DIR


def load_knowledge_base() -> pd.DataFrame:
    """Return the searchable collection, with one row per unique answer from all cleaned records."""
    df = pd.read_csv(PROCESSED_DIR / "saq_clean.csv")
    return df.drop_duplicates(subset="answer").reset_index(drop=True)


class TfidfRetriever:
    """This class is the baseline retriever, which represents each text as a sparse vector of TF-IDF weights."""

    name = "tfidf"

    def __init__(self, answers: list[str]):
        # Common search settings: English stop words removed, unigrams and bigrams, and dampened term counts.
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english",
                                          ngram_range=(1, 2), sublinear_tf=True)
        self.answer_vectors = self.vectorizer.fit_transform(answers)

    def scores(self, questions: list[str]) -> np.ndarray:
        """Return the cosine similarity between every question and every answer."""
        question_vectors = self.vectorizer.transform(questions)
        # The vectors have length 1, so their dot product is the cosine similarity.
        return (question_vectors @ self.answer_vectors.T).toarray()


class EmbeddingRetriever:
    """This class represents each text as a 384-number embedding and can also compare with dataset questions."""

    def __init__(self, model, answers: list[str], kb_questions: list[str] | None = None,
                 max_seq_length: int = 512, batch_size: int = 64):
        # Imported here so that the TF-IDF baseline can run without PyTorch.
        from sentence_transformers import SentenceTransformer

        self.model = model if isinstance(model, SentenceTransformer) else SentenceTransformer(model, device="cpu")
        self.model.max_seq_length = max_seq_length
        self.answer_vectors = self.encode(answers, batch_size)
        self.question_vectors = self.encode(kb_questions, batch_size) if kb_questions is not None else None

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

    def scores(self, questions: list[str], hide_own_question: np.ndarray | None = None) -> np.ndarray:
        query_vectors = self.encode(questions)
        scores = query_vectors @ self.answer_vectors.T
        if self.question_vectors is not None:
            question_scores = query_vectors @ self.question_vectors.T
            if hide_own_question is not None:
                # During evaluation, a held-out question must not be matched with its own copy.
                question_scores[np.arange(len(questions)), hide_own_question] = -1.0
            scores = np.maximum(scores, question_scores)
        return scores


def gold_ranks(scores: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Return the rank of each correct answer, where 1 is the best rank and ties count against the model."""
    gold_scores = scores[np.arange(len(gold)), gold][:, None]
    return 1 + (scores > gold_scores).sum(axis=1) + (scores == gold_scores).sum(axis=1) - 1


def retrieval_metrics(ranks: np.ndarray) -> dict:
    """Compute Recall@1, Recall@3 and the mean reciprocal rank (MRR) from the ranks of the correct answers."""
    return {"recall@1": round(float(np.mean(ranks <= 1)), 4),
            "recall@3": round(float(np.mean(ranks <= 3)), 4),
            "mrr": round(float(np.mean(1 / ranks)), 4)}
