"""This module ranks every knowledge-base answer for each question and saves the resulting metrics and predictions."""

import numpy as np
import pandas as pd

from src.config import PROBE_DIR, PROCESSED_DIR, RESULTS_DIR
from src.retrieval import gold_ranks, retrieval_metrics

SPLITS = ["val", "test", "paraphrase"]  # The paraphrases are 40 reworded validation and test questions.


def load_queries(split: str) -> tuple[pd.DataFrame, bool]:
    """Return the questions of one set with a flag that is true for dataset questions and false for paraphrases."""
    if split != "paraphrase":
        return pd.read_csv(PROCESSED_DIR / f"{split}.csv").assign(source_split=split), True
    para = pd.read_csv(PROBE_DIR / "paraphrases.csv")
    para = para.merge(pd.read_csv(PROCESSED_DIR / "saq_clean.csv")[["id", "answer"]], on="id")
    return pd.DataFrame({"id": para.id, "question": para.paraphrase, "answer": para.answer,
                         "source_split": para.split}), False


def evaluate_split(retriever, kb: pd.DataFrame, split: str) -> tuple[dict, pd.DataFrame]:
    """Rank the knowledge-base answers for every question of one set and return the metrics and predictions."""
    queries, dataset_questions = load_queries(split)
    answer_to_row = {answer: i for i, answer in enumerate(kb.answer)}
    gold = queries.answer.map(answer_to_row).to_numpy()
    assert not pd.isna(gold).any(), "every question's answer must be in the knowledge base"
    gold = gold.astype(int)

    if dataset_questions and getattr(retriever, "question_vectors", None) is not None:
        # A held-out dataset question must not be matched with its own copy in the knowledge base.
        scores = retriever.scores(queries.question.tolist(), hide_own_question=gold)
    else:
        scores = retriever.scores(queries.question.tolist())
    ranks = gold_ranks(scores, gold)
    top1 = scores.argmax(axis=1)
    predictions = pd.DataFrame({
        "id": queries.id,
        "source_split": queries.source_split,
        "question": queries.question,
        "rank_of_correct_answer": ranks,
        "correct_at_1": ranks == 1,
        "top1_score": scores.max(axis=1).round(4),
        "correct_answer_score": scores[np.arange(len(gold)), gold].round(4),
        "top1_matched_question": kb.question.to_numpy()[top1],
        "top1_answer": kb.answer.to_numpy()[top1],
        "correct_answer": queries.answer,
    })
    metrics = retrieval_metrics(ranks)
    metrics["questions"] = len(queries)
    return metrics, predictions


def run_experiment(name: str, retriever, kb: pd.DataFrame, description: str, **info) -> dict:
    """Evaluate one retriever on all three sets and save its predictions and its row in the experiments table."""
    out_dir = RESULTS_DIR / "experiments" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"name": name, "description": description, **info}
    for split in SPLITS:
        metrics, predictions = evaluate_split(retriever, kb, split)
        results[split] = metrics
        predictions.to_csv(out_dir / f"predictions_{split}.csv", index=False)

    row = {"name": name, "description": description, **info,
           **{f"{s}_{k}": results[s][k] for s in SPLITS for k in ["recall@1", "recall@3", "mrr"]}}
    path = RESULTS_DIR / "experiments.csv"
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not table.empty:
        table = table[table.name != name]
    pd.concat([table, pd.DataFrame([row])], ignore_index=True).sort_values("name").to_csv(path, index=False)
    return results


def experiments_table(names: list[str] | None = None) -> pd.DataFrame:
    """Return the summary table of saved experiments, optionally limited to the given names in that order."""
    table = pd.read_csv(RESULTS_DIR / "experiments.csv")
    if names:
        table = table.set_index("name").loc[names].reset_index()
    cols = ["name", "description"] + [f"{s}_{k}" for s in SPLITS for k in ["recall@1", "recall@3", "mrr"]]
    return table[cols]
