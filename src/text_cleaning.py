"""This module removes noise such as answer prefixes, exam marks and leaked answers but keeps the medical wording."""
import re

NBSP = " "  # A non-breaking space, which appears in some records.
SPEAKER_TAG = re.compile(r"\[speaker \d+\]:", re.IGNORECASE)               # "[Speaker 2]:"
EXAM_MARKS = re.compile(r"\(\s*\d+\s*(?:marks?|mks?|m)\s*\)", re.IGNORECASE)  # "(2marks)", "(5m)"
ANSWER_PREFIX = re.compile(r"^\s*answer\s*:\s*", re.IGNORECASE)              # "Answer: ..."
MCQ_OPTION = re.compile(r"(?<=\s)[A-E]\)\s+(?=[A-Z])")                      # "The reason B) Diabetes ..."


def normalize_whitespace(text: str) -> str:
    """Turn every run of spaces, tabs and line breaks into a single space."""
    return re.sub(r"\s+", " ", text.replace(NBSP, " ")).strip()


def clean_question(text: str) -> str:
    """Clean a question, from the dataset or typed by a user, into one tidy line."""
    text = SPEAKER_TAG.sub(" ", text)
    text = EXAM_MARKS.sub(" ", text)
    return normalize_whitespace(text)


def clean_answer(text: str) -> str:
    """Clean an answer but keep its line breaks, because many expert answers are lists."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").replace(NBSP, " ")
    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    text = ANSWER_PREFIX.sub("", text)
    text = MCQ_OPTION.sub("", text)
    return text.lstrip(".,;: ").strip()


def strip_leaked_answer(question: str, answer: str) -> str:
    """Remove answer text left in a raw question after the question mark, as an option letter or on the last line."""
    if "?" in question:
        head, tail = question.rsplit("?", 1)
        tail = normalize_whitespace(tail)
        leaked = len(tail) > 3 and tail.lower() in normalize_whitespace(answer).lower()
        if leaked or re.fullmatch(r"[A-E]\)?", tail):
            return head + "?"

    lines = [line for line in question.splitlines() if line.strip()]
    if len(lines) > 1 and normalize_whitespace(lines[-1]).lower() == normalize_whitespace(answer).lower():
        return "\n".join(lines[:-1])
    return question
