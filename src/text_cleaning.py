"""Light text cleaning shared by data preparation and the web app.

The cleaning only removes noise that is not part of the medical content. It deliberately does not lowercase the
text, remove stop words or apply stemming, because the Transformer model needs complete sentences and the TF-IDF
vectorizer lowercases the text on its own.
"""
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
    """Remove answer text that was accidentally pasted at the end of a question.

    The function must run on the raw question, before its line breaks are removed. It handles three cases found in
    the data: text after the question mark that belongs to the answer ("...in Africa? Rheumatic heart disease, ..."),
    a leftover multiple-choice letter ("...in Africa? A"), and an answer written on the last line of the question
    ("...carcinoma of thyroid." followed by a line containing "MEN 2").
    """
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
