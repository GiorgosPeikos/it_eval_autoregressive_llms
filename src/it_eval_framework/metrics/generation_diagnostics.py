from __future__ import annotations

import re
from collections import Counter


def repeated_ngram_rate(text: str, n: int = 3) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeats = sum(count - 1 for count in counts.values() if count > 1)
    return repeats / max(len(grams), 1)


def distinct_n(text: str, n: int) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def unfinished_output(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped[-1] not in ".!?»\"”’)"


def suspicious_artifacts(text: str) -> list[str]:
    flags = []
    if "\ufffd" in text:
        flags.append("replacement_character")
    if re.search(r"[ÃÂ][\w]", text):
        flags.append("possible_mojibake")
    if re.search(r"(.)\1{7,}", text):
        flags.append("long_character_repeat")
    return flags


def summarize_generation(text: str) -> dict:
    return {
        "output_length_chars": len(text),
        "output_length_words": len(text.split()),
        "repeated_3gram_rate": repeated_ngram_rate(text, 3),
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "distinct_3": distinct_n(text, 3),
        "unfinished_output": unfinished_output(text),
        "artifact_flags": suspicious_artifacts(text),
    }
