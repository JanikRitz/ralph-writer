from __future__ import annotations

import re


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def line_number_of_index(text: str, index: int) -> int:
    return text[:index].count("\n") + 1


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    remaining = len(text) - max_chars
    return f"{text[:max_chars]}… (+{remaining} chars)"
