from __future__ import annotations

import json
import re
from typing import Iterable, Sequence


WHITESPACE_RE = re.compile(r"\s+")


def normalize_app_name(app_name: str) -> str:
    return WHITESPACE_RE.sub(" ", app_name).strip().lower()


def clean_review_text(text: str | None) -> str:
    if not text:
        return ""
    collapsed = WHITESPACE_RE.sub(" ", text).strip()
    return collapsed


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" .,;:") + "..."


def safe_json_loads(raw_text: str) -> dict:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")

    return json.loads(raw_text[start : end + 1])


def format_reviews_for_prompt(reviews: Sequence[str], limit: int = 5, max_words: int = 40) -> str:
    trimmed = [truncate_words(review, max_words) for review in reviews[:limit]]
    return "\n".join(f"- {review}" for review in trimmed)
