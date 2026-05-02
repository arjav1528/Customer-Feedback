from __future__ import annotations

from typing import Any

import requests
from google_play_scraper import Sort, reviews

from utils import clean_review_text, dedupe_keep_order, normalize_app_name

APP_MAP: dict[str, dict[str, str]] = {
    "groww": {
        "play": "com.nextbillion.groww",
        "appstore": "1474699700",
    }
}

APPLE_RSS_TEMPLATE = (
    "https://itunes.apple.com/rss/customerreviews/page={page}/id={app_id}"
    "/sortby=mostrecent/json?l=en&cc=us"
)
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ProductPulseMVP/1.0; +https://streamlit.io)",
}


def fetch_play_store_reviews(app_id: str, count: int = 250) -> list[str]:
    collected: list[str] = []
    continuation_token: str | None = None

    while len(collected) < count:
        batch, continuation_token = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=min(100, count - len(collected)),
            continuation_token=continuation_token,
        )

        if not batch:
            break

        collected.extend(clean_review_text(item.get("content")) for item in batch)
        if continuation_token is None:
            break

    return dedupe_keep_order(collected)


def _extract_apple_review_text(entry: dict[str, Any]) -> str:
    content = entry.get("content", {})
    if isinstance(content, dict):
        return clean_review_text(content.get("label"))
    if isinstance(content, str):
        return clean_review_text(content)
    return ""


def fetch_app_store_reviews(app_id: str, count: int = 250, max_pages: int = 10) -> list[str]:
    collected: list[str] = []

    for page in range(1, max_pages + 1):
        if len(collected) >= count:
            break

        url = APPLE_RSS_TEMPLATE.format(page=page, app_id=app_id)
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
        response.raise_for_status()

        payload = response.json()
        entries = payload.get("feed", {}).get("entry", [])
        if isinstance(entries, dict):
            entries = [entries]
        if not entries:
            break

        added_this_page = 0
        for entry in entries:
            text = _extract_apple_review_text(entry)
            if not text:
                continue
            collected.append(text)
            added_this_page += 1
            if len(collected) >= count:
                break

        if added_this_page == 0:
            break

    return dedupe_keep_order(collected)


def get_reviews(app_name: str, count_per_store: int = 250) -> list[str]:
    app_key = normalize_app_name(app_name)
    if app_key not in APP_MAP:
        supported = ", ".join(sorted(APP_MAP))
        raise ValueError(f"Unsupported app '{app_name}'. Supported apps: {supported}.")

    app_config = APP_MAP[app_key]
    results: list[str] = []
    errors: list[str] = []

    try:
        results.extend(fetch_play_store_reviews(app_config["play"], count=count_per_store))
    except Exception as exc:  # pragma: no cover - network dependent
        errors.append(f"Play Store fetch failed: {exc}")

    try:
        results.extend(fetch_app_store_reviews(app_config["appstore"], count=count_per_store))
    except Exception as exc:  # pragma: no cover - network dependent
        errors.append(f"App Store fetch failed: {exc}")

    cleaned = dedupe_keep_order(clean_review_text(text) for text in results)
    if cleaned:
        return cleaned

    raise RuntimeError("Could not fetch reviews from either store. " + " | ".join(errors))
