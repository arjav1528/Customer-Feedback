from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil
from typing import Any

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


NEGATIVE_TERMS = ("fail", "not", "error", "slow", "crash", "issue", "stuck", "broken")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def cluster_reviews(reviews: list[str], max_clusters: int = 5) -> dict[int, list[str]]:
    if not reviews:
        return {}

    if len(reviews) == 1:
        return {0: reviews[:]}

    cluster_count = min(max_clusters, max(1, ceil(len(reviews) / 20)))
    cluster_count = min(cluster_count, len(reviews))

    if cluster_count == 1:
        return {0: reviews[:]}

    embeddings = get_embedding_model().encode(reviews, show_progress_bar=False)
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = model.fit_predict(embeddings)

    clusters: dict[int, list[str]] = defaultdict(list)
    for label, review in zip(labels, reviews, strict=True):
        clusters[int(label)].append(review)

    return dict(clusters)


def count_negative_signals(reviews: list[str], negative_terms: tuple[str, ...] = NEGATIVE_TERMS) -> int:
    negative_count = 0
    for review in reviews:
        text = review.lower()
        if any(term in text for term in negative_terms):
            negative_count += 1
    return negative_count


def score_cluster(reviews: list[str]) -> dict[str, Any]:
    volume = len(reviews)
    negative_count = count_negative_signals(reviews)
    priority_score = volume * (negative_count + 1)
    return {
        "volume": volume,
        "negative_count": negative_count,
        "priority_score": priority_score,
    }


def _keyword_summary(reviews: list[str], top_n: int = 6) -> list[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "have",
        "from",
        "your",
        "after",
        "they",
        "when",
        "just",
        "very",
        "been",
        "into",
        "there",
        "their",
        "app",
        "not",
        "but",
        "too",
        "you",
        "are",
        "was",
        "had",
        "its",
        "get",
        "can",
        "all",
        "about",
    }
    counter: Counter[str] = Counter()
    for review in reviews:
        for raw_word in review.lower().split():
            word = "".join(ch for ch in raw_word if ch.isalpha())
            if len(word) < 4 or word in stopwords:
                continue
            counter[word] += 1
    return [word for word, _ in counter.most_common(top_n)]


def fallback_theme_name(reviews: list[str]) -> str:
    keywords = _keyword_summary(reviews, top_n=3)
    if not keywords:
        return "General feedback"
    return " / ".join(word.title() for word in keywords[:2])


def select_user_quotes(reviews: list[str], count: int = 3) -> list[str]:
    ranked = sorted(reviews, key=lambda text: (-count_negative_signals([text]), -len(text)))
    quotes: list[str] = []
    for review in ranked:
        stripped = review.strip()
        if len(stripped.split()) < 5:
            continue
        quotes.append(stripped)
        if len(quotes) == count:
            break
    return quotes


def analyze_clusters(clusters: dict[int, list[str]]) -> list[dict[str, Any]]:
    scored_clusters: list[dict[str, Any]] = []
    for cluster_id, reviews in clusters.items():
        metrics = score_cluster(reviews)
        scored_clusters.append(
            {
                "cluster_id": cluster_id,
                "reviews": reviews,
                "sample_reviews": reviews[:5],
                "quotes": select_user_quotes(reviews, count=3),
                "fallback_theme": fallback_theme_name(reviews),
                **metrics,
            }
        )

    return sorted(scored_clusters, key=lambda item: item["priority_score"], reverse=True)


def rank_clusters(clusters: dict[int, list[str]], limit: int = 5) -> list[dict[str, Any]]:
    return analyze_clusters(clusters)[:limit]
