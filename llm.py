from __future__ import annotations

import os
from typing import Any, Sequence

from openai import OpenAI

from process import select_user_quotes
from utils import format_reviews_for_prompt, safe_json_loads, truncate_words


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def _chat_completion(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    client = _get_client()
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=temperature,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    )
    return response.output_text.strip()


def _fallback_summary(app_name: str, clusters: Sequence[dict[str, Any]]) -> dict[str, Any]:
    top_three = list(clusters[:3])
    issue_descriptions = [
        f"{cluster['theme']} (score {cluster['priority_score']}, {cluster['volume']} reviews)"
        for cluster in top_three
    ]
    summary = (
        f"This week's {app_name.title()} product pulse highlights {', '.join(issue_descriptions)}. "
        "User feedback points to reliability and usability pain points that are affecting key journeys. "
        "The highest priority cluster should be addressed first because it combines strong review volume "
        "with repeated negative language. The next two clusters are important follow-on opportunities to "
        "improve trust, reduce friction, and limit repeat support demand."
    )
    quotes: list[str] = []
    for cluster in top_three:
        quotes.extend(cluster.get("quotes", []))
    action_items = [
        f"Investigate root causes behind {cluster['theme'].lower()}."
        for cluster in top_three[:3]
    ]
    default_actions = [
        "Review top complaint patterns from recent reviews and support tickets.",
        "Prioritize the highest-friction user journey for a focused fix sprint.",
        "Add instrumentation to measure whether the shipped fix reduces negative feedback.",
    ]
    while len(action_items) < 3:
        action_items.append(default_actions[len(action_items)])
    return {
        "summary": truncate_words(summary, 250),
        "quotes": quotes[:3],
        "action_items": action_items[:3],
    }


def generate_theme_label(cluster_reviews: Sequence[str], fallback_theme: str = "General feedback") -> str:
    prompt = (
        "Summarize these user reviews into a 2-3 word product issue theme.\n"
        "Be specific, avoid punctuation, and do not mention sentiment.\n\n"
        f"{format_reviews_for_prompt(cluster_reviews, limit=6, max_words=35)}"
    )
    try:
        raw = _chat_completion(
            system_prompt="You are a product analyst who names review clusters.",
            user_prompt=prompt,
            temperature=0.1,
        )
        label = raw.replace('"', "").strip()
        return truncate_words(label, 4) or fallback_theme
    except Exception:
        return fallback_theme


def generate_weekly_summary(app_name: str, clusters: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cluster_lines = []
    for index, cluster in enumerate(clusters[:3], start=1):
        sample_reviews = format_reviews_for_prompt(cluster["reviews"], limit=3, max_words=25)
        cluster_lines.append(
            f"Issue {index}: {cluster['theme']}\n"
            f"Priority score: {cluster['priority_score']}\n"
            f"Volume: {cluster['volume']}\n"
            f"Sample reviews:\n{sample_reviews}"
        )

    prompt = (
        "Create a concise weekly product pulse from the review insights below.\n"
        "Return valid JSON with exactly these keys:\n"
        "{\n"
        '  "summary": string (max 250 words),\n'
        '  "quotes": [string, string, string],\n'
        '  "action_items": [string, string, string]\n'
        "}\n"
        "Use real quotes from the provided reviews when possible.\n"
        "Do not include any personally identifying information.\n\n"
        + "\n\n".join(cluster_lines)
    )
    try:
        raw = _chat_completion(
            system_prompt="You are an expert PM creating a weekly product pulse for stakeholders.",
            user_prompt=prompt,
            temperature=0.3,
        )
        parsed = safe_json_loads(raw)
        return {
            "summary": truncate_words(str(parsed.get("summary", "")), 250),
            "quotes": list(parsed.get("quotes", []))[:3],
            "action_items": list(parsed.get("action_items", []))[:3],
        }
    except Exception:
        return _fallback_summary(app_name, clusters)


def generate_prd(app_name: str, top_cluster: dict[str, Any]) -> str:
    prompt = (
        "Create a product requirement doc including these sections:\n"
        "- Problem\n"
        "- User Impact\n"
        "- Hypothesis\n"
        "- Solution\n"
        "- Metrics\n\n"
        f"App: {app_name}\n"
        f"Theme: {top_cluster['theme']}\n"
        f"Priority score: {top_cluster['priority_score']}\n"
        f"Volume: {top_cluster['volume']}\n"
        "Representative user reviews:\n"
        f"{format_reviews_for_prompt(top_cluster['reviews'], limit=8, max_words=35)}"
    )
    try:
        return _chat_completion(
            system_prompt="You are a senior product manager writing concise PRDs.",
            user_prompt=prompt,
            temperature=0.25,
        )
    except Exception:
        return (
            f"Problem\n{top_cluster['theme']} is the most urgent issue in {app_name.title()} based on review volume "
            f"and negative language.\n\n"
            "User Impact\nUsers are encountering friction in a core journey, reducing trust and satisfaction.\n\n"
            "Hypothesis\nIf we remove the main source of failure and clarify the workflow, complaints in this theme "
            "will decline and user confidence will improve.\n\n"
            "Solution\nAudit the end-to-end journey, reproduce the top failure modes from review text, ship the highest "
            "impact fix first, and add instrumentation around the affected flow.\n\n"
            "Metrics\nReduction in negative reviews for this theme, improved completion rate for the affected journey, "
            "and lower support/contact volume."
        )


def generate_email_draft(app_name: str, clusters: Sequence[dict[str, Any]], summary: dict[str, Any]) -> str:
    top_issue = clusters[0]["theme"] if clusters else "General feedback"
    actions_text = "\n".join(f"- {item}" for item in summary.get("action_items", [])[:3])
    prompt = (
        "Write a short email summarizing weekly product insights for stakeholders.\n"
        "Keep it crisp, professional, and under 180 words.\n\n"
        f"App: {app_name}\n"
        f"Top issue: {top_issue}\n"
        f"Summary:\n{summary.get('summary', '')}\n\n"
        f"Action items:\n{actions_text}"
    )
    try:
        return _chat_completion(
            system_prompt="You are a product lead sending a stakeholder update.",
            user_prompt=prompt,
            temperature=0.35,
        )
    except Exception:
        return (
            f"Subject: Weekly Product Pulse - {app_name.title()}\n\n"
            f"Team,\n\nThis week's review analysis shows {top_issue.lower()} as the top user issue. "
            f"{summary.get('summary', '')}\n\n"
            "Recommended next steps:\n"
            f"{actions_text}\n\n"
            "Thanks."
        )


def enrich_clusters_with_labels(clusters: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for cluster in clusters:
        reviews = cluster.get("reviews", [])
        quotes = cluster.get("quotes") or select_user_quotes(reviews, count=3)
        theme = generate_theme_label(reviews, fallback_theme=cluster.get("fallback_theme", "General feedback"))
        enriched.append({**cluster, "theme": theme, "quotes": quotes})
    return enriched
