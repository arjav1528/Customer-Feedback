from __future__ import annotations

import os
from typing import Any

import streamlit as st

from fetch_reviews import APP_MAP, get_reviews
from llm import enrich_clusters_with_labels, generate_email_draft, generate_prd, generate_weekly_summary
from process import cluster_reviews, rank_clusters


st.set_page_config(page_title="AI Review Intelligence", page_icon="🚀", layout="wide")


def _build_analysis_payload(app_name: str, top_clusters: list[dict[str, Any]]) -> dict[str, Any]:
    labeled_clusters = enrich_clusters_with_labels(top_clusters)
    summary = generate_weekly_summary(app_name, labeled_clusters)
    prd = generate_prd(app_name, labeled_clusters[0]) if labeled_clusters else "No cluster available."
    email = generate_email_draft(app_name, labeled_clusters, summary)

    return {
        "themes": labeled_clusters,
        "summary": summary,
        "prd": prd,
        "email": email,
        "top_issue": labeled_clusters[0] if labeled_clusters else None,
    }


def _render_theme_cards(themes: list[dict[str, Any]]) -> None:
    for idx, theme in enumerate(themes, start=1):
        with st.container(border=True):
            st.subheader(f"{idx}. {theme['theme']}")
            metrics = st.columns(3)
            metrics[0].metric("Priority Score", theme["priority_score"])
            metrics[1].metric("Review Volume", theme["volume"])
            metrics[2].metric("Negative Signal", theme["negative_count"])
            st.write("Sample quotes:")
            for quote in theme["quotes"]:
                st.markdown(f"> {quote}")


def main() -> None:
    st.title("AI Review Intelligence & Product Pulse")
    st.caption("End-to-end MVP for public app review analysis.")

    with st.sidebar:
        st.header("Setup")
        st.write("Provide an app name from the hardcoded MVP mapping.")
        st.code("\n".join(f"- {name}" for name in sorted(APP_MAP)), language="text")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Leave blank to rely on OPENAI_API_KEY from the environment.",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    app_name = st.text_input("App name", value="groww", placeholder="groww")

    if st.button("Analyze", type="primary"):
        with st.spinner("Fetching reviews and building product pulse..."):
            try:
                reviews = get_reviews(app_name)
                clusters = cluster_reviews(reviews)
                ranked_clusters = rank_clusters(clusters, limit=5)
                analysis = _build_analysis_payload(app_name, ranked_clusters)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                return

        st.success(f"Analyzed {len(reviews)} unique reviews across public stores.")

        top_issue = analysis["themes"][0]["theme"] if analysis["themes"] else "N/A"
        st.metric("Top issue", top_issue)

        tab1, tab2, tab3, tab4 = st.tabs(["Top Themes", "Summary", "PRD", "Email Draft"])

        with tab1:
            _render_theme_cards(analysis["themes"])

        with tab2:
            summary = analysis["summary"]
            st.subheader("Weekly Summary")
            st.write(summary["summary"])

            st.markdown("**3 user quotes**")
            for quote in summary["quotes"]:
                st.markdown(f"> {quote}")

            st.markdown("**3 action items**")
            for action in summary["action_items"]:
                st.markdown(f"- {action}")

        with tab3:
            st.subheader("PRD for Top Issue")
            st.write(analysis["prd"])

        with tab4:
            st.subheader("Stakeholder Email Draft")
            st.write(analysis["email"])


if __name__ == "__main__":
    main()
