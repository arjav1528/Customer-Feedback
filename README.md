# Customer-Feedback

AI Review Intelligence & Product Pulse MVP built with Streamlit.

## What it does

- fetches public reviews from Google Play and Apple App Store
- embeds and clusters reviews into up to 5 themes
- scores themes by volume and negative language
- generates:
  - weekly product summary
  - 3 user quotes
  - 3 action items
  - a PRD for the top issue
  - a stakeholder email draft

## Project files

- `app.py` - Streamlit entry point
- `fetch_reviews.py` - Play Store and App Store review fetchers
- `process.py` - embeddings, clustering, scoring, quote selection
- `llm.py` - OpenAI prompting and fallback generation
- `utils.py` - shared helpers

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
streamlit run app.py
```

Optional:

```bash
export OPENAI_MODEL=gpt-4.1-mini
```

## MVP app mapping

Supported app names:

- `groww`

`fetch_reviews.py` contains the hardcoded `APP_MAP` and can be extended with more apps later.

## Notes

- Uses only public review data.
- The app strips reviews down to text only.
- If OpenAI is unavailable, the app falls back to deterministic outputs so the end-to-end demo still works.
