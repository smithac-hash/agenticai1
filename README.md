# NewsAI
Run: `pip install -r requirements.txt && python app.py` → http://localhost:5000, then click **Fetch news**.
Optional: `export ANTHROPIC_API_KEY=...` for LLM summaries (otherwise extractive summarizer is used).
Pipeline: RSS ingest → keyword categorizer → summarizer → TF-IDF cosine clustering (duplicates) → per-user ranking (interest weights + keywords + recency) with 👍/👎 feedback learning.
