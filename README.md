# tow-support-assistant

A support assistant that reads customer chat messages (English / Hindi / Hinglish) and drafts replies for human agents, grounded in a knowledge base and past approved replies.

---

## Setup

### 1. Clone & enter the project

```bash
cd tow-support-assistant
```

### 2. Create the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in **at minimum** these two secrets:

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token (read scope is enough) |

Leave `LLM_MODEL` and `EMBEDDING_MODEL` empty for now — you will fill these after the model check (step 5).

---

## Running the model check

After filling `GROQ_API_KEY` and `HF_TOKEN` in `.env`:

```bash
python scripts/check_models.py
```

This will:
- List every Groq chat model your key can see, run a ping + a Hindi/Hinglish translation test on each, and show latency & rate-limit info.
- Test each candidate HF embedding model for availability, vector size, latency, and multilingual cosine-similarity quality.
- Print a summary table and suggest top-2 LLMs and top-2 embedding models with reasoning.

**Share the output** so you can decide together which models to write into `LLM_MODEL` and `EMBEDDING_MODEL` in `.env`.

---

## Project structure

```
tow-support-assistant/
├── app/
│   ├── server.py          # FastAPI: /api/draft, /api/feedback, /api/features, /review, /api/review/*
│   ├── pipeline.py        # LangGraph draft pipeline (query -> clarify | retrieve -> hold | draft)
│   ├── retriever.py       # Hybrid search: Chroma vectors + BM25, rank fusion
│   ├── kb_ingest.py       # Rebuilds the Chroma KB index from every feature file (ids like tow:B5)
│   ├── features.py        # Feature registry (data/features.json), topics, add feature / add topic
│   ├── review.py          # Weekly Review: cluster agent edits, propose, apply to ONE feature's file
│   ├── past_replies.py    # Embedded agent edits (review workspace + short-term past replies)
│   ├── store.py           # SQLite state: chat_support_drafts, support_review_proposals
│   ├── llm.py, prompts.py, kb_common.py, keywords.py, config.py, mysql.py, logging_config.py
│   └── static/review.html # Weekly Review page
├── data/
│   ├── kb/                # tow.md, pk.md, popular-singer.md, friendship.md (one file per feature)
│   ├── kb_backups/        # copy of a KB file before each Apply
│   ├── features.json, keywords.json
│   └── app.db, chroma/    # engine state and vectors (not committed)
├── docs/web-api-contract.md  # API contract for the Lovable portal
├── scripts/               # evaluate_batch.py (evaluation set), check_models.py
├── dummy/                 # old flow kept for reference
├── .env, .env.example, requirements.txt
```

## Everyday commands

```bash
.venv/bin/python3 -m app.kb_ingest                      # after editing any data/kb/*.md
.venv/bin/uvicorn app.server:app --port 8000            # run the engine
.venv/bin/python3 scripts/evaluate_batch.py             # evaluation set (writes data/eval_results.json)
```

Full design: `../rox-chat/docs/architecture.md` (Part 1).

---

## Notes

- **Never commit `.env`** — it is in `.gitignore`.
- ChromaDB data (`data/chroma/`) and the SQLite DB (`data/app.db`) are also ignored.
- Log files in `logs/` rotate at 10 MB, keeping 5 backups.
