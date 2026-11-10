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
│   ├── __init__.py
│   ├── config.py          # Settings loader (pydantic-settings + .env)
│   └── logging_config.py  # Structured logging with rich colours
├── data/
│   └── .gitkeep           # Put your KB file here (not committed)
├── logs/
│   └── .gitkeep           # Rotating log files (not committed)
├── scripts/
│   └── check_models.py    # Live model availability & quality check
├── .env                   # Your secrets (NOT committed)
├── .env.example           # Template — safe to commit
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Notes

- **Never commit `.env`** — it is in `.gitignore`.
- ChromaDB data (`data/chroma/`) and the SQLite DB (`data/app.db`) are also ignored.
- Log files in `logs/` rotate at 10 MB, keeping 5 backups.
