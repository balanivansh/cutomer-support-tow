"""Keyword list for the query builder, kept in data/keywords.json. The weekly review can add words."""
import json

from app.kb_common import KB_PATH, log_error

KW_PATH = KB_PATH.parent / "keywords.json"
DEFAULTS = ["Tug of War", "free credit", "Gold Coins", "Health Heart", "weekly review", "recording",
            "Reveal Time", "leaderboard", "daily live target", "Piggy icon", "broadcaster", "participant"]


def save(words):
    KW_PATH.write_text(json.dumps({"keywords": words}, ensure_ascii=False, indent=1), encoding="utf-8")


def load():
    try:
        return json.loads(KW_PATH.read_text(encoding="utf-8"))["keywords"]
    except FileNotFoundError:
        save(DEFAULTS)
        return list(DEFAULTS)
    except Exception as e:
        log_error("CONFIG_ERROR", "keywords", "Could not read keywords.json, using defaults", e)
        return list(DEFAULTS)


def add(new_words):
    """Adds words that are not in the list yet (case-insensitive). Returns the words added."""
    current = load()
    seen = {w.lower() for w in current}
    added = []
    for w in new_words:
        w = w.strip()
        if w and w.lower() not in seen:
            seen.add(w.lower())
            added.append(w)
    if added:
        save(current + added)
    return added
