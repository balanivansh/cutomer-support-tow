"""Features shown on the review page, kept in data/features.json (created on first use).
Each feature: id, name, kb_file (a file inside data/) and audiences (names besides "Everyone").
Topics are read from the KB file headings: "## PART B: BROADCASTER" gives the topic "Broadcaster"."""
import json
import re
import threading

from app.kb_common import DATA_DIR, KB_DIR, log_error

FEATURES_PATH = DATA_DIR / "features.json"
PART_RE = re.compile(r"^##\s+PART\s+([A-Z]):\s*(.*)$")
# The id is also the prefix of the chunk ids ("tow:B5"): never change it once chunks exist.
# kb_file is a file inside data/kb/.
_AUD = ["Broadcaster", "Participant"]
DEFAULTS = [
    {"id": "tow", "name": "Tug of War", "kb_file": "tow.md", "audiences": list(_AUD)},
    {"id": "pk", "name": "PK Battle", "kb_file": "pk.md", "audiences": list(_AUD)},
    {"id": "popular_singer", "name": "Popular Singer Contest", "kb_file": "popular-singer.md",
     "audiences": list(_AUD)},
    {"id": "friendship", "name": "Friendship Feature", "kb_file": "friendship.md", "audiences": list(_AUD)},
]
# Starter file for a feature added later from the review page (the 4 real features have their own files).
TEMPLATE = """<!-- {name}: support knowledge base. One question = one "### <letter><number>. Question" block,
     then an "**Audience:**" line (Everyone, Broadcaster or Participant), then the answer in plain English. -->

## PART A: OVERVIEW
"""
_lock = threading.Lock()


def _defaults():
    return json.loads(json.dumps(DEFAULTS))


def _save(items):
    FEATURES_PATH.write_text(json.dumps({"features": items}, ensure_ascii=False, indent=1), encoding="utf-8")


def load():
    try:
        return json.loads(FEATURES_PATH.read_text(encoding="utf-8"))["features"]
    except FileNotFoundError:
        _save(DEFAULTS)
        return _defaults()
    except Exception as e:
        log_error("CONFIG_ERROR", "features", "Could not read features.json, using defaults", e)
        return _defaults()


def get(fid):
    return next((f for f in load() if f["id"] == fid), None)


def find_by_name(name):
    n = (name or "").strip().lower()
    return next((f for f in load() if f["name"].lower() == n), None) if n else None


def kb_path(feature):
    return KB_DIR / feature["kb_file"]


def topics(feature):
    """[{"part": "B", "name": "Broadcaster"}, ...] in file order."""
    try:
        text = kb_path(feature).read_text(encoding="utf-8")
    except OSError as e:
        log_error("CONFIG_ERROR", "features", f"Cannot read the file of feature {feature.get('name')}", e)
        return []
    out = []
    for line in text.splitlines():
        m = PART_RE.match(line.rstrip())
        if m:
            out.append({"part": m.group(1), "name": m.group(2).strip().title()})
    return out


def add_audience(fid, name):
    """Adds an audience to ONE feature only. Returns that feature's audience list."""
    name = " ".join((name or "").split())
    if not name:
        raise ValueError("Type a name")
    if len(name) > 30:
        raise ValueError("The name is too long")
    if "," in name:
        raise ValueError("Do not use a comma in the name")
    if name.lower() == "everyone":
        raise ValueError("Everyone is already there")
    with _lock:
        items = load()
        f = next((x for x in items if x["id"] == fid), None)
        if not f:
            raise ValueError("Feature not found")
        auds = f.setdefault("audiences", [])
        if name.lower() not in {a.lower() for a in auds}:
            auds.append(name)
            _save(items)
        return list(auds)


def add_feature(name, kb_file=None, audiences=None):
    """Add a new feature to features.json."""
    name = " ".join((name or "").strip().split())
    if not name:
        raise ValueError("Feature name cannot be empty")
    fid = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not fid:
        raise ValueError("Use letters or numbers in the feature name")
    with _lock:
        items = load()
        if any(f["id"] == fid or f["name"].lower() == name.lower() for f in items):
            raise ValueError(f"Feature '{name}' already exists")
        new_f = {
            "id": fid,
            "name": name,
            "kb_file": kb_file or f"{fid}.md",
            "audiences": audiences or list(_AUD)
        }
        path = kb_path(new_f)
        if path.exists():
            raise ValueError(f"The file {path.name} already exists")
        KB_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(TEMPLATE.format(name=name), encoding="utf-8")
        items.append(new_f)
        _save(items)
        return new_f


def add_topic(fid, name):
    """Human action: adds a "## PART X: NAME" heading at the end of ONE feature's KB file. Returns the topic list.
    No re-index needed: a topic with no questions has no chunks yet."""
    name = " ".join((name or "").split())
    if not name:
        raise ValueError("Type a topic name")
    if len(name) > 40 or "#" in name or ":" in name:
        raise ValueError("The topic name is too long or has a # or : in it")
    with _lock:
        f = next((x for x in load() if x["id"] == fid), None)
        if not f:
            raise ValueError("Feature not found")
        existing = topics(f)
        if name.lower() in {t["name"].lower() for t in existing}:
            raise ValueError("This topic already exists")
        used = {t["part"] for t in existing}
        letter = next((chr(c) for c in range(ord("A"), ord("Z") + 1) if chr(c) not in used), None)
        if not letter:
            raise ValueError("No free topic letter left")
        path = kb_path(f)
        text = path.read_text(encoding="utf-8").rstrip("\n")
        path.write_text(f"{text}\n\n---\n\n## PART {letter}: {name.upper()}\n", encoding="utf-8")
        return topics(f)
