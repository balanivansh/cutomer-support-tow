"""Features shown on the review page, kept in data/features.json (created on first use).
Each feature: id, name, kb_file (a file inside data/) and audiences (names besides "Everyone").
Topics are read from the KB file headings: "## PART B: BROADCASTER" gives the topic "Broadcaster"."""
import json
import re
import threading

from app.kb_common import KB_PATH, log_error

FEATURES_PATH = KB_PATH.parent / "features.json"
PART_RE = re.compile(r"^##\s+PART\s+([A-Z]):\s*(.*)$")
DEFAULTS = [{"id": "tow", "name": "Tug of War", "kb_file": KB_PATH.name,
             "audiences": ["Broadcaster", "Participant"]}]
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
    return KB_PATH.parent / feature["kb_file"]


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
