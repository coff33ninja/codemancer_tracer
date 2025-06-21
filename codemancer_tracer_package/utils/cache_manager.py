import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(".llm_cache")

def get_codebase_hash(files: list[Path]) -> str:
    """Generates a SHA256 hash of the codebase content."""
    hasher = hashlib.sha256()
    for filepath in sorted(files):
        with open(filepath, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()

def load_cached_summary(codebase_hash: str) -> str | None:
    """Loads cached AI summary if available."""
    cache_file = CACHE_DIR / f"{codebase_hash}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)["summary"]
    return None

def save_cached_summary(codebase_hash: str, summary: str):
    """Saves AI summary to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{codebase_hash}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f)