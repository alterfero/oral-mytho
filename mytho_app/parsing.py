from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable

from mytho_app.constants import ABSTRACT_FIELDS, CSV_COLUMNS, KEYWORD_FIELD, PATTERN_FIELD, TITLE_FIELDS

WHITESPACE_RE = re.compile(r"\s+")
KEYWORD_SPLIT_RE = re.compile(r"[;\n]+")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\ufeff", "").strip()


def normalize_text(value: str) -> str:
    return WHITESPACE_RE.sub(" ", clean_text(value)).strip().lower()


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        item = clean_text(value)
        if not item:
            continue
        marker = normalize_text(item)
        if marker in seen:
            continue
        seen.add(marker)
        cleaned.append(item)
    return cleaned


def split_keywords(value: str) -> list[str]:
    parts = [clean_text(piece) for piece in KEYWORD_SPLIT_RE.split(clean_text(value))]
    return dedupe_preserve_order(parts)


def split_patterns(value: str) -> list[str]:
    text = clean_text(value).replace("\r\n", "\n")
    if not text:
        return []
    if "§§" in text:
        pieces = [piece.strip(" \n;") for piece in text.split("§§")]
    else:
        pieces = [piece.strip(" \n;") for piece in re.split(r"[;\n]+", text)]
    return dedupe_preserve_order(pieces)


def serialize_keywords(values: Iterable[str]) -> str:
    return " ; ".join(dedupe_preserve_order(values))


def serialize_patterns(values: Iterable[str]) -> str:
    items = dedupe_preserve_order(values)
    if not items:
        return ""
    return "\n".join(f"§§ {item}" for item in items)


def build_entry_label(entry: dict) -> str:
    fields = entry.get("fields", {})
    titles = [clean_text(fields.get(field, "")) for field in TITLE_FIELDS]
    for title in titles:
        if title:
            return f"{title} [{entry['entry_id']}]"
    summary = clean_text(fields.get("1-sentence summary", ""))
    if summary:
        snippet = summary[:72] + ("..." if len(summary) > 72 else "")
        return f"{snippet} [{entry['entry_id']}]"
    return entry["entry_id"]


def build_search_text(fields: dict[str, str], patterns: list[str], keywords: list[str]) -> str:
    values: list[str] = []
    for field in TITLE_FIELDS + ABSTRACT_FIELDS:
        text = clean_text(fields.get(field, ""))
        if text:
            values.append(text)
    values.extend(patterns)
    values.extend(keywords)
    return "\n".join(values)


def ensure_all_columns(fields: dict[str, str]) -> dict[str, str]:
    normalized = {column: clean_text(fields.get(column, "")) for column in CSV_COLUMNS}
    extras = {key: clean_text(value) for key, value in fields.items() if key not in normalized}
    normalized.update(extras)
    return normalized


def sync_entry_fields(entry: dict) -> dict:
    fields = ensure_all_columns(entry.get("fields", {}))
    patterns = dedupe_preserve_order(entry.get("patterns", split_patterns(fields.get(PATTERN_FIELD, ""))))
    keywords = dedupe_preserve_order(entry.get("keywords", split_keywords(fields.get(KEYWORD_FIELD, ""))))
    fields[PATTERN_FIELD] = serialize_patterns(patterns)
    fields[KEYWORD_FIELD] = serialize_keywords(keywords)

    entry["fields"] = fields
    entry["patterns"] = patterns
    entry["keywords"] = keywords
    entry["search_text"] = build_search_text(fields, patterns, keywords)
    entry["label"] = build_entry_label(entry)
    entry["updated_at"] = now_iso()
    return entry


def csv_row_to_entry(row: dict[str, str], row_number: int) -> dict:
    fields = ensure_all_columns(row)
    timestamp = now_iso()
    entry = {
        "entry_id": f"csv-{row_number:05d}",
        "source_row_number": row_number,
        "record_origin": "csv_import",
        "created_at": timestamp,
        "updated_at": timestamp,
        "fields": fields,
        "patterns": split_patterns(fields.get(PATTERN_FIELD, "")),
        "keywords": split_keywords(fields.get(KEYWORD_FIELD, "")),
    }
    return sync_entry_fields(entry)


def blank_entry() -> dict:
    timestamp = now_iso()
    entry = {
        "entry_id": "",
        "source_row_number": None,
        "record_origin": "manual",
        "created_at": timestamp,
        "updated_at": timestamp,
        "fields": {column: "" for column in CSV_COLUMNS},
        "patterns": [],
        "keywords": [],
    }
    return sync_entry_fields(entry)

