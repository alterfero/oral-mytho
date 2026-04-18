from __future__ import annotations

import csv
import io
import uuid
from pathlib import Path

from mytho_app.constants import CSV_COLUMNS, EMBEDDING_MODEL_NAME
from mytho_app.embeddings import build_faiss_index
from mytho_app.parsing import clean_text, csv_row_to_entry, normalize_text, now_iso, sync_entry_fields
from mytho_app.storage import artifact_paths, read_jsonl, write_json, write_jsonl


class CSVValidationError(ValueError):
    """Raised when an uploaded CSV is malformed or missing required columns."""


def _load_csv_entries_from_handle(handle) -> tuple[list[dict], list[str]]:
    reader = csv.DictReader(handle)
    raw_fieldnames = reader.fieldnames or []
    fieldnames = [clean_text(name) for name in raw_fieldnames if name is not None]
    if not fieldnames:
        raise CSVValidationError("The uploaded file does not contain a readable CSV header row.")

    missing_columns = [column for column in CSV_COLUMNS if column not in fieldnames]
    if missing_columns:
        preview = ", ".join(missing_columns[:5])
        suffix = "..." if len(missing_columns) > 5 else ""
        raise CSVValidationError(f"The uploaded CSV is missing required columns: {preview}{suffix}")

    entries: list[dict] = []
    for row_number, row in enumerate(reader, start=1):
        extra_values = row.get(None, []) if row else []
        if any(clean_text(value) for value in extra_values):
            raise CSVValidationError(
                f"Data row {row_number} has more values than the header defines. "
                "Please check quoting and separators in the uploaded CSV."
            )

        normalized_row = {
            clean_text(key): clean_text(value)
            for key, value in (row or {}).items()
            if key is not None
        }
        if not any(normalized_row.values()):
            continue
        entries.append(csv_row_to_entry(normalized_row, row_number))

    if not entries:
        raise CSVValidationError("The uploaded CSV has a header row but no story entries.")

    return entries, fieldnames


def load_csv_entries(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        entries, _ = _load_csv_entries_from_handle(handle)
    return entries


def validate_uploaded_csv_bytes(csv_bytes: bytes) -> dict:
    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CSVValidationError(
            "The uploaded file could not be decoded as UTF-8 CSV. Please export it as UTF-8 and try again."
        ) from exc

    entries, fieldnames = _load_csv_entries_from_handle(io.StringIO(text, newline=""))
    return {
        "entries": entries,
        "fieldnames": fieldnames,
        "entry_count": len(entries),
        "column_count": len(fieldnames),
    }


def write_entries_jsonl(entries: list[dict], output_dir: Path, *, backup_existing: bool = False) -> Path:
    paths = artifact_paths(output_dir)
    normalized = [sync_entry_fields(entry) for entry in entries]
    write_jsonl(paths["jsonl"], normalized, backup_existing=backup_existing)
    return paths["jsonl"]


def replace_jsonl_from_entries(entries: list[dict], output_dir: Path, *, backup_existing: bool = True) -> Path:
    paths = artifact_paths(output_dir)
    jsonl_path = write_entries_jsonl(entries, output_dir, backup_existing=backup_existing)
    for stale_key in ("manifest", "patterns_index", "keywords_index", "patterns_faiss", "keywords_faiss"):
        if paths[stale_key].exists():
            paths[stale_key].unlink()
    return jsonl_path


def build_term_index(entries: list[dict], field_name: str) -> list[dict]:
    index: dict[str, dict] = {}
    for entry in entries:
        for term in entry.get(field_name, []):
            marker = normalize_text(term)
            if not marker:
                continue
            payload = index.setdefault(
                marker,
                {"normalized": marker, "text": term, "entry_ids": [], "entry_count": 0},
            )
            if entry["entry_id"] not in payload["entry_ids"]:
                payload["entry_ids"].append(entry["entry_id"])

    items = []
    for payload in index.values():
        payload["entry_ids"].sort()
        payload["entry_count"] = len(payload["entry_ids"])
        items.append(payload)

    items.sort(key=lambda item: (-item["entry_count"], item["text"].lower()))
    return items


def rebuild_artifacts_from_entries(
    entries: list[dict],
    output_dir: Path,
    *,
    source_csv: Path | None = None,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> dict:
    paths = artifact_paths(output_dir)

    patterns = build_term_index(entries, "patterns")
    keywords = build_term_index(entries, "keywords")

    write_json(paths["patterns_index"], patterns)
    write_json(paths["keywords_index"], keywords)

    pattern_stats = build_faiss_index([item["text"] for item in patterns], model_name, paths["patterns_faiss"])
    keyword_stats = build_faiss_index([item["text"] for item in keywords], model_name, paths["keywords_faiss"])

    manifest = {
        "created_at": now_iso(),
        "source_csv": str(source_csv) if source_csv else None,
        "jsonl_mtime_ns": paths["jsonl"].stat().st_mtime_ns if paths["jsonl"].exists() else None,
        "entry_count": len(entries),
        "pattern_count": len(patterns),
        "keyword_count": len(keywords),
        "embedding_model_name": model_name,
        "artifacts": {
            "jsonl": str(paths["jsonl"]),
            "patterns_index": str(paths["patterns_index"]),
            "keywords_index": str(paths["keywords_index"]),
            "patterns_faiss": str(paths["patterns_faiss"]),
            "keywords_faiss": str(paths["keywords_faiss"]),
        },
        "pattern_vector_index": pattern_stats,
        "keyword_vector_index": keyword_stats,
    }
    write_json(paths["manifest"], manifest)
    return manifest


def import_csv_to_jsonl(csv_path: Path, output_dir: Path, *, backup_existing: bool = True) -> dict:
    entries = load_csv_entries(csv_path)
    jsonl_path = replace_jsonl_from_entries(entries, output_dir, backup_existing=backup_existing)
    return {"entries": entries, "jsonl_path": str(jsonl_path), "entry_count": len(entries)}


def rebuild_artifacts_from_jsonl(output_dir: Path, *, model_name: str = EMBEDDING_MODEL_NAME) -> dict:
    entries = read_jsonl(artifact_paths(output_dir)["jsonl"])
    return rebuild_artifacts_from_entries(entries, output_dir, model_name=model_name)


def entry_to_csv_row(entry: dict) -> dict[str, str]:
    normalized_entry = sync_entry_fields(
        {
            "entry_id": entry.get("entry_id", ""),
            "fields": dict(entry.get("fields", {})),
            "patterns": list(entry.get("patterns", [])),
            "keywords": list(entry.get("keywords", [])),
        }
    )
    return {column: clean_text(normalized_entry["fields"].get(column, "")) for column in CSV_COLUMNS}


def entries_to_csv_bytes(entries: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for entry in entries:
        writer.writerow(entry_to_csv_row(entry))
    return buffer.getvalue().encode("utf-8-sig")


def make_manual_entry_id() -> str:
    return f"manual-{uuid.uuid4().hex[:8]}"
