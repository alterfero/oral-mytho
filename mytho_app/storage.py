from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

from mytho_app.constants import (
    JSONL_FILENAME,
    KEYWORD_FAISS_FILENAME,
    KEYWORD_INDEX_FILENAME,
    MANIFEST_FILENAME,
    PATTERN_FAISS_FILENAME,
    PATTERN_INDEX_FILENAME,
)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup_dir = ensure_directory(path.parent / "backups")
    timestamp = path.stat().st_mtime_ns
    backup_path = backup_dir / f"{path.stem}-{timestamp}{path.suffix}"
    backup_path.write_bytes(path.read_bytes())
    return backup_path


def atomic_write_text(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        tmp.write(content)
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def write_json(path: Path, payload: dict | list) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False))


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, entries: list[dict], *, backup_existing: bool = False) -> None:
    if backup_existing:
        backup_file(path)
    lines = [json.dumps(entry, ensure_ascii=False, sort_keys=False) for entry in entries]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                entries.append(json.loads(stripped))
    return entries


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    base = ensure_directory(output_dir)
    return {
        "output_dir": base,
        "jsonl": base / JSONL_FILENAME,
        "manifest": base / MANIFEST_FILENAME,
        "patterns_index": base / PATTERN_INDEX_FILENAME,
        "keywords_index": base / KEYWORD_INDEX_FILENAME,
        "patterns_faiss": base / PATTERN_FAISS_FILENAME,
        "keywords_faiss": base / KEYWORD_FAISS_FILENAME,
    }


def artifact_status(output_dir: Path) -> dict:
    paths = artifact_paths(output_dir)
    manifest = read_json(paths["manifest"]) if paths["manifest"].exists() else {}
    jsonl_mtime_ns = paths["jsonl"].stat().st_mtime_ns if paths["jsonl"].exists() else None
    jsonl_synced = bool(manifest) and manifest.get("jsonl_mtime_ns") == jsonl_mtime_ns
    status = {
        "output_dir": str(paths["output_dir"]),
        "jsonl_ready": paths["jsonl"].exists(),
        "patterns_index_ready": paths["patterns_index"].exists(),
        "keywords_index_ready": paths["keywords_index"].exists(),
        "patterns_faiss_ready": paths["patterns_faiss"].exists(),
        "keywords_faiss_ready": paths["keywords_faiss"].exists(),
        "manifest_ready": paths["manifest"].exists(),
        "jsonl_synced": jsonl_synced,
        "ready": False,
        "manifest": manifest,
    }
    status["ready"] = all(
        [
            status["jsonl_ready"],
            status["patterns_index_ready"],
            status["keywords_index_ready"],
            status["patterns_faiss_ready"],
            status["keywords_faiss_ready"],
            status["manifest_ready"],
            status["jsonl_synced"],
        ]
    )
    return status


def clear_artifacts(output_dir: Path) -> dict[str, int]:
    base = Path(output_dir)
    deleted_files = 0
    deleted_dirs = 0

    files_to_delete = [
        base / JSONL_FILENAME,
        base / MANIFEST_FILENAME,
        base / PATTERN_INDEX_FILENAME,
        base / KEYWORD_INDEX_FILENAME,
        base / PATTERN_FAISS_FILENAME,
        base / KEYWORD_FAISS_FILENAME,
    ]
    for path in files_to_delete:
        if path.exists():
            path.unlink()
            deleted_files += 1

    backups_dir = base / "backups"
    if backups_dir.exists():
        shutil.rmtree(backups_dir)
        deleted_dirs += 1

    if base.exists() and not any(base.iterdir()):
        base.rmdir()
        deleted_dirs += 1

    return {"deleted_files": deleted_files, "deleted_dirs": deleted_dirs}
