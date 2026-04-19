from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path

from mytho_app.constants import CSV_COLUMNS
from mytho_app.parsing import (
    csv_row_to_entry,
    serialize_keywords,
    serialize_patterns,
    split_keywords,
    split_patterns,
    sync_entry_fields,
)
from mytho_app.pipeline import CSVValidationError, build_term_index, entries_to_csv_bytes, validate_uploaded_csv_bytes, write_entries_jsonl
from mytho_app.storage import artifact_paths, clear_artifacts, read_jsonl, write_json
from mytho_app.ui_state import apply_pending_widget_reset, mark_widget_for_reset, pending_widget_reset_key


class ParsingTests(unittest.TestCase):
    def test_split_patterns_handles_section_markers(self) -> None:
        text = "§§ first pattern\n§§ second pattern\n§§ first pattern"
        self.assertEqual(split_patterns(text), ["first pattern", "second pattern"])

    def test_split_keywords_deduplicates_values(self) -> None:
        text = "wolf ; moon\nwolf;  river"
        self.assertEqual(split_keywords(text), ["wolf", "moon", "river"])

    def test_sync_entry_fields_serializes_terms(self) -> None:
        entry = {
            "entry_id": "manual-1",
            "fields": {"Motifs (Eng)": "", "Keywords (Eng)": ""},
            "patterns": ["pattern one", "pattern two"],
            "keywords": ["keyword one", "keyword two"],
        }
        synced = sync_entry_fields(entry)
        self.assertEqual(synced["fields"]["Motifs (Eng)"], serialize_patterns(["pattern one", "pattern two"]))
        self.assertEqual(synced["fields"]["Keywords (Eng)"], serialize_keywords(["keyword one", "keyword two"]))


class PipelineTests(unittest.TestCase):
    def test_term_index_collects_entry_ids(self) -> None:
        entries = [
            {"entry_id": "e1", "patterns": ["A clever child"], "keywords": ["moon"]},
            {"entry_id": "e2", "patterns": ["A clever child", "Sky woman"], "keywords": ["moon", "river"]},
        ]
        patterns = build_term_index(entries, "patterns")
        clever_child = next(item for item in patterns if item["text"] == "A clever child")
        self.assertEqual(clever_child["entry_count"], 2)
        self.assertEqual(clever_child["entry_ids"], ["e1", "e2"])

    def test_write_entries_jsonl_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            entries = [csv_row_to_entry({"Story title (Eng)": "Test title", "Motifs (Eng)": "§§ pattern", "Keywords (Eng)": "wolf"}, 1)]
            write_entries_jsonl(entries, output_dir)
            loaded = read_jsonl(artifact_paths(output_dir)["jsonl"])
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["fields"]["Story title (Eng)"], "Test title")

    def test_entries_to_csv_bytes_exports_normalized_rows(self) -> None:
        entry = {
            "entry_id": "manual-1",
            "fields": {"Story title (Eng)": "Test title", "Motifs (Eng)": "", "Keywords (Eng)": ""},
            "patterns": ["pattern one", "pattern two"],
            "keywords": ["wolf", "moon"],
        }
        exported = entries_to_csv_bytes([entry]).decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(exported))
        rows = list(reader)
        self.assertEqual(reader.fieldnames, CSV_COLUMNS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Story title (Eng)"], "Test title")
        self.assertEqual(rows[0]["Keywords (Eng)"], "wolf ; moon")
        self.assertEqual(rows[0]["Motifs (Eng)"], "§§ pattern one\n§§ pattern two")

    def test_validate_uploaded_csv_bytes_accepts_expected_schema(self) -> None:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "Story title (Eng)": "Test title",
                "Keywords (Eng)": "wolf",
                "Motifs (Eng)": "§§ pattern",
            }
        )
        validation = validate_uploaded_csv_bytes(buffer.getvalue().encode("utf-8"))
        self.assertEqual(validation["entry_count"], 1)
        self.assertEqual(validation["column_count"], len(CSV_COLUMNS))
        self.assertEqual(validation["entries"][0]["fields"]["Story title (Eng)"], "Test title")

    def test_validate_uploaded_csv_bytes_rejects_missing_columns(self) -> None:
        csv_text = "Story title (Eng),Keywords (Eng)\nTest,wolf\n"
        with self.assertRaises(CSVValidationError):
            validate_uploaded_csv_bytes(csv_text.encode("utf-8"))

    def test_clear_artifacts_removes_generated_files_and_backups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "processed"
            entries = [csv_row_to_entry({"Story title (Eng)": "Test title", "Motifs (Eng)": "§§ pattern", "Keywords (Eng)": "wolf"}, 1)]
            write_entries_jsonl(entries, output_dir)
            write_entries_jsonl(entries, output_dir, backup_existing=True)

            paths = artifact_paths(output_dir)
            write_json(paths["manifest"], {"entry_count": 1})
            paths["patterns_index"].write_text("[]", encoding="utf-8")
            paths["keywords_index"].write_text("[]", encoding="utf-8")
            paths["patterns_faiss"].write_text("pattern-faiss", encoding="utf-8")
            paths["keywords_faiss"].write_text("keyword-faiss", encoding="utf-8")

            summary = clear_artifacts(output_dir)

            self.assertEqual(summary["deleted_files"], 6)
            self.assertGreaterEqual(summary["deleted_dirs"], 1)
            self.assertFalse(output_dir.exists())


class UIStateTests(unittest.TestCase):
    def test_mark_and_apply_widget_reset(self) -> None:
        session_state = {"pattern_input": "wolf"}
        mark_widget_for_reset(session_state, "pattern_input")
        self.assertTrue(session_state[pending_widget_reset_key("pattern_input")])

        apply_pending_widget_reset(session_state, "pattern_input")

        self.assertEqual(session_state["pattern_input"], "")
        self.assertNotIn(pending_widget_reset_key("pattern_input"), session_state)


if __name__ == "__main__":
    unittest.main()
