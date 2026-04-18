from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path

import streamlit as st

from mytho_app.constants import (
    APP_TITLE,
    CSV_COLUMNS,
    DEFAULT_OUTPUT_DIR,
    EMBEDDING_MODEL_NAME,
    FIELD_SECTIONS,
    KEYWORD_FIELD,
    LONG_TEXT_FIELDS,
    PATTERN_FIELD,
    PREVIEW_FIELDS,
)
from mytho_app.embeddings import (
    EmbeddingDependencyError,
    add_embeddings,
    create_in_memory_index,
    encode_texts,
    load_search_assets,
    semantic_search,
)
from mytho_app.parsing import blank_entry, clean_text, normalize_text, now_iso, sync_entry_fields
from mytho_app.pipeline import (
    CSVValidationError,
    entries_to_csv_bytes,
    make_manual_entry_id,
    replace_jsonl_from_entries,
    rebuild_artifacts_from_entries,
    rebuild_artifacts_from_jsonl,
    validate_uploaded_csv_bytes,
    write_entries_jsonl,
)
from mytho_app.storage import artifact_paths, artifact_status, read_json, read_jsonl
from mytho_app.ui_state import apply_pending_widget_reset, mark_widget_for_reset

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

LIVE_TERM_STATE_KEY = "_live_term_state"
ADD_ENTRY_FEEDBACK_KEY = "_add_entry_feedback"
LAST_PROCESSED_UPLOAD_KEY = "_last_processed_upload_key"


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-1: #ffffff;
            --bg-2: #ffffff;
            --ink: #23313a;
            --muted: #5b6b73;
            --accent: #0f766e;
            --accent-soft: rgba(15, 118, 110, 0.10);
            --line: rgba(35, 49, 58, 0.12);
            --warm: #b45309;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(180, 83, 9, 0.08), transparent 32%),
                radial-gradient(circle at top right, rgba(15, 118, 110, 0.12), transparent 30%),
                linear-gradient(180deg, var(--bg-1), var(--bg-2));
            color: var(--ink);
        }
        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 3rem;
        }
        .app-hero {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 40px rgba(35, 49, 58, 0.05);
            backdrop-filter: blur(6px);
        }
        .app-hero h1 {
            margin: 0;
            color: var(--ink);
            letter-spacing: -0.02em;
        }
        .app-hero p {
            margin: 0.45rem 0 0 0;
            color: var(--muted);
            font-size: 1rem;
        }
        .soft-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.9rem 1rem;
            box-shadow: 0 12px 30px rgba(35, 49, 58, 0.04);
        }
        .suggestion-row {
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            background: rgba(255, 255, 255, 0.72);
            margin-bottom: 0.55rem;
        }
        .chip {
            display: inline-block;
            background: var(--accent-soft);
            color: var(--ink);
            border: 1px solid rgba(15, 118, 110, 0.18);
            border-radius: 999px;
            padding: 0.28rem 0.58rem;
            margin: 0.15rem 0.3rem 0.15rem 0;
            font-size: 0.92rem;
        }
        .muted {
            color: var(--muted);
            font-size: 0.94rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--line);
            border-radius: 14px 14px 0 0;
            color: var(--ink) !important;
        }
        .stTabs [data-baseweb="tab"] * {
            color: inherit !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: var(--accent-soft);
            border-color: rgba(15, 118, 110, 0.22);
            color: var(--accent) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("output_dir", str(DEFAULT_OUTPUT_DIR))
    st.session_state.setdefault("add_form_version", 0)


def current_output_dir() -> Path:
    return Path(st.session_state["output_dir"]).expanduser()

def entry_label(entry: dict) -> str:
    return entry.get("label") or entry["entry_id"]


def display_status_badges(status: dict) -> None:
    badges = [
        ("JSONL", status["jsonl_ready"]),
        ("Patterns", status["patterns_index_ready"]),
        ("Keywords", status["keywords_index_ready"]),
        ("Pattern FAISS", status["patterns_faiss_ready"]),
        ("Keyword FAISS", status["keywords_faiss_ready"]),
        ("Manifest", status["manifest_ready"]),
        ("In sync", status["jsonl_synced"]),
    ]
    lb = []
    for badge in badges:
        if badge[1]:
             lb.append("{}✅".format(badge[0]))
        else:
            lb.append("{}❌".format(badge[0]))
    st.write(", ".join(lb))


def show_manifest_summary(status: dict) -> None:
    manifest = status.get("manifest") or {}
    if not manifest:
        return
    metrics = st.columns(4)
    metrics[0].metric("Entries", manifest.get("entry_count", 0))
    metrics[1].metric("Unique patterns", manifest.get("pattern_count", 0))
    metrics[2].metric("Unique keywords", manifest.get("keyword_count", 0))

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### Workspace")
        output_dir = st.text_input("Processed data directory", value=st.session_state["output_dir"])
        st.session_state["output_dir"] = output_dir

        status = artifact_status(current_output_dir())
        st.markdown("### Artifact status")
        st.caption(Path(status["output_dir"]).resolve())
        if status["ready"]:
            st.success("Everything is ready for management.")
        else:
            st.warning("Some computed artifacts are still missing.")

        page = st.radio("Navigate", ["Data processing", "Data management"], label_visibility="visible")
        st.markdown("### Model")
        st.caption(EMBEDDING_MODEL_NAME)
    return page


def render_header(page: str) -> None:
    descriptions = {
        "Data processing": "Convert the source CSV into a durable JSONL dataset, build lexical indices, and generate semantic search artifacts for motifs and keywords.",
        "Data management": "Add, edit, and delete entries while reusing existing motifs and keywords through semantic suggestions.",
    }
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>{APP_TITLE}</h1>
            <p>{descriptions[page]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def cached_search_assets(index_path: str, model_name: str):
    return load_search_assets(Path(index_path), model_name)


def clear_caches() -> None:
    cached_search_assets.clear()


def clear_live_term_state() -> None:
    st.session_state.pop(LIVE_TERM_STATE_KEY, None)


def prune_live_term_state(output_dir: Path) -> None:
    state = st.session_state.get(LIVE_TERM_STATE_KEY)
    if not state:
        return

    state = get_live_term_state(output_dir)
    for kind in ("patterns", "keywords"):
        _, base_items, _ = get_term_artifacts(kind, output_dir)
        base_markers = {item["normalized"] for item in base_items}
        bucket = state[kind]
        if not bucket["items"]:
            continue
        bucket["items"] = [item for item in bucket["items"] if item["normalized"] not in base_markers]
        rebuild_live_term_index(kind, output_dir)


def get_term_artifacts(kind: str, output_dir: Path) -> tuple[Path, list[dict], str]:
    paths = artifact_paths(output_dir)
    manifest = read_json(paths["manifest"]) if paths["manifest"].exists() else {}
    model_name = manifest.get("embedding_model_name", EMBEDDING_MODEL_NAME)
    if kind == "patterns":
        return paths["patterns_faiss"], load_term_index(paths["patterns_index"]), model_name
    return paths["keywords_faiss"], load_term_index(paths["keywords_index"]), model_name


def get_live_term_state(output_dir: Path) -> dict:
    resolved_output_dir = str(output_dir.resolve())
    state = st.session_state.get(LIVE_TERM_STATE_KEY)
    if not state or state.get("output_dir") != resolved_output_dir:
        state = {
            "output_dir": resolved_output_dir,
            "patterns": {"items": [], "index": None},
            "keywords": {"items": [], "index": None},
        }
        st.session_state[LIVE_TERM_STATE_KEY] = state
    return state


def reserve_live_term(kind: str, term: str, output_dir: Path) -> None:
    marker = normalize_text(term)
    if not marker:
        return

    bucket = get_live_term_state(output_dir)[kind]
    for item in bucket["items"]:
        if item["normalized"] == marker:
            item["selection_count"] += 1
            return


def rebuild_live_term_index(kind: str, output_dir: Path) -> None:
    bucket = get_live_term_state(output_dir)[kind]
    if not bucket["items"]:
        bucket["index"] = None
        return

    index_path, _, model_name = get_term_artifacts(kind, output_dir)
    model, _ = cached_search_assets(str(index_path), model_name)
    embeddings = encode_texts(model, [item["text"] for item in bucket["items"]])
    index = create_in_memory_index(int(embeddings.shape[1]))
    add_embeddings(index, embeddings)
    bucket["index"] = index


def release_live_term(kind: str, term: str, output_dir: Path) -> None:
    marker = normalize_text(term)
    if not marker:
        return

    bucket = get_live_term_state(output_dir)[kind]
    for position, item in enumerate(bucket["items"]):
        if item["normalized"] != marker:
            continue
        item["selection_count"] = max(0, item["selection_count"] - 1)
        if item["selection_count"] == 0:
            bucket["items"].pop(position)
            rebuild_live_term_index(kind, output_dir)
        return


def register_live_term(kind: str, term: str, output_dir: Path) -> bool:
    value = clean_text(term)
    marker = normalize_text(value)
    if not marker:
        return False

    index_path, base_items, model_name = get_term_artifacts(kind, output_dir)
    if any(item["normalized"] == marker for item in base_items):
        return False

    bucket = get_live_term_state(output_dir)[kind]
    for item in bucket["items"]:
        if item["normalized"] == marker:
            item["selection_count"] += 1
            return False

    model, _ = cached_search_assets(str(index_path), model_name)
    embedding = encode_texts(model, [value])
    if bucket["index"] is None:
        bucket["index"] = create_in_memory_index(int(embedding.shape[1]))
    add_embeddings(bucket["index"], embedding)
    bucket["items"].append(
        {
            "normalized": marker,
            "text": value,
            "entry_count": 0,
            "is_live": True,
            "selection_count": 1,
        }
    )
    return True


def add_term_to_selection(state_key: str, term: str) -> bool:
    existing = list(st.session_state.get(state_key, []))
    updated = list(dict.fromkeys(existing + [term]))
    st.session_state[state_key] = updated
    return len(updated) > len(existing)


def load_term_index(index_path: Path) -> list[dict]:
    return read_json(index_path) if index_path.exists() else []


def lexical_fallback(items: list[dict], query: str, limit: int = 5) -> list[dict]:
    marker = normalize_text(query)
    results = []
    for position, item in enumerate(items):
        score = 1.0 if marker == item["normalized"] else 0.0
        if marker and marker in item["normalized"]:
            score = max(score, 0.72)
        elif marker and any(token in item["normalized"] for token in marker.split()):
            score = max(score, 0.48)
        if score:
            results.append({"text": item["text"], "score": score, "position": position})
    results.sort(key=lambda row: (-row["score"], row["text"].lower()))
    return results[:limit]


def find_similar_terms(kind: str, query: str, output_dir: Path) -> tuple[list[dict], str]:
    index_path, items, model_name = get_term_artifacts(kind, output_dir)
    live_state = get_live_term_state(output_dir)[kind]
    live_items = live_state["items"]
    live_index = live_state["index"]

    if not items and not live_items:
        return [], "No indexed terms found yet."

    try:
        model, index = cached_search_assets(str(index_path), model_name)
        ranked_by_term: dict[str, dict] = {}

        for result in semantic_search(model, index, [item["text"] for item in items], query, limit=5):
            item = items[result["position"]]
            ranked_by_term[item["normalized"]] = {
                "text": item["text"],
                "score": result["score"],
                "entry_count": item["entry_count"],
                "is_live": False,
            }

        if live_items and live_index is not None:
            for result in semantic_search(model, live_index, [item["text"] for item in live_items], query, limit=5):
                item = live_items[result["position"]]
                current = ranked_by_term.get(item["normalized"])
                candidate = {
                    "text": item["text"],
                    "score": result["score"],
                    "entry_count": item["entry_count"],
                    "is_live": True,
                }
                if current is None or candidate["score"] > current["score"]:
                    ranked_by_term[item["normalized"]] = candidate

        enriched = sorted(ranked_by_term.values(), key=lambda row: (-row["score"], row["text"].lower()))[:5]
        return enriched, "semantic + live updates" if live_items else "semantic"
    except EmbeddingDependencyError:
        combined_items = [dict(item, is_live=False) for item in items] + [dict(item, is_live=True) for item in live_items]
        fallback = lexical_fallback(combined_items, query, limit=5)
        enriched = []
        for result in fallback:
            item = combined_items[result["position"]]
            enriched.append(
                {
                    "text": item["text"],
                    "score": result["score"],
                    "entry_count": item["entry_count"],
                    "is_live": item.get("is_live", False),
                }
            )
        return enriched, "lexical fallback"


def render_processing_page() -> None:
    status = artifact_status(current_output_dir())
    jsonl_path = artifact_paths(current_output_dir())["jsonl"]

    st.markdown("### Source data")
    st.info(
        "Upload a CSV from your computer to begin. The app validates the file and, if it is well-formed, "
        "immediately rebuilds the JSONL dataset, the lexical indices, and both semantic indices."
    )

    c1, c2 = st.columns([1.15, 1.0])
    with c1:
        uploaded_file = st.file_uploader(
            "Upload a source CSV",
            type=["csv"],
            help="Uploading a new file replaces the current processed dataset after validation succeeds.",
        )
        st.caption("A timestamped backup of the existing JSONL is kept automatically before replacement.")

        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            upload_key = f"{current_output_dir().resolve()}::{hashlib.sha256(uploaded_bytes).hexdigest()}"
            if st.session_state.get(LAST_PROCESSED_UPLOAD_KEY) != upload_key:
                progress = st.progress(0)
                status_box = st.status(f"Validating `{uploaded_file.name}`...", expanded=True)
                try:
                    validation = validate_uploaded_csv_bytes(uploaded_bytes)
                    status_box.write(
                        f"Validation passed: {validation['entry_count']} data rows and "
                        f"{validation['column_count']} columns."
                    )
                    progress.progress(20, text="Validation complete")

                    status_box.update(label="Writing JSONL dataset...", state="running")
                    replace_jsonl_from_entries(validation["entries"], current_output_dir(), backup_existing=True)
                    progress.progress(55, text="JSONL updated")

                    status_box.update(label="Building pattern, keyword, and FAISS indices...", state="running")
                    manifest = rebuild_artifacts_from_entries(
                        validation["entries"],
                        current_output_dir(),
                        source_csv=uploaded_file.name,
                    )
                    progress.progress(100, text="Processing finished")

                    clear_caches()
                    clear_live_term_state()
                    st.session_state[LAST_PROCESSED_UPLOAD_KEY] = upload_key
                    status_box.update(label="Upload validated and processing finished.", state="complete", expanded=False)
                    st.success(
                        f"Processed `{uploaded_file.name}` successfully: {manifest['entry_count']} entries, "
                        f"{manifest['pattern_count']} patterns, and {manifest['keyword_count']} keywords."
                    )
                    status = artifact_status(current_output_dir())
                except CSVValidationError as exc:
                    status_box.update(label="Upload validation failed.", state="error", expanded=True)
                    st.error(str(exc))
                except EmbeddingDependencyError as exc:
                    status_box.update(label="Upload saved to JSONL, but processing failed.", state="error", expanded=True)
                    st.error(
                        f"The file passed validation and the JSONL was replaced, but semantic processing failed: {exc}"
                    )
            else:
                st.caption(f"`{uploaded_file.name}` is already the current processed source for this output directory.")

        st.markdown("### Export")
        st.caption("Download a new CSV rebuilt from the current JSONL dataset.")
        if jsonl_path.exists():
            export_entries = read_jsonl(jsonl_path)
            export_bytes = entries_to_csv_bytes(export_entries)
            export_stamp = datetime.fromtimestamp(jsonl_path.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
            st.download_button(
                "Export latest JSONL to CSV",
                data=export_bytes,
                file_name=f"entries-from-jsonl-{export_stamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("Export latest JSONL to CSV", disabled=True, use_container_width=True)
            st.caption("The export button will be available after a valid CSV upload has been processed.")

    with c2:
        st.markdown("### Current artifact status")
        display_status_badges(status)
        show_manifest_summary(status)
        st.caption(
            "Uploading a valid CSV replaces the working dataset. After that, edits happen in JSONL and stay in sync "
            "with the semantic indices automatically."
        )


def preview_entry(entry: dict) -> None:
    fields = entry.get("fields", {})
    preview_lines = []
    for field in PREVIEW_FIELDS:
        value = clean_text(fields.get(field, ""))
        if value:
            preview_lines.append(f"**{field}**: {value}")
    preview_lines.append(f"**Patterns**: {len(entry.get('patterns', []))}")
    preview_lines.append(f"**Keywords**: {len(entry.get('keywords', []))}")
    st.markdown("\n\n".join(preview_lines))


def render_selected_terms(prefix: str, label: str, terms: list[str], *, kind: str, output_dir: Path) -> list[str]:
    st.markdown(f"**Selected {label.lower()}**")
    if not terms:
        st.caption(f"No {label.lower()} selected yet.")
        return terms

    updated = list(terms)
    for index, term in enumerate(terms):
        text_col, button_col = st.columns([6, 3])
        text_col.markdown(f"<span class='chip'>{term}</span>", unsafe_allow_html=True)
        if button_col.button("Remove", key=f"{prefix}_remove_{label}_{index}"):
            release_live_term(kind, term, output_dir)
            updated.pop(index)
            st.session_state[f"{prefix}_{label}"] = updated
            st.rerun()
    return updated


def render_term_picker(
    *,
    prefix: str,
    label: str,
    kind: str,
    output_dir: Path,
) -> list[str]:
    state_key = f"{prefix}_{label}"
    input_key = f"{prefix}_{label}_input"
    selected_terms = st.session_state.get(state_key, [])
    st.session_state[state_key] = render_selected_terms(
        prefix,
        label,
        selected_terms,
        kind=kind,
        output_dir=output_dir,
    )
    selected_terms = st.session_state[state_key]

    apply_pending_widget_reset(st.session_state, input_key)
    candidate = st.text_input(
        f"Add a {label[:-1].lower()}",
        key=input_key,
        placeholder=f"Write a {label[:-1].lower()} to search similar existing ones",
    )
    if candidate.strip():
        suggestions, mode = find_similar_terms(kind, candidate, output_dir)
        if suggestions:
            st.caption(f"Top suggestions ({mode})")
            for index, suggestion in enumerate(suggestions):
                row = st.columns([5.5, 1.4])
                if suggestion.get("is_live"):
                    summary = "new in this session, not yet saved"
                else:
                    count = suggestion["entry_count"]
                    summary = f"{count} linked entry" if count == 1 else f"{count} linked entries"
                row[0].markdown(
                    (
                        "<div class='suggestion-row'>"
                        f"<strong>{suggestion['text']}</strong><br>"
                        f"<span class='muted'>Similarity {suggestion['score']:.3f} · "
                        f"{summary}</span>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                if row[1].button("Use it", key=f"{prefix}_{label}_suggestion_{index}"):
                    if add_term_to_selection(state_key, suggestion["text"]):
                        reserve_live_term(kind, suggestion["text"], output_dir)
                    mark_widget_for_reset(st.session_state, input_key)
                    st.rerun()
        else:
            st.caption("No close match found yet.")

    add_col, clear_col = st.columns([1, 1])
    if add_col.button(f"Keep typed {label[:-1].lower()}", key=f"{prefix}_{label}_keep"):
        value = clean_text(st.session_state.get(input_key, ""))
        if value:
            if add_term_to_selection(state_key, value):
                try:
                    register_live_term(kind, value, output_dir)
                except EmbeddingDependencyError as exc:
                    st.warning(
                        f"The term was added to the draft, but the live semantic index could not be refreshed: {exc}"
                    )
            mark_widget_for_reset(st.session_state, input_key)
            st.rerun()
    if clear_col.button("Clear input", key=f"{prefix}_{label}_clear"):
        mark_widget_for_reset(st.session_state, input_key)
        st.rerun()

    return st.session_state[state_key]


def render_fields_editor(prefix: str, draft: dict) -> dict[str, str]:
    fields = draft["fields"]
    updated_fields = dict(fields)

    for section_title, columns in FIELD_SECTIONS:
        with st.expander(section_title, expanded=section_title == "Titles and Summaries"):
            for field_name in columns:
                widget_key = f"{prefix}_{field_name}"
                initial_value = clean_text(fields.get(field_name, ""))
                if field_name in LONG_TEXT_FIELDS:
                    updated_fields[field_name] = st.text_area(
                        field_name,
                        value=initial_value,
                        height=120 if field_name in {"Abstract (Eng)", "Abstract (Fr)"} else 90,
                        key=widget_key,
                    )
                else:
                    updated_fields[field_name] = st.text_input(
                        field_name,
                        value=initial_value,
                        key=widget_key,
                    )

    for field_name in CSV_COLUMNS:
        if field_name in {PATTERN_FIELD, KEYWORD_FIELD}:
            continue
        if any(field_name in section_fields for _, section_fields in FIELD_SECTIONS):
            continue
        widget_key = f"{prefix}_{field_name}"
        updated_fields[field_name] = st.text_input(
            field_name,
            value=clean_text(fields.get(field_name, "")),
            key=widget_key,
        )

    return updated_fields


def compose_entry(prefix: str, base_entry: dict) -> dict:
    draft = {
        "entry_id": base_entry["entry_id"],
        "source_row_number": base_entry.get("source_row_number"),
        "record_origin": base_entry.get("record_origin", "manual"),
        "created_at": base_entry.get("created_at", now_iso()),
        "updated_at": now_iso(),
        "fields": render_fields_editor(prefix, base_entry),
        "patterns": st.session_state.get(f"{prefix}_Patterns", list(base_entry.get("patterns", []))),
        "keywords": st.session_state.get(f"{prefix}_Keywords", list(base_entry.get("keywords", []))),
    }
    return sync_entry_fields(draft)


def render_add_tab(output_dir: Path) -> None:
    version = st.session_state["add_form_version"]
    prefix = f"add_{version}"
    base_key = f"{prefix}_base"
    feedback = st.session_state.pop(ADD_ENTRY_FEEDBACK_KEY, None)

    if base_key not in st.session_state:
        entry = blank_entry()
        entry["entry_id"] = make_manual_entry_id()
        entry["created_at"] = now_iso()
        entry["updated_at"] = entry["created_at"]
        st.session_state[base_key] = entry
        st.session_state[f"{prefix}_Patterns"] = []
        st.session_state[f"{prefix}_Keywords"] = []

    base_entry = st.session_state[base_key]

    st.markdown("### New entry")
    st.caption("Patterns and keywords are managed separately so the semantic helper can suggest close existing terms.")
    if feedback:
        st.success(feedback)

    left, right = st.columns([1.2, 1.2])
    with left:
        patterns = render_term_picker(prefix=prefix, label="Patterns", kind="patterns", output_dir=output_dir)
    with right:
        keywords = render_term_picker(prefix=prefix, label="Keywords", kind="keywords", output_dir=output_dir)

    st.session_state[f"{prefix}_Patterns"] = patterns
    st.session_state[f"{prefix}_Keywords"] = keywords

    draft = compose_entry(prefix, base_entry)
    preview_entry(draft)

    if st.button("Save new entry", use_container_width=True):
        status_box = st.status("Saving the entry and reprocessing the dataset...", expanded=True)
        status_box.write("Writing the new entry to the JSONL dataset.")
        entries = read_jsonl(artifact_paths(output_dir)["jsonl"])
        entries.append(draft)
        write_entries_jsonl(entries, output_dir, backup_existing=True)
        try:
            status_box.update(label="Reprocessing patterns, keywords, and semantic indices...", state="running")
            manifest = rebuild_artifacts_from_entries(entries, output_dir)
            clear_caches()
            prune_live_term_state(output_dir)
            status_box.update(label="Entry saved and reprocessing finished.", state="complete", expanded=False)
            st.session_state[ADD_ENTRY_FEEDBACK_KEY] = (
                f"Entry {draft['entry_id']} saved successfully. Reprocessing finished for "
                f"{manifest['entry_count']} entries, {manifest['pattern_count']} patterns, "
                f"and {manifest['keyword_count']} keywords."
            )
            st.session_state["add_form_version"] += 1
            st.rerun()
        except EmbeddingDependencyError as exc:
            status_box.update(label="Entry saved, but reprocessing failed.", state="error", expanded=True)
            st.error(
                f"Entry saved to JSONL, but the semantic indices could not be refreshed: {exc}"
            )


def render_edit_tab(output_dir: Path, entries: list[dict]) -> None:
    if not entries:
        st.info("No entries are available yet.")
        return

    labels = [entry_label(entry) for entry in entries]
    selected_label = st.selectbox("Choose an entry to edit", labels)
    entry = entries[labels.index(selected_label)]
    prefix = f"edit_{entry['entry_id']}"

    st.session_state.setdefault(f"{prefix}_Patterns", list(entry.get("patterns", [])))
    st.session_state.setdefault(f"{prefix}_Keywords", list(entry.get("keywords", [])))

    left, right = st.columns([1.2, 1.2])
    with left:
        patterns = render_term_picker(prefix=prefix, label="Patterns", kind="patterns", output_dir=output_dir)
    with right:
        keywords = render_term_picker(prefix=prefix, label="Keywords", kind="keywords", output_dir=output_dir)

    st.session_state[f"{prefix}_Patterns"] = patterns
    st.session_state[f"{prefix}_Keywords"] = keywords

    updated_entry = compose_entry(prefix, entry)
    preview_entry(updated_entry)

    if st.button("Save changes", use_container_width=True):
        updated_entries = []
        for current in entries:
            updated_entries.append(updated_entry if current["entry_id"] == updated_entry["entry_id"] else current)
        write_entries_jsonl(updated_entries, output_dir, backup_existing=True)
        try:
            rebuild_artifacts_from_entries(updated_entries, output_dir)
            clear_caches()
            prune_live_term_state(output_dir)
            st.success(f"Updated entry {updated_entry['entry_id']} and refreshed all indices.")
        except EmbeddingDependencyError as exc:
            st.error(
                f"Entry saved to JSONL, but the semantic indices could not be refreshed: {exc}"
            )


def render_delete_tab(output_dir: Path, entries: list[dict]) -> None:
    if not entries:
        st.info("No entries are available yet.")
        return

    labels = [entry_label(entry) for entry in entries]
    selected_label = st.selectbox("Choose an entry to delete", labels, key="delete_select")
    entry = entries[labels.index(selected_label)]
    preview_entry(entry)

    checkbox = st.checkbox("I understand this will permanently delete the selected entry from the JSONL.")
    if st.button("Delete entry", type="secondary", use_container_width=True, disabled=not checkbox):
        remaining = [current for current in entries if current["entry_id"] != entry["entry_id"]]
        write_entries_jsonl(remaining, output_dir, backup_existing=True)
        try:
            rebuild_artifacts_from_entries(remaining, output_dir)
            clear_caches()
            prune_live_term_state(output_dir)
            st.success(f"Deleted entry {entry['entry_id']} and refreshed all indices.")
            st.rerun()
        except EmbeddingDependencyError as exc:
            st.error(
                f"Entry deleted from JSONL, but the semantic indices could not be refreshed: {exc}"
            )


def ensure_management_artifacts_ready(output_dir: Path) -> tuple[dict, str | None, str | None]:
    status = artifact_status(output_dir)
    if status["ready"] or not status["jsonl_ready"]:
        return status, None, None

    try:
        with st.spinner("Refreshing indices from the current JSONL..."):
            manifest = rebuild_artifacts_from_jsonl(output_dir)
        clear_caches()
        prune_live_term_state(output_dir)
        refreshed_status = artifact_status(output_dir)
        message = (
            f"Indices refreshed automatically from JSONL: {manifest['entry_count']} entries, "
            f"{manifest['pattern_count']} patterns, {manifest['keyword_count']} keywords."
        )
        return refreshed_status, message, None
    except EmbeddingDependencyError as exc:
        return artifact_status(output_dir), None, str(exc)


def render_management_page() -> None:
    output_dir = current_output_dir()
    get_live_term_state(output_dir)
    status, auto_refresh_message, auto_refresh_error = ensure_management_artifacts_ready(output_dir)
    st.markdown("### Dataset summary")
    show_manifest_summary(status)

    if auto_refresh_message:
        st.success(auto_refresh_message)
    if auto_refresh_error:
        st.error(
            "The app detected that the computed indices were missing or out of date, "
            f"but automatic refresh failed: {auto_refresh_error}"
        )

    if not status["ready"]:
        st.markdown("### Artifact verification")
        display_status_badges(status)
        st.warning(
            "The management interface is locked until the JSONL, both lexical indices, both FAISS indices, "
            "and the manifest have all been created."
        )
        return

    if auto_refresh_error:
        with st.expander("Artifact verification details", expanded=True):
            display_status_badges(status)

    jsonl_path = artifact_paths(output_dir)["jsonl"]
    entries = read_jsonl(jsonl_path)
    entries.sort(key=lambda item: entry_label(item).lower())

    add_tab, edit_tab, delete_tab = st.tabs(["Add entry", "Edit entry", "Delete entry"])
    with add_tab:
        render_add_tab(output_dir)
    with edit_tab:
        render_edit_tab(output_dir, entries)
    with delete_tab:
        render_delete_tab(output_dir, entries)


def main() -> None:
    apply_theme()
    init_state()
    page = render_sidebar()
    render_header(page)

    if page == "Data processing":
        render_processing_page()
    else:
        render_management_page()


if __name__ == "__main__":
    main()
