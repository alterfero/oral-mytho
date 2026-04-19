from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path

import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
except ModuleNotFoundError:  # pragma: no cover - optional UI dependency
    folium = None
    st_folium = None

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
from mytho_app.exploration import (
    build_exploration_network,
    entry_id_from_map_click,
    entry_title,
    popup_html_for_entry,
    primary_abstract,
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
from mytho_app.storage import artifact_paths, artifact_status, clear_artifacts, read_json, read_jsonl
from mytho_app.ui_state import apply_pending_widget_reset, mark_widget_for_reset

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

LIVE_TERM_STATE_KEY = "_live_term_state"
ADD_ENTRY_FEEDBACK_KEY = "_add_entry_feedback"
LAST_PROCESSED_UPLOAD_KEY = "_last_processed_upload_key"
EXPLORATION_SELECTED_ENTRY_KEY = "_exploration_selected_entry_id"
CLEAN_SESSION_FEEDBACK_KEY = "_clean_session_feedback"


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

        page = st.radio("Navigate", ["Data processing", "Data management", "Exploration"], label_visibility="visible")
        st.markdown("### Model")
        st.caption(EMBEDDING_MODEL_NAME)
    return page


def render_header(page: str) -> None:
    descriptions = {
        "Data processing": "Convert the source CSV into a durable JSONL dataset, build lexical indices, and generate semantic search artifacts for motifs and keywords.",
        "Data management": "Add, edit, and delete entries while reusing existing motifs and keywords through semantic suggestions.",
        "Exploration": "Search for an idea, pick the closest existing pattern, and explore geographically linked stories plus neighboring semantic patterns on an interactive world map.",
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


def reset_clean_session_state() -> None:
    clear_caches()
    clear_live_term_state()
    keys_to_drop = [
        ADD_ENTRY_FEEDBACK_KEY,
        LAST_PROCESSED_UPLOAD_KEY,
        EXPLORATION_SELECTED_ENTRY_KEY,
    ]
    for key in keys_to_drop:
        st.session_state.pop(key, None)

    for key in list(st.session_state):
        if key.startswith(("exploration_", "edit_", "delete_")):
            st.session_state.pop(key, None)
    st.session_state["add_form_version"] = 0


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


def find_similar_terms(kind: str, query: str, output_dir: Path, *, limit: int = 5) -> tuple[list[dict], str]:
    index_path, items, model_name = get_term_artifacts(kind, output_dir)
    live_state = get_live_term_state(output_dir)[kind]
    live_items = live_state["items"]
    live_index = live_state["index"]

    if not items and not live_items:
        return [], "No indexed terms found yet."

    try:
        model, index = cached_search_assets(str(index_path), model_name)
        ranked_by_term: dict[str, dict] = {}

        for result in semantic_search(model, index, [item["text"] for item in items], query, limit=limit):
            item = items[result["position"]]
            ranked_by_term[item["normalized"]] = {
                "text": item["text"],
                "score": result["score"],
                "entry_count": item["entry_count"],
                "is_live": False,
            }

        if live_items and live_index is not None:
            for result in semantic_search(model, live_index, [item["text"] for item in live_items], query, limit=limit):
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

        enriched = sorted(ranked_by_term.values(), key=lambda row: (-row["score"], row["text"].lower()))[:limit]
        return enriched, "semantic + live updates" if live_items else "semantic"
    except EmbeddingDependencyError:
        combined_items = [dict(item, is_live=False) for item in items] + [dict(item, is_live=True) for item in live_items]
        fallback = lexical_fallback(combined_items, query, limit=limit)
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
    clean_feedback = st.session_state.pop(CLEAN_SESSION_FEEDBACK_KEY, None)

    if clean_feedback:
        st.session_state.pop("processing_confirm_clean", None)
        st.success(clean_feedback)

    st.markdown("### Source data")
    st.markdown(
        "*Upload a CSV from your computer to begin. The app validates the file and, if it is well-formed,* "
        "*rebuilds the internal dataset, the lexical indices, and both semantic indices.*"
    )

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

    with st.sidebar:
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

    st.markdown("### Current artifact status")
    display_status_badges(status)
    show_manifest_summary(status)
    st.caption(
        "Uploading a valid CSV replaces the working dataset. After that, edits happen in JSONL and stay in sync "
        "with the semantic indices automatically."
    )

    st.markdown("### End session")
    st.warning(
        "Use this to remove the processed dataset, semantic indices, manifest, and saved backups from the current "
        "processed data directory."
    )
    confirm_clean = st.checkbox(
        "I understand this will delete all generated artefacts for the current session.",
        key="processing_confirm_clean",
    )
    if st.button("Clean session", type="secondary", disabled=not confirm_clean):
        summary = clear_artifacts(current_output_dir())
        reset_clean_session_state()
        st.session_state[CLEAN_SESSION_FEEDBACK_KEY] = (
            f"Session cleaned. Deleted {summary['deleted_files']} artefact"
            f"{'' if summary['deleted_files'] == 1 else 's'} and removed {summary['deleted_dirs']} director"
            f"{'y' if summary['deleted_dirs'] == 1 else 'ies'}."
        )
        st.rerun()


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
        with st.expander(section_title):
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


def ensure_semantic_artifacts_ready(output_dir: Path) -> tuple[dict, str | None, str | None]:
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
    status, auto_refresh_message, auto_refresh_error = ensure_semantic_artifacts_ready(output_dir)
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

    with st.sidebar:
        st.markdown("### Export")
        st.caption("Download a new CSV rebuilt from the current JSONL dataset.")
        if jsonl_path.exists():
            export_entries = read_jsonl(jsonl_path)
            export_bytes = entries_to_csv_bytes(export_entries)
            export_stamp = datetime.fromtimestamp(jsonl_path.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
            st.download_button(
                "Export current version to CSV",
                data=export_bytes,
                file_name=f"entries-from-jsonl-{export_stamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("Export latest JSONL to CSV", disabled=True, use_container_width=True)
            st.caption("The export button will be available after a valid CSV upload has been processed.")


    add_tab, edit_tab, delete_tab = st.tabs(["Add entry", "Edit entry", "Delete entry"])
    with add_tab:
        render_add_tab(output_dir)
    with edit_tab:
        render_edit_tab(output_dir, entries)
    with delete_tab:
        render_delete_tab(output_dir, entries)


def render_exploration_details(marker: dict | None) -> None:
    st.markdown("### Story detail")
    if not marker:
        st.info("Click a marker on the map to inspect a story.")
        return

    entry = marker["entry"]
    fields = entry.get("fields", {})
    abstract = primary_abstract(entry)

    st.markdown(f"#### {entry_title(entry)}")
    badges = [entry["entry_id"]]
    if marker["kind"] == "original":
        badges.append("Selected pattern")
    else:
        badges.append(f"Similarity {marker['similarity']:.3f}")
    if not marker.get("has_location", True):
        badges.append("no location")
    territory = clean_text(fields.get("territory", ""))
    if territory:
        badges.append(territory)
    st.caption(" · ".join(badges))

    if abstract:
        st.markdown("**Abstract**")
        st.write(abstract)

    if marker["matched_patterns"]:
        st.markdown("**Matched patterns**")
        for item in marker["matched_patterns"][:6]:
            if marker["kind"] == "original":
                st.markdown(f"- {item['pattern']}")
            else:
                st.markdown(f"- {item['pattern']} ({item['score']:.3f})")

    with st.expander("Full entry", expanded=True):
        for field_name in CSV_COLUMNS:
            value = clean_text(fields.get(field_name, ""))
            if not value:
                continue
            st.markdown(f"**{field_name}**")
            st.write(value)


def render_exploration_map(network: dict, *, map_key: str) -> str | None:
    if folium is None or st_folium is None:
        st.error("The map dependencies are not installed. Install `folium` and `streamlit-folium` to use Exploration.")
        return None

    all_markers = network["original_markers"] + network["related_markers"]
    if not all_markers:
        return None

    primary_marker = network["original_markers"][0] if network["original_markers"] else network["related_markers"][0]
    story_map = folium.Map(
        location=list(primary_marker["coordinates"]),
        zoom_start=3,
        control_scale=True,
        prefer_canvas=True,
    )

    related_group = folium.FeatureGroup(name=f"Similar-pattern stories ({len(network['related_markers'])})", show=True)
    connection_group = folium.FeatureGroup(name="Connections", show=True)
    original_group = folium.FeatureGroup(name=f"Selected pattern stories ({len(network['original_markers'])})", show=True)

    for connection in network["connections"]:
        folium.PolyLine(
            locations=[connection["source_coordinates"], connection["target_coordinates"]],
            color=connection["color"],
            weight=3,
            opacity=0.62,
            dash_array="8 8",
        ).add_to(connection_group)

    for marker in network["related_markers"]:
        related_patterns = ", ".join(item["pattern"] for item in marker["matched_patterns"][:3])
        popup_lines = [
            f"<strong>{marker['title']}</strong>",
            f"Entry {marker['entry_id']}",
            f"Similarity {marker['similarity']:.3f}",
        ]
        if not marker.get("has_location", True):
            popup_lines.append("no location")
        if related_patterns:
            popup_lines.append(f"Patterns: {related_patterns}")
        folium.CircleMarker(
            location=list(marker["coordinates"]),
            radius=6,
            color="#1f2937",
            weight=1.5,
            fill=True,
            fill_color=marker["color"],
            fill_opacity=0.82,
            tooltip=marker["hover_title"],
            popup=folium.Popup(popup_html_for_entry(marker["entry_id"], popup_lines), max_width=360),
        ).add_to(related_group)

    related_group.add_to(story_map)
    connection_group.add_to(story_map)

    for marker in network["original_markers"]:
        popup_lines = [
            f"<strong>{marker['title']}</strong>",
            f"Entry {marker['entry_id']}",
            "Selected pattern story",
        ]
        if not marker.get("has_location", True):
            popup_lines.append("no location")
        folium.CircleMarker(
            location=list(marker["coordinates"]),
            radius=9,
            color="#ffffff",
            weight=2.5,
            fill=True,
            fill_color="#d7263d",
            fill_opacity=1.0,
            tooltip=marker["hover_title"],
            popup=folium.Popup(popup_html_for_entry(marker["entry_id"], popup_lines), max_width=320),
        ).add_to(original_group)

    original_group.add_to(story_map)

    if network["bounds"]:
        story_map.fit_bounds(network["bounds"], padding=(28, 28))

    map_state = st_folium(
        story_map,
        key=map_key,
        height=640,
        use_container_width=True,
        returned_objects=["last_object_clicked", "last_object_clicked_popup"],
    )
    clicked_entry_id = entry_id_from_map_click(
        all_markers,
        clean_text((map_state or {}).get("last_object_clicked_popup", "")),
        (map_state or {}).get("last_object_clicked"),
    )
    return clicked_entry_id


def render_exploration_page() -> None:
    output_dir = current_output_dir()
    get_live_term_state(output_dir)
    status, auto_refresh_message, auto_refresh_error = ensure_semantic_artifacts_ready(output_dir)

    st.markdown("### Explore story geography")
    st.caption(
        "Describe an idea, pick the closest indexed pattern, then inspect where matching stories sit in relation "
        "to stories tagged with semantically similar patterns."
    )

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
            "Exploration is available after the JSONL, both lexical indices, both FAISS indices, "
            "and the manifest have all been created."
        )
        return

    entries = read_jsonl(artifact_paths(output_dir)["jsonl"])
    if not entries:
        st.info("No stories are available yet.")
        return

    query = st.text_input(
        "Describe the pattern you want to explore",
        key="exploration_query",
        placeholder="Example: someone gets inside a coconut and drifts at sea",
    )

    suggestions: list[dict] = []
    search_mode = ""
    if query.strip():
        raw_suggestions, search_mode = find_similar_terms("patterns", query, output_dir, limit=12)
        suggestions = [item for item in raw_suggestions if not item.get("is_live")]

    if not query.strip():
        st.info("Start with a sentence or motif description to retrieve the closest existing patterns.")
        return

    if not suggestions:
        st.warning("No existing pattern matched that description closely enough to explore yet.")
        return

    option_map = {
        (
            f"{item['text']} · similarity {item['score']:.3f} · "
            f"{item['entry_count']} stor{'y' if item['entry_count'] == 1 else 'ies'}"
        ): item
        for item in suggestions
    }
    selected_label = st.selectbox(
        "Closest existing patterns",
        list(option_map.keys()),
        key="exploration_selected_pattern",
        help=f"Suggestions are based on {search_mode}.",
    )
    selected_pattern = option_map[selected_label]["text"]

    related_candidates, related_mode = find_similar_terms("patterns", selected_pattern, output_dir, limit=60)
    related_patterns = [
        item
        for item in related_candidates
        if normalize_text(item["text"]) != normalize_text(selected_pattern)
    ]

    st.markdown("### Similarity filter")
    st.caption("Show only related stories whose similarity score is at or above this threshold.")
    slider_col, value_col = st.columns([5, 1.2])
    with slider_col:
        threshold = st.slider(
            "Minimum similarity",
            min_value=0.00,
            max_value=1.00,
            value=0.62,
            step=0.01,
            format="%.2f",
            key="exploration_threshold",
        )
    with value_col:
        st.metric("Current", f"{threshold:.2f}")
    displayed_related_patterns = [item for item in related_patterns if float(item["score"]) >= threshold]

    network = build_exploration_network(
        entries,
        selected_pattern,
        displayed_related_patterns,
        minimum_similarity=threshold,
    )
    visible_markers = {
        marker["entry_id"]: marker for marker in network["original_markers"] + network["related_markers"]
    }

    metric_columns = st.columns(4)
    metric_columns[0].metric("Original stories", len(network["original_markers"]))
    metric_columns[1].metric("Similar-pattern stories", len(network["related_markers"]))
    metric_columns[2].metric("Similar patterns", len(displayed_related_patterns))
    metric_columns[3].metric("Connections", len(network["connections"]))

    st.caption(
        f"Selected pattern: {selected_pattern} · related patterns computed with {related_mode}. "
        f"Showing related stories at similarity {threshold:.2f} or above."
    )
    if network["missing_original_coords"]:
        st.caption(
            f"{network['missing_original_coords']} exact-pattern stor"
            f"{'y is' if network['missing_original_coords'] == 1 else 'ies are'} "
            "shown at 170 E, 0 with the label `no location` because the original coordinates are missing or malformed."
        )
    if network["missing_related_coords"]:
        st.caption(
            f"{network['missing_related_coords']} related stor"
            f"{'y is' if network['missing_related_coords'] == 1 else 'ies are'} "
            "shown around 170 E, 0 with the label `no location` because their coordinates are missing or malformed."
        )

    with st.expander("Patterns currently shown", expanded=False):
        st.markdown(f"- {selected_pattern} (selected pattern)")
        for item in displayed_related_patterns:
            st.markdown(f"- {item['text']} ({item['score']:.3f})")

    map_col, detail_col = st.columns([1.65, 1.0])
    with map_col:
        clicked_entry_id = render_exploration_map(
            network,
            map_key=f"exploration_map_{normalize_text(selected_pattern)}_{threshold:.2f}",
        )
        if clicked_entry_id and clicked_entry_id in visible_markers:
            st.session_state[EXPLORATION_SELECTED_ENTRY_KEY] = clicked_entry_id
        elif st.session_state.get(EXPLORATION_SELECTED_ENTRY_KEY) not in visible_markers:
            st.session_state.pop(EXPLORATION_SELECTED_ENTRY_KEY, None)

        if not visible_markers:
            st.warning("No stories with usable coordinates are available for this pattern and threshold.")

    with detail_col:
        selected_entry_id = st.session_state.get(EXPLORATION_SELECTED_ENTRY_KEY)
        render_exploration_details(visible_markers.get(selected_entry_id))


def main() -> None:
    apply_theme()
    init_state()
    page = render_sidebar()
    render_header(page)

    if page == "Data processing":
        render_processing_page()
    elif page == "Data management":
        render_management_page()
    else:
        render_exploration_page()


if __name__ == "__main__":
    main()
