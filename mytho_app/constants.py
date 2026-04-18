from __future__ import annotations

from pathlib import Path

APP_TITLE = "Oral Literature Workbench"
DEFAULT_OUTPUT_DIR = Path("data/processed")

JSONL_FILENAME = "entries.jsonl"
MANIFEST_FILENAME = "manifest.json"
PATTERN_INDEX_FILENAME = "patterns_index.json"
KEYWORD_INDEX_FILENAME = "keywords_index.json"
PATTERN_FAISS_FILENAME = "patterns.faiss"
KEYWORD_FAISS_FILENAME = "keywords.faiss"

PATTERN_FIELD = "Motifs (Eng)"
KEYWORD_FIELD = "Keywords (Eng)"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CSV_COLUMNS = [
    "Entered by",
    "Source first or second hand",
    "Source",
    "pages",
    "Other source",
    "URL ?",
    "territory",
    "lg group",
    "original language",
    "lg of publication",
    "bilingual?",
    "storyteller",
    "date of recording",
    "place of recording",
    "space coord",
    "editor",
    "translator",
    "Story title (Eng)",
    "Story title (French)",
    "Story title (other)",
    "1-sentence summary",
    "Abstract (Eng)",
    "Abstract (Fr)",
    KEYWORD_FIELD,
    PATTERN_FIELD,
    "proposition de nouveaux motifs",
    "species",
    "non-human",
    "placenames",
    "named characters",
    "external link",
    "description of link",
    "Connection to other stories",
    "Megamotifs",
    "Thème",
    "Conte type",
    "Autres infos données dans le texte, pour la fiche conte",
    "ATU conte-type(AI ?)",
    "ATU motifs (AI?)",
]

TITLE_FIELDS = [
    "Story title (Eng)",
    "Story title (French)",
    "Story title (other)",
]

ABSTRACT_FIELDS = [
    "1-sentence summary",
    "Abstract (Eng)",
    "Abstract (Fr)",
]

LONG_TEXT_FIELDS = {
    "Other source",
    "URL ?",
    "1-sentence summary",
    "Abstract (Eng)",
    "Abstract (Fr)",
    "proposition de nouveaux motifs",
    "external link",
    "description of link",
    "Connection to other stories",
    "Megamotifs",
    "Thème",
    "Conte type",
    "Autres infos données dans le texte, pour la fiche conte",
    "ATU motifs (AI?)",
}

FIELD_SECTIONS = [
    (
        "Titles and Summaries",
        [
            "Story title (Eng)",
            "Story title (French)",
            "Story title (other)",
            "1-sentence summary",
            "Abstract (Eng)",
            "Abstract (Fr)",
        ],
    ),
    (
        "Sources and Context",
        [
            "Entered by",
            "Source first or second hand",
            "Source",
            "pages",
            "Other source",
            "URL ?",
            "editor",
            "translator",
            "external link",
            "description of link",
        ],
    ),
    (
        "Place, Language, and People",
        [
            "territory",
            "lg group",
            "original language",
            "lg of publication",
            "bilingual?",
            "storyteller",
            "date of recording",
            "place of recording",
            "space coord",
            "species",
            "non-human",
            "placenames",
            "named characters",
        ],
    ),
    (
        "Classification and Notes",
        [
            "proposition de nouveaux motifs",
            "Connection to other stories",
            "Megamotifs",
            "Thème",
            "Conte type",
            "Autres infos données dans le texte, pour la fiche conte",
            "ATU conte-type(AI ?)",
            "ATU motifs (AI?)",
        ],
    ),
]

PREVIEW_FIELDS = [
    "Story title (Eng)",
    "Story title (French)",
    "Story title (other)",
    "1-sentence summary",
    "territory",
    "lg group",
    "original language",
    "Source",
]

