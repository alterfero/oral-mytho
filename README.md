# oral-mytho

Local Streamlit app for processing a story CSV into JSONL, lexical indices, and semantic search artifacts for motifs and keywords.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Workflow

1. Open the `Data processing` page.
2. Import the CSV into `data/processed/entries.jsonl`.
3. Build the pattern and keyword indices plus both FAISS stores.
4. Switch to `Data management` to add, edit, or delete entries.

The app keeps automatic backups of `entries.jsonl` before overwriting it.
