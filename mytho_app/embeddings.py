from __future__ import annotations

from pathlib import Path

import numpy as np


class EmbeddingDependencyError(RuntimeError):
    """Raised when sentence-transformers or FAISS is unavailable."""


NETWORK_HINTS = (
    "nodename nor servname",
    "Temporary failure in name resolution",
    "Name or service not known",
    "client has been closed",
    "Connection error",
)


def require_embedding_dependencies():
    try:
        import faiss  # type: ignore
    except ModuleNotFoundError as exc:
        raise EmbeddingDependencyError(
            "FAISS is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ModuleNotFoundError as exc:
        raise EmbeddingDependencyError(
            "sentence-transformers is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return SentenceTransformer, faiss


def load_sentence_transformer(model_name: str):
    SentenceTransformer, _ = require_embedding_dependencies()
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - depends on runtime network state
        message = str(exc)
        if any(hint in message for hint in NETWORK_HINTS):
            try:
                return SentenceTransformer(model_name, local_files_only=True)
            except Exception as cached_exc:
                raise EmbeddingDependencyError(
                    "The embedding model is not available locally yet. "
                    "Run the vector build once with network access so it can be cached."
                ) from cached_exc
        raise


def encode_texts(model, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype="float32")


def create_in_memory_index(dimension: int):
    _, faiss = require_embedding_dependencies()
    return faiss.IndexFlatIP(dimension)


def add_embeddings(index, embeddings: np.ndarray) -> None:
    matrix = np.asarray(embeddings, dtype="float32")
    if matrix.size == 0:
        return
    index.add(matrix)


def build_faiss_index(texts: list[str], model_name: str, output_path: Path) -> dict:
    if not texts:
        raise ValueError("Cannot build a vector index from an empty text list.")

    _, faiss = require_embedding_dependencies()
    model = load_sentence_transformer(model_name)
    matrix = encode_texts(model, texts)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(output_path))
    return {
        "count": int(matrix.shape[0]),
        "dimension": int(matrix.shape[1]),
        "model_name": model_name,
    }


def load_search_assets(index_path: Path, model_name: str):
    _, faiss = require_embedding_dependencies()
    model = load_sentence_transformer(model_name)
    index = faiss.read_index(str(index_path))
    return model, index


def semantic_search(model, index, texts: list[str], query: str, limit: int = 5) -> list[dict]:
    if not query.strip() or not texts:
        return []

    query_vector = encode_texts(model, [query])
    distances, indices = index.search(query_vector, min(limit, len(texts)))

    results: list[dict] = []
    for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
        if idx < 0 or idx >= len(texts):
            continue
        results.append({"text": texts[idx], "score": float(score), "position": int(idx)})
    return results
