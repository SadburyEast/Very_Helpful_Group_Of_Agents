"""
Cross-session long-term memory.

Deliberately does NOT use Ollama for embeddings. The 9B generation model
already occupies most of the 3070's 8GB VRAM; adding a second model into
Ollama's rotation means either GPU contention during concurrent calls or
Ollama swapping models in and out of VRAM between them -- a real latency
cost for something as cheap as embedding a short query. A small CPU-only
sentence-transformers model sidesteps both: embeddings are fast enough
on CPU that this is a non-issue, and it keeps 100% of VRAM free for
generation, which is the thing that actually needs it.

Chroma is used as a local persistent store (no server) so recall
actually survives between separate `python main.py ...` invocations --
that's the entire point of "cross-session," as opposed to MemorySaver's
in-process-only checkpointing.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings
from state import RecalledSession

_embedder: SentenceTransformer | None = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        # CPU explicitly -- see module docstring.
        _embedder = SentenceTransformer(settings.embedding_model, device="cpu")
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        os.makedirs(settings.memory_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=settings.memory_dir)
        _collection = client.get_or_create_collection(
            name="research_sessions",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def recall(task: str) -> list[RecalledSession]:
    """
    Embed `task` and return past sessions above the similarity threshold,
    most similar first. Empty list on a cold store or no sufficiently
    close match -- this is the common case and not an error.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    embedding = _get_embedder().encode(task).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(settings.recall_top_k, collection.count()),
    )

    recalled: list[RecalledSession] = []
    ids = results.get("ids", [[]])[0]
    if not ids:
        return []

    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for distance, meta in zip(distances, metadatas):
        # Chroma cosine space returns distance; similarity = 1 - distance.
        similarity = 1 - distance
        if similarity < settings.recall_similarity_threshold:
            continue
        recalled.append(
            RecalledSession(
                task=meta["task"],
                final_answer=meta["final_answer"],
                timestamp=meta["timestamp"],
                similarity=round(similarity, 3),
            )
        )

    return recalled


def persist(task: str, final_answer: str) -> None:
    """Write a completed session to memory for future recall. Best-effort."""
    if not final_answer.strip():
        # Don't pollute memory with failed/empty runs.
        return

    collection = _get_collection()
    embedding = _get_embedder().encode(task).tolist()
    session_id = f"session-{datetime.now(timezone.utc).timestamp()}"

    collection.add(
        ids=[session_id],
        embeddings=[embedding],
        metadatas=[
            {
                "task": task,
                "final_answer": final_answer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
