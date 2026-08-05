"""
Vector memory for the onboarding knowledge base.

Chunks knowledge.md, embeds each chunk once (BAAI/bge-small-en-v1.5 via fastembed),
caches vectors to disk (rebuilt only when knowledge.md changes), and retrieves the
top-k most relevant chunks for a query via cosine similarity.

Falls back gracefully: if fastembed/model is unavailable, callers can detect
`store.ready == False` and use the full knowledge base instead.
"""

import re
import json
import hashlib
from pathlib import Path

import numpy as np

MAX_CHARS = 700           # target chunk size
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _chunk_markdown(text: str) -> list[str]:
    """Split into section-aware chunks, prefixing each with its source header."""
    section = "Recykal Onboarding"
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(f"[Source: {section}]\n{buf.strip()}")
        buf = ""

    for para in paragraphs:
        p = para.strip()
        if not p:
            continue
        # Track the current file/section header for context
        m = re.match(r"##\s*File:\s*(.+)", p)
        if m:
            flush()
            section = m.group(1).strip()
            continue
        if re.match(r"^-{3,}$", p):
            # Horizontal rule — in this source content it marks a page break,
            # which is a real topic boundary often enough that blending
            # across it hurts retrieval (two unrelated topics landing in one
            # chunk dilutes the embedding so neither scores well). Flush
            # instead of silently absorbing it into the current chunk.
            flush()
            continue
        if p == "```":
            continue
        p = p.strip("`").strip()
        if not p:
            continue
        if len(buf) + len(p) + 1 > MAX_CHARS and buf:
            flush()
        buf = f"{buf}\n{p}" if buf else p
    flush()
    return [c for c in chunks if len(c) > 20]


# Shared across all VectorStore instances/rebuilds — loading the ONNX model is
# expensive (100MB+), and knowledge.md gets rebuilt at runtime now (HR uploads),
# not just once at startup. Without this, every rebuild would momentarily hold
# two full copies of the model in memory (the old instance, not yet garbage
# collected, plus the new one loading) — enough to trip an OOM kill on a small VPS.
_SHARED_MODEL = None


# onnxruntime's default thread pool and batch size scale their memory use with
# both threads and input size — on a small VPS a big knowledge-base rebuild
# (hundreds of chunks in one call) can spike well past what a single small
# request needs. threads=1 and a small EMBED_BATCH_SIZE bound peak memory
# regardless of corpus size, at the cost of a somewhat slower rebuild.
EMBED_BATCH_SIZE = 16


def _shared_model_lazy():
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        from fastembed import TextEmbedding
        _SHARED_MODEL = TextEmbedding(EMBED_MODEL, threads=1)
    return _SHARED_MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """L2-normalized embeddings for arbitrary text, via the same shared model
    used for knowledge-base retrieval. Used outside the RAG path too — e.g.
    clustering similar questions for the repeated-questions report."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vecs = np.array(
        list(_shared_model_lazy().embed(texts, batch_size=EMBED_BATCH_SIZE)),
        dtype=np.float32,
    )
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def cluster_similar_texts(texts: list[str], threshold: float = 0.83) -> list[list[int]]:
    """Greedy cosine-similarity clustering — groups indices into `texts`
    whose pairwise similarity exceeds `threshold`. Every index appears in
    exactly one cluster (singletons included); callers filter for clusters
    with 2+ members to find actual repeats. O(n^2) — fine for a week's worth
    of questions, not meant for large corpora."""
    if not texts:
        return []
    vecs = embed_texts(texts)
    sims = vecs @ vecs.T
    assigned = [False] * len(texts)
    clusters: list[list[int]] = []
    for i in range(len(texts)):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, len(texts)):
            if not assigned[j] and sims[i, j] >= threshold:
                group.append(j)
                assigned[j] = True
        clusters.append(group)
    return clusters


class VectorStore:
    def __init__(self, knowledge_path: Path, cache_path: Path | None = None):
        self.knowledge_path = Path(knowledge_path)
        self.cache_path = Path(cache_path) if cache_path else \
            self.knowledge_path.with_suffix(".vec.npz")
        self.ready = False
        self.chunks: list[str] = []
        self.vectors: np.ndarray | None = None
        try:
            self._build_or_load()
            self.ready = True
        except Exception as e:  # pragma: no cover
            print(f"⚠️  VectorStore unavailable, will fall back to full KB: {e}")

    def _file_hash(self) -> str:
        h = hashlib.sha256(self.knowledge_path.read_bytes()).hexdigest()
        return f"{h}:{MAX_CHARS}:{EMBED_MODEL}"

    def _embed(self, texts: list[str]) -> np.ndarray:
        return embed_texts(texts)

    def _build_or_load(self):
        current_hash = self._file_hash()
        if self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=True)
            if str(data.get("hash")) == current_hash:
                self.chunks = list(data["chunks"])
                self.vectors = data["vectors"]
                print(f"✓ Vector cache loaded: {len(self.chunks)} chunks")
                return

        text = self.knowledge_path.read_text(encoding="utf-8")
        self.chunks = _chunk_markdown(text)
        self.vectors = self._embed(self.chunks)
        np.savez(
            self.cache_path,
            hash=current_hash,
            chunks=np.array(self.chunks, dtype=object),
            vectors=self.vectors,
        )
        print(f"✓ Vector store built: {len(self.chunks)} chunks embedded")

    def retrieve(self, query: str, k: int = 8) -> list[str]:
        if not self.ready or self.vectors is None:
            return []
        qv = self._embed([query])[0]
        sims = self.vectors @ qv
        top = np.argsort(-sims)[:k]
        return [self.chunks[i] for i in top]
