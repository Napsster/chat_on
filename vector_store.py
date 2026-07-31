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
        if p in ("---", "```"):
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


def _shared_model_lazy():
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        from fastembed import TextEmbedding
        _SHARED_MODEL = TextEmbedding(EMBED_MODEL)
    return _SHARED_MODEL


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

    def _model_lazy(self):
        return _shared_model_lazy()

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._model_lazy().embed(texts)), dtype=np.float32)
        # L2-normalize so cosine similarity == dot product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

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
