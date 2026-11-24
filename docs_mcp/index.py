"""Document index: section chunks + BM25, optionally fused with embeddings.

Retrieval settings come from rag-eval-lab, where they were measured on a labelled set:
section chunks, top-5; embeddings raised recall@5 from 0.885 (BM25) to 0.962.
https://github.com/ahmedelalfiee/rag-eval-lab
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP = frozenset("a an and are as at be by can do does for from how i if in is it my of on or should the "
                  "this to what when where which why will with you your".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def docs_url(page: str) -> str:
    path = page.removesuffix("index").rstrip("/")
    return "https://fastapi.tiangolo.com/" + (path + "/" if path else "")


@dataclass(frozen=True)
class Chunk:
    id: str
    page: str
    title: str
    section: str
    text: str


def _title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else fallback


def chunk_sections(page: str, text: str, max_words: int = 350, overlap: int = 50) -> list[Chunk]:
    title = _title(text, page)
    parts = re.split(r"^(#{2,3}\s+.+)$", text, flags=re.M)
    sections = [(title, parts[0])] + [(parts[i].lstrip("#").strip(), parts[i] + "\n" + parts[i + 1])
                                      for i in range(1, len(parts), 2)]
    out: list[Chunk] = []
    for heading, body in sections:
        words = body.split()
        if len(words) < 15:
            continue
        step = max_words - overlap
        pieces = [body.strip()] if len(words) <= max_words else \
            [" ".join(words[i:i + max_words]) for i in range(0, len(words) - overlap, step)]
        for piece in pieces:
            out.append(Chunk(f"{page}#{len(out)}", page, title, heading, piece))
    return out


def load_pages(path: Path | None = None) -> dict[str, str]:
    src = path or resources.files("docs_mcp") / "data" / "pages.jsonl"
    with src.open(encoding="utf-8") as f:
        return {d["page"]: d["text"] for d in map(json.loads, f)}


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        docs = [tokenize(f"{c.title} {c.section} {c.text}") for c in chunks]
        self.lens = [len(d) for d in docs]
        self.avgdl = sum(self.lens) / max(1, len(docs))
        self.tf = []
        df: dict[str, int] = {}
        for d in docs:
            counts: dict[str, int] = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tf):
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) /
                    (tf[t] + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)) for t in q if t in tf)
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [(self.chunks[i], s) for s, i in scored[:k]]


class Dense:
    """OpenAI embeddings with an on-disk cache keyed by model + text hash."""

    def __init__(self, chunks: list[Chunk], model: str = "text-embedding-3-small", cache: Path | None = None):
        from openai import OpenAI
        self.client, self.model, self.chunks = OpenAI(), model, chunks
        self.cache_path = cache or Path(os.getenv("DOCS_MCP_CACHE", Path.home() / ".cache" / "fastapi-docs-mcp" / "embeddings.json"))
        self.cache: dict[str, list[float]] = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.vectors = [self._unit(v) for v in self._embed([f"{c.title} - {c.section}\n{c.text}" for c in chunks])]

    def _key(self, text: str) -> str:
        return self.model + ":" + hashlib.sha256(text.encode()).hexdigest()[:24]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self.cache]
        for i in range(0, len(missing), 128):
            batch = missing[i:i + 128]
            for t, d in zip(batch, self.client.embeddings.create(model=self.model, input=batch).data):
                self.cache[self._key(t)] = d.embedding
        if missing:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        return [self.cache[self._key(t)] for t in texts]

    @staticmethod
    def _unit(v: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = self._unit(self._embed([query])[0])
        scored = sorted(((sum(a * b for a, b in zip(q, v)), i) for i, v in enumerate(self.vectors)), reverse=True)
        return [(self.chunks[i], s) for s, i in scored[:k]]


def _dense_available() -> bool:
    """Use embeddings only when a key is set AND the optional `dense` extra is installed."""
    if not os.getenv("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


class DocsIndex:
    def __init__(self, pages: dict[str, str] | None = None, mode: str | None = None):
        self.pages = pages if pages is not None else load_pages()
        self.titles = {p: _title(t, p) for p, t in self.pages.items()}
        self.chunks = [c for p, t in self.pages.items() for c in chunk_sections(p, t)]
        self.bm25 = BM25(self.chunks)
        self.mode = mode or os.getenv("DOCS_MCP_RETRIEVER") or ("dense" if _dense_available() else "bm25")
        self.dense = Dense(self.chunks) if self.mode == "dense" else None

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        return (self.dense or self.bm25).search(query, k)

    def close_matches(self, page: str, n: int = 3) -> list[str]:
        return difflib.get_close_matches(page, list(self.pages), n=n, cutoff=0.4)
