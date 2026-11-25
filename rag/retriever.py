"""BM25 keyword retriever in pure Python (no vector database needed)."""

import math
import re
from collections import Counter
from typing import List, Tuple

from .chunker import Chunk

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "if", "in", "is", "it", "my", "of", "on", "or", "that",
    "the", "this", "to", "was", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your",
}


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS]


def stem(word: str) -> str:
    """Very small suffix stripper so 'returns' matches 'return'."""
    for suffix in ("ing", "ies", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


class BM25Retriever:
    """Classic Okapi BM25 ranking over a list of chunks."""

    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("Cannot build a retriever with no chunks")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c.text) for c in chunks]
        self.doc_freqs = [Counter(t) for t in self.doc_tokens]
        self.avg_len = sum(len(t) for t in self.doc_tokens) / len(self.doc_tokens)

        df: Counter = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def score(self, query_tokens: List[str], i: int) -> float:
        freqs = self.doc_freqs[i]
        length = len(self.doc_tokens[i])
        total = 0.0
        for term in query_tokens:
            if term not in freqs:
                continue
            tf = freqs[term]
            denom = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
            total += self.idf[term] * tf * (self.k1 + 1) / denom
        return total

    def search(self, query: str, k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top-k chunks for `query`, best first. Zero-score chunks are dropped."""
        q = tokenize(query)
        scored = [(self.chunks[i], self.score(q, i)) for i in range(len(self.chunks))]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: s[1], reverse=True)
        return scored[:k]
