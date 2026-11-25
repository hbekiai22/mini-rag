"""Split documents into small, overlapping chunks for retrieval."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import re


@dataclass
class Chunk:
    text: str
    source: str  # file name the chunk came from
    index: int   # position of the chunk inside that file

    @property
    def id(self) -> str:
        return f"{self.source}#{self.index}"


def split_text(text: str, max_words: int = 120, overlap: int = 20) -> List[str]:
    """Split text into chunks of roughly `max_words` words.

    Paragraphs are kept together where possible; a paragraph longer than
    `max_words` is cut into windows that overlap by `overlap` words so a
    sentence spanning a cut is still findable.
    """
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []

    def flush():
        if current:
            chunks.append(" ".join(current))
            current.clear()

    for para in paragraphs:
        words = para.split()
        if len(words) > max_words:
            flush()
            step = max_words - overlap
            for start in range(0, len(words), step):
                chunks.append(" ".join(words[start:start + max_words]))
                if start + max_words >= len(words):
                    break
            continue
        if len(current) + len(words) > max_words:
            flush()
        current.extend(words)
    flush()
    return chunks


def load_documents(folder: str, patterns: Iterable[str] = ("*.md", "*.txt")) -> List[Chunk]:
    """Read every .md/.txt file in `folder` and return its chunks."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"No such folder: {folder}")
    chunks: List[Chunk] = []
    files = sorted(p for pattern in patterns for p in root.glob(pattern))
    for path in files:
        text = path.read_text(encoding="utf-8")
        for i, piece in enumerate(split_text(text)):
            chunks.append(Chunk(text=piece, source=path.name, index=i))
    return chunks
