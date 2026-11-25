"""Retrieve relevant chunks, then ask Claude to answer using only those chunks."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .chunker import Chunk, load_documents
from .retriever import BM25Retriever

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You answer questions using only the numbered context passages you are given.
- Cite the passages you used in square brackets, like [1] or [2][3].
- If the passages do not contain the answer, say you don't know based on the documents. Do not guess.
- Keep answers short: a few sentences or a brief list."""


@dataclass
class Answer:
    question: str
    text: str
    sources: List[Tuple[Chunk, float]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [self.text, "", "Sources:"]
        for n, (chunk, score) in enumerate(self.sources, 1):
            lines.append(f"  [{n}] {chunk.id} (score {score:.2f})")
        return "\n".join(lines)


def build_prompt(question: str, hits: List[Tuple[Chunk, float]]) -> str:
    context = "\n\n".join(
        f"[{n}] (from {chunk.source})\n{chunk.text}" for n, (chunk, _) in enumerate(hits, 1)
    )
    return f"<context>\n{context}\n</context>\n\nQuestion: {question}"


class RAG:
    def __init__(self, docs_folder: str, top_k: int = 3, client=None, model: str = MODEL):
        self.retriever = BM25Retriever(load_documents(docs_folder))
        self.top_k = top_k
        self.model = model
        self._client = client  # injectable for tests

    @property
    def client(self):
        if self._client is None:
            import anthropic  # imported lazily so retrieval works without the SDK

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def retrieve(self, question: str) -> List[Tuple[Chunk, float]]:
        return self.retriever.search(question, k=self.top_k)

    def ask(self, question: str, history: Optional[List[dict]] = None) -> Answer:
        hits = self.retrieve(question)
        if not hits:
            return Answer(question, "I couldn't find anything about that in the documents.", [])

        messages = list(history or []) + [
            {"role": "user", "content": build_prompt(question, hits)}
        ]
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages,
            output_config={"effort": "low"},  # short grounded Q&A doesn't need deep reasoning
            # If the model declines, the API retries on a fallback model automatically.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return Answer(question, "The model declined to answer this question.", hits)

        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return Answer(question, text, hits)
