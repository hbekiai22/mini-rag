# mini-rag

[![tests](https://github.com/hbekiai22/mini-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/hbekiai22/mini-rag/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A compact, dependency-light implementation of **Retrieval-Augmented Generation (RAG)** in Python. It answers questions about a collection of documents and **cites the passages it used**. When the documents don't contain the answer, it says so instead of guessing.

The codebase is about 250 lines and has no vector database or embedding model, so each stage of the pipeline is easy to read, test and replace.

---

## Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Examples](#examples)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Design decisions and limitations](#design-decisions-and-limitations)
- [Roadmap](#roadmap)
- [License](#license)

## How it works

```mermaid
flowchart LR
    A[Documents<br/>.md / .txt] --> B[Chunker<br/>~120-word overlapping chunks]
    B --> C[BM25 index]
    Q[User question] --> C
    C -->|top-k chunks| D[Prompt builder<br/>numbered context passages]
    D --> E[Claude]
    E --> F[Answer with citations<br/>e.g. &quot;... 30 days [1]&quot;]
```

| Stage | Module | Responsibility |
|---|---|---|
| Ingestion | [`rag/chunker.py`](rag/chunker.py) | Loads files and packs paragraphs into overlapping chunks so content near a boundary stays retrievable |
| Retrieval | [`rag/retriever.py`](rag/retriever.py) | Okapi BM25 ranking with tokenisation, stop-word removal and light stemming |
| Generation | [`rag/pipeline.py`](rag/pipeline.py) | Builds a grounded prompt, calls Claude, and handles empty results and model refusals |

The bundled dataset in [`data/`](data/) is a handbook for **Orbit Bikes**, a fictional e-bike company. Because the company doesn't exist, the model can only answer correctly by reading the retrieved passages, so the demo shows retrieval actually doing its job.

## Quick start

**Requirements:** Python 3.9+ and an [Anthropic API key](https://console.anthropic.com/) for answer generation. Retrieval works without a key.

```bash
git clone https://github.com/hbekiai22/mini-rag.git
cd mini-rag
python -m venv .venv && source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY="your-key"
```

## Usage

### Command line

```bash
mini-rag "Can I return a bike I've ridden 30 km?"
```

| Option | Default | Description |
|---|---|---|
| `--docs PATH` | `data` | Folder of `.md` / `.txt` files to index |
| `--k N` | `3` | Number of chunks passed to the model |
| `--retrieve-only` | off | Print retrieved chunks and scores without calling the model |

`python -m rag ...` works the same way without installing the package.

### Python API

```python
from rag import RAG

rag = RAG("data", top_k=3)
answer = rag.ask("Does the warranty cover worn-out chains?")

print(answer.text)        # grounded answer with [n] citations
for chunk, score in answer.sources:
    print(chunk.id, round(score, 2))
```

## Examples

| Script | Demonstrates | API key |
|---|---|:---:|
| [`01_retrieve_only.py`](examples/01_retrieve_only.py) | Building the index and inspecting BM25 scores | — |
| [`02_ask_claude.py`](examples/02_ask_claude.py) | End-to-end RAG, including an unanswerable question | ✓ |
| [`03_your_own_docs.py`](examples/03_your_own_docs.py) | Running the pipeline on your own folder of notes | optional |
| [`04_chat.py`](examples/04_chat.py) | Multi-turn chat with conversation history | ✓ |

**Retrieval** (actual output):

```
$ python examples/01_retrieve_only.py
Loaded 9 chunks from data/

Q: How long do I have to return a bike?
    3.97  returns_policy.md#0       # Returns Policy — Orbit Bikes Orbit Bikes accepts returns of unused b...
    0.45  models.md#0               # Bike Models — Orbit Bikes ## Orbit City A step-through commuter bike...

Q: What does error E07 mean?
    5.37  troubleshooting.md#0      # Troubleshooting — Orbit Bikes ## The display shows error E07 E07 mea...
    1.50  troubleshooting.md#1      ## Pedal assist cuts out over bumps This usually means a loose battery...
```

**Generation** (representative output; exact wording varies between runs):

```
$ mini-rag "Can I return a bike I've ridden 30 km? Will it cost anything?"
Yes. Bikes ridden up to 50 km can still be returned within 30 days of delivery,
but a 10% restocking fee applies because the bike shows signs of riding [1].

Sources:
  [1] returns_policy.md#0 (score 6.12)
  [2] warranty.md#0 (score 1.05)
  [3] models.md#0 (score 0.61)
```

```
$ mini-rag "Do you ship to Canada?"
I don't know based on the documents. They list shipping to EU countries, the
United Kingdom, Norway, and Switzerland only [1].
```

## Project structure

```
mini-rag/
├── rag/
│   ├── chunker.py        # document loading and chunking
│   ├── retriever.py      # BM25 retriever
│   ├── pipeline.py       # prompt construction and model call
│   └── __main__.py       # command-line interface
├── data/                 # sample knowledge base (fictional)
├── examples/             # runnable usage examples
├── tests/                # offline unit tests (fake model client)
├── .github/workflows/    # CI: tests on Python 3.9–3.13
└── pyproject.toml
```

## Testing

```bash
python -m unittest discover -s tests -v
```

The suite covers chunking, ranking quality on the sample data, prompt construction, conversation history, the no-results path and refusal handling. The model client is injected, so tests run offline and cost nothing. CI runs them on every push across Python 3.9–3.13.

## Design decisions and limitations

- **BM25 instead of embeddings.** Lexical search is fast, deterministic, needs no model download and is easy to inspect. The trade-off is that it matches words, not meaning: *"kids"* won't match *"children"*.
- **Grounding in the prompt.** The system prompt restricts the model to the numbered passages, requires citations and asks it to admit when the answer isn't present. This is the main safeguard against hallucination.
- **Dependency injection.** `RAG(client=...)` accepts any object with the same interface, which keeps the pipeline testable and makes it easy to swap the model.
- **In-memory index.** The index is rebuilt at start-up, which suits small collections of up to a few thousand chunks.

## Roadmap

- [ ] Embedding-based and hybrid (BM25 + vector) retrieval
- [ ] Re-ranking of retrieved candidates
- [ ] Markdown-aware chunking by heading
- [ ] PDF ingestion
- [ ] Evaluation set measuring retrieval recall and answer faithfulness

## License

Released under the [MIT License](LICENSE).
