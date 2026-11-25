# mini-rag

A small, readable **Retrieval-Augmented Generation (RAG)** project in about 250 lines of Python.

It answers questions about a folder of documents by:

1. **Chunking** each `.md` / `.txt` file into ~120-word overlapping pieces ([`rag/chunker.py`](rag/chunker.py))
2. **Retrieving** the most relevant chunks with BM25 keyword search, written in pure Python with no vector DB ([`rag/retriever.py`](rag/retriever.py))
3. **Generating** an answer with Claude, which is told to use *only* those chunks and cite them as `[1]`, `[2]` ([`rag/pipeline.py`](rag/pipeline.py))

The sample data in [`data/`](data/) is a made-up handbook for a fictional e-bike company, "Orbit Bikes". Because the company isn't real, the model can only know the answers by reading the documents, which makes it easy to see retrieval working.

```
question ──► BM25 search over chunks ──► top-k chunks ──► Claude (answer + citations)
```

## Quick start

```bash
git clone https://github.com/hbekiai22/mini-rag.git
cd mini-rag
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

```bash
python -m rag "Can I return a bike I've ridden 30 km?"
```

Retrieval alone works without an API key or any install:

```bash
python -m rag "what is the range of the cargo bike" --retrieve-only
```

## Examples

| File | What it shows | API key? |
|---|---|---|
| [`examples/01_retrieve_only.py`](examples/01_retrieve_only.py) | Load docs, build the index, look at search scores | No |
| [`examples/02_ask_claude.py`](examples/02_ask_claude.py) | Full RAG with cited answers, plus an unanswerable question | Yes |
| [`examples/03_your_own_docs.py`](examples/03_your_own_docs.py) | Point it at your own notes folder | Optional |
| [`examples/04_chat.py`](examples/04_chat.py) | Interactive chat that remembers follow-up context | Yes |

### Example output: retrieval (real output)

```
$ python examples/01_retrieve_only.py
Loaded 9 chunks from data/

Q: How long do I have to return a bike?
    3.97  returns_policy.md#0       # Returns Policy — Orbit Bikes Orbit Bikes accepts returns of unused b...
    0.45  models.md#0               # Bike Models — Orbit Bikes ## Orbit City A step-through commuter bike...

Q: What does error E07 mean?
    5.37  troubleshooting.md#0      # Troubleshooting — Orbit Bikes ## The display shows error E07 E07 mea...
    1.50  troubleshooting.md#1      ## Pedal assist cuts out over bumps This usually means a loose battery...

Q: Which bike can carry kids?
    2.24  models.md#1               ## Orbit Cargo A longtail cargo bike that carries two children or up t...
    0.45  models.md#0               # Bike Models — Orbit Bikes ## Orbit City A step-through commuter bike...
```

### Example output: full RAG (illustrative; wording will vary)

```
$ python -m rag "Can I return a bike I've ridden 30 km? Will it cost anything?"
Yes. Bikes ridden up to 50 km can still be returned within 30 days of delivery,
but a 10% restocking fee applies because the bike shows signs of riding [1].

Sources:
  [1] returns_policy.md#0 (score 6.12)
  [2] warranty.md#0 (score 1.05)
  [3] models.md#0 (score 0.61)
```

```
$ python -m rag "Do you ship to Canada?"
I don't know based on the documents. They list shipping to EU countries, the
United Kingdom, Norway, and Switzerland only [1].
```

## Try these questions

- How long does the Orbit Trail take to charge with the fast charger?
- Does the warranty cover my chain wearing out?
- My pedal assist cuts out over bumps. What should I do?
- What's included if I pay for "Ready to ride" delivery?
- Where do I find my frame serial number?

## Tests

The tests use a fake Claude client, so they run offline:

```bash
python -m unittest discover -s tests -v
```

## How it works, and ideas to extend it

- **Chunking**: paragraphs are packed together up to `max_words`; long paragraphs are cut into overlapping windows so a sentence on a boundary is still findable.
- **BM25** scores a chunk higher when it contains rare query words many times, normalised by chunk length. It's fast and needs no model, but it only matches *words*: "kids" won't match "children". Good next steps:
  - Swap in embedding search (e.g. `sentence-transformers` + cosine similarity) or combine both (hybrid search).
  - Add a re-ranking step.
  - Split by headings for Markdown-aware chunking.
- **Generation**: the system prompt tells Claude to answer only from the numbered passages, cite them, and say "I don't know" otherwise. That is the main defence against hallucination in RAG.

Model: `claude-opus-5-5` (change `MODEL` in `rag/pipeline.py`).

## License

MIT
