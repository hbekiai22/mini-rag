"""Command line: python -m rag "your question" [--docs data] [--k 3] [--retrieve-only]"""

import argparse

from .pipeline import RAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about a folder of documents.")
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("--docs", default="data", help="folder of .md/.txt files (default: data)")
    parser.add_argument("--k", type=int, default=3, help="number of chunks to retrieve")
    parser.add_argument("--retrieve-only", action="store_true",
                        help="show retrieved chunks without calling Claude (no API key needed)")
    args = parser.parse_args()

    rag = RAG(args.docs, top_k=args.k)
    if args.retrieve_only:
        hits = rag.retrieve(args.question)
        if not hits:
            print("No matching chunks.")
        for n, (chunk, score) in enumerate(hits, 1):
            print(f"[{n}] {chunk.id}  score={score:.2f}\n    {chunk.text[:200]}...\n")
        return
    print(rag.ask(args.question))


if __name__ == "__main__":
    main()
