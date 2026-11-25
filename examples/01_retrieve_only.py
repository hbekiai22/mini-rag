"""Example 1: see what the retriever finds. No API key needed.

Run from the repo root:  python examples/01_retrieve_only.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rag import BM25Retriever, load_documents

chunks = load_documents("data")
print(f"Loaded {len(chunks)} chunks from data/\n")
retriever = BM25Retriever(chunks)

for question in [
    "How long do I have to return a bike?",
    "What does error E07 mean?",
    "Which bike can carry kids?",
]:
    print(f"Q: {question}")
    for chunk, score in retriever.search(question, k=2):
        print(f"   {score:5.2f}  {chunk.id:24}  {chunk.text[:70]}...")
    print()
