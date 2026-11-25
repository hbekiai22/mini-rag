"""Example 2: full RAG — retrieve chunks, then have Claude answer with citations.

Needs:  pip install anthropic   and   export ANTHROPIC_API_KEY=...
Run:    python examples/02_ask_claude.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rag import RAG

rag = RAG("data", top_k=3)

for question in [
    "Can I return a bike I've ridden 30 km? Will it cost anything?",
    "How long does the Orbit Trail take to charge with the fast charger?",
    "Does the warranty cover my chain wearing out?",
    "Do you ship to Canada?",  # not in the docs -> the model should say it doesn't know
]:
    print("Q:", question)
    print(rag.ask(question))
    print("-" * 60)
