"""Example 4: an interactive chat loop that keeps conversation history,
so follow-ups like "and how much does it weigh?" make sense to Claude.

Needs ANTHROPIC_API_KEY.  Run: python examples/04_chat.py   (Ctrl+C or 'quit' to exit)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rag import RAG

rag = RAG("data", top_k=3)
history = []

print("Ask about Orbit Bikes (type 'quit' to exit)\n")
while True:
    try:
        question = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if question.lower() in {"quit", "exit"}:
        break
    if not question:
        continue
    answer = rag.ask(question, history=history)
    print(f"\nbot> {answer}\n")
    history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer.text},
    ]
