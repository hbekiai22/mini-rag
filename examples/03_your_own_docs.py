"""Example 3: point the pipeline at your own folder of .md / .txt files.

Run:  python examples/03_your_own_docs.py path/to/your/notes "your question"
Add --retrieve-only to skip the Claude call.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rag import RAG

if len(sys.argv) < 3:
    sys.exit(__doc__)

folder, question = sys.argv[1], sys.argv[2]
rag = RAG(folder, top_k=4)

if "--retrieve-only" in sys.argv:
    for chunk, score in rag.retrieve(question):
        print(f"{score:5.2f}  {chunk.id}\n       {chunk.text[:150]}...\n")
else:
    print(rag.ask(question))
