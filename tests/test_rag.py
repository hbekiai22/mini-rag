import sys
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import RAG, BM25Retriever, load_documents, split_text
from rag.retriever import tokenize

DATA = str(ROOT / "data")


class FakeClient:
    """Stands in for anthropic.Anthropic() so tests run offline."""

    def __init__(self, reply="Returns are accepted within 30 days [1].", stop_reason="end_turn"):
        self.calls = []
        reply_block = SimpleNamespace(type="text", text=reply)
        response = SimpleNamespace(content=[reply_block], stop_reason=stop_reason)

        def create(**kwargs):
            self.calls.append(kwargs)
            return response

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


class ChunkerTests(unittest.TestCase):
    def test_short_paragraphs_are_merged(self):
        self.assertEqual(split_text("one two\n\nthree four", max_words=10, overlap=2), ["one two three four"])

    def test_long_paragraph_is_windowed_with_overlap(self):
        text = " ".join(f"w{i}" for i in range(25))
        chunks = split_text(text, max_words=10, overlap=2)
        self.assertEqual(chunks[0].split()[-2:], chunks[1].split()[:2])
        self.assertEqual(chunks[-1].split()[-1], "w24")

    def test_loads_sample_data(self):
        chunks = load_documents(DATA)
        self.assertGreater(len(chunks), 5)
        self.assertIn("returns_policy.md", {c.source for c in chunks})

    def test_missing_folder(self):
        with self.assertRaises(FileNotFoundError):
            load_documents("/no/such/folder")


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = BM25Retriever(load_documents(DATA))

    def top_source(self, query):
        return self.retriever.search(query, k=1)[0][0].source

    def test_tokenize_stems_and_drops_stopwords(self):
        self.assertEqual(tokenize("The returns of batteries"), ["return", "battery"])

    def test_finds_the_right_document(self):
        self.assertEqual(self.top_source("how many days to return a bike"), "returns_policy.md")
        self.assertEqual(self.top_source("error E07 on display"), "troubleshooting.md")
        self.assertEqual(self.top_source("battery capacity warranty charge cycles"), "warranty.md")
        self.assertEqual(self.top_source("express delivery cost"), "shipping.md")

    def test_no_match_returns_empty(self):
        self.assertEqual(self.retriever.search("xylophone zebra"), [])


class PipelineTests(unittest.TestCase):
    def test_prompt_contains_context_and_question(self):
        client = FakeClient()
        rag = RAG(DATA, top_k=2, client=client)
        answer = rag.ask("How long do I have to return a bike?")

        self.assertEqual(answer.text, "Returns are accepted within 30 days [1].")
        self.assertEqual(len(answer.sources), 2)
        sent = client.calls[0]["messages"][-1]["content"]
        self.assertIn("<context>", sent)
        self.assertIn("30 days", sent)
        self.assertIn("Question: How long do I have to return a bike?", sent)

    def test_skips_api_when_nothing_retrieved(self):
        client = FakeClient()
        answer = RAG(DATA, client=client).ask("xylophone zebra")
        self.assertEqual(client.calls, [])
        self.assertIn("couldn't find", answer.text)

    def test_handles_refusal(self):
        client = FakeClient(reply="", stop_reason="refusal")
        answer = RAG(DATA, client=client).ask("battery warranty")
        self.assertIn("declined", answer.text)

    def test_history_is_sent(self):
        client = FakeClient()
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        RAG(DATA, client=client).ask("battery warranty", history=history)
        self.assertEqual(client.calls[0]["messages"][:2], history)


if __name__ == "__main__":
    unittest.main()
