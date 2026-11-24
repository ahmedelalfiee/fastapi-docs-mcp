"""The search_docs tool must keep the retrieval quality measured in rag-eval-lab.

Same 52 labelled questions; offline BM25 baseline there: recall@5 = 0.885, MRR = 0.807.
"""
import json
import os
from pathlib import Path

os.environ.pop("OPENAI_API_KEY", None)
from docs_mcp.server import search_docs  # noqa: E402

QUESTIONS = [json.loads(x) for x in (Path(__file__).parent.parent / "eval" / "questions.jsonl").read_text().splitlines()]


def test_search_docs_recall_and_mrr():
    ranks = []
    for q in (q for q in QUESTIONS if q["answerable"]):
        pages = [h.page for h in search_docs(q["question"], k=10).hits]
        ranks.append(next((i for i, p in enumerate(pages, 1) if p in q["gold_pages"]), None))
    recall5 = sum(1 for r in ranks if r and r <= 5) / len(ranks)
    mrr = sum(1 / r for r in ranks if r) / len(ranks)
    assert recall5 >= 0.85 and mrr >= 0.78, (recall5, mrr)
