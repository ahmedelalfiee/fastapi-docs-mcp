"""Every test talks to the server through a real MCP client (in-process transport)."""
import os

import pytest
from mcp import Client

os.environ.pop("OPENAI_API_KEY", None)          # tests always use the offline BM25 index
from docs_mcp.server import mcp  # noqa: E402

pytestmark = pytest.mark.anyio


async def test_exposes_three_read_only_tools_with_schemas():
    async with Client(mcp) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert set(tools) == {"search_docs", "get_page", "list_pages"}
    for t in tools.values():
        assert t.annotations.read_only_hint and not t.annotations.destructive_hint
        assert t.output_schema and t.description
    assert tools["search_docs"].input_schema["properties"]["k"]["maximum"] == 10


async def test_search_returns_structured_hits_with_urls():
    async with Client(mcp) as c:
        r = await c.call_tool("search_docs", {"query": "How do I receive uploaded files?", "k": 3})
    assert not r.is_error
    hits = r.structured_content["hits"]
    assert len(hits) == 3 and hits[0]["page"] == "tutorial/request-files"
    assert hits[0]["url"] == "https://fastapi.tiangolo.com/tutorial/request-files/"
    assert r.structured_content["retriever"] == "bm25"


async def test_invalid_arguments_are_rejected_with_a_readable_error():
    async with Client(mcp) as c:
        short = await c.call_tool("search_docs", {"query": "x"})
        big_k = await c.call_tool("search_docs", {"query": "upload files", "k": 50})
    assert short.is_error and "at least 3 characters" in short.content[0].text
    assert big_k.is_error


async def test_get_page_accepts_urls_and_suggests_on_typos():
    async with Client(mcp) as c:
        ok = await c.call_tool("get_page", {"page": "https://fastapi.tiangolo.com/tutorial/cors/"})
        typo = await c.call_tool("get_page", {"page": "tutorial/request-file"})
    assert ok.structured_content["title"] == "CORS (Cross-Origin Resource Sharing)"
    assert typo.is_error and "tutorial/request-files" in typo.content[0].text


async def test_list_pages_filters_by_prefix():
    async with Client(mcp) as c:
        r = await c.call_tool("list_pages", {"prefix": "tutorial/security"})
    pages = [p["page"] for p in r.structured_content["pages"]]
    assert r.structured_content["count"] == len(pages) >= 5
    assert all(p.startswith("tutorial/security") for p in pages)


async def test_server_sends_usage_instructions():
    async with Client(mcp) as c:
        assert "search_docs" in (c.instructions or "")
        assert c.server_info.name == "fastapi-docs"


async def test_runs_over_stdio_like_a_desktop_client():
    """Launch the real entry point as a subprocess and talk to it over stdin/stdout."""
    import sys
    from pathlib import Path

    from mcp import StdioServerParameters

    root = str(Path(__file__).resolve().parent.parent)
    params = StdioServerParameters(command=sys.executable, args=["-m", "docs_mcp"], cwd=root,
                                   env={"PYTHONPATH": root, "PATH": os.environ.get("PATH", "")})
    async with Client(params) as c:
        r = await c.call_tool("search_docs", {"query": "background tasks after response", "k": 1})
    assert r.structured_content["hits"][0]["page"] == "tutorial/background-tasks"


def test_key_without_openai_package_falls_back_to_bm25(monkeypatch):
    import builtins

    from docs_mcp import index as ix
    real_import = builtins.__import__

    def no_openai(name, *a, **k):
        if name == "openai":
            raise ImportError
        return real_import(name, *a, **k)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("DOCS_MCP_RETRIEVER", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_openai)
    assert ix.DocsIndex(pages={"p": "# P\n\n" + "words " * 40}).mode == "bm25"
