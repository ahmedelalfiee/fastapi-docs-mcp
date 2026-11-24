"""The MCP server: three read-only tools over the FastAPI documentation."""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from . import __version__
from .index import DocsIndex, docs_url

INSTRUCTIONS = """Search the official FastAPI documentation (snapshot, English).
Use search_docs first with a natural-language question; cite the returned urls in your answer.
Use get_page when a search result is relevant but you need the whole page.
If nothing relevant comes back, say the docs don't cover it instead of guessing."""

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

mcp = MCPServer("fastapi-docs", version=__version__, instructions=INSTRUCTIONS)


@lru_cache(maxsize=1)
def index() -> DocsIndex:
    return DocsIndex()


class Hit(BaseModel):
    page: str
    section: str
    url: str
    score: float
    text: str


class SearchResult(BaseModel):
    query: str
    retriever: str
    hits: list[Hit]


class Page(BaseModel):
    page: str
    title: str
    url: str
    words: int
    text: str


class PageRef(BaseModel):
    page: str
    title: str
    url: str


class PageList(BaseModel):
    count: int
    pages: list[PageRef]


@mcp.tool(title="Search the FastAPI docs", annotations=READ_ONLY)
def search_docs(
    query: Annotated[str, Field(min_length=3, max_length=300, description="A question or keywords, e.g. 'how to upload files'")],
    k: Annotated[int, Field(ge=1, le=10, description="How many passages to return")] = 5,
) -> SearchResult:
    """Find the most relevant passages in the FastAPI documentation for a question."""
    idx = index()
    hits = [Hit(page=c.page, section=c.section, url=docs_url(c.page), score=round(s, 4), text=c.text)
            for c, s in idx.search(query, k)]
    return SearchResult(query=query, retriever=idx.mode, hits=hits)


@mcp.tool(title="Read a FastAPI docs page", annotations=READ_ONLY)
def get_page(
    page: Annotated[str, Field(min_length=1, max_length=120, description="Page id from search_docs or list_pages, e.g. 'tutorial/request-files'")],
) -> Page:
    """Return the full text of one documentation page."""
    idx = index()
    page = page.strip().strip("/").removeprefix("https://fastapi.tiangolo.com/").strip("/")
    if page not in idx.pages:
        hint = idx.close_matches(page)
        # ToolError = an anticipated failure: the model sees this message and can recover.
        raise ToolError(f"Unknown page {page!r}." + (f" Did you mean: {', '.join(hint)}?" if hint else " Use list_pages."))
    text = idx.pages[page]
    return Page(page=page, title=idx.titles[page], url=docs_url(page), words=len(text.split()), text=text)


@mcp.tool(title="List FastAPI docs pages", annotations=READ_ONLY)
def list_pages(
    prefix: Annotated[str, Field(max_length=60, description="Optional section filter, e.g. 'tutorial/security' or 'advanced'")] = "",
) -> PageList:
    """List documentation pages, optionally only those under a section."""
    idx = index()
    refs = [PageRef(page=p, title=idx.titles[p], url=docs_url(p)) for p in idx.pages if p.startswith(prefix.strip("/"))]
    return PageList(count=len(refs), pages=refs)


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
