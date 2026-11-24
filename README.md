# fastapi-docs-mcp

[![ci](https://github.com/ahmedelalfiee/fastapi-docs-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmedelalfiee/fastapi-docs-mcp/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)
![MCP](https://img.shields.io/badge/MCP-Python%20SDK%202.x-black)

An [MCP](https://modelcontextprotocol.io) server that lets AI assistants such as Claude Code,
Claude Desktop and Cursor **search the FastAPI documentation** while they work, and cite it.

The retrieval behind it isn't guessed: it comes from [rag-eval-lab](https://github.com/ahmedelalfiee/rag-eval-lab),
where the same chunking and retrievers were measured on 52 labelled questions. Offline it uses
BM25 (recall@5 **0.885**); with an OpenAI key it switches to embeddings (recall@5 **0.962**).

## In Claude Code

```text
> Using the FastAPI docs, how do I force every request to use HTTPS?

  fastapi-docs - search_docs(query: "force HTTPS redirect all requests") (MCP)
  → retriever: dense · top hit: advanced/middleware

● Use HTTPSRedirectMiddleware ... app.add_middleware(HTTPSRedirectMiddleware)
  Source: https://fastapi.tiangolo.com/advanced/middleware/
```

This is a question keyword search gets wrong (its top hits are middleware basics, SQL databases
and testing); with embeddings the right page comes back first.

## Tools

| Tool | Input | Returns |
|---|---|---|
| `search_docs` | `query` (3–300 chars), `k` (1–10, default 5) | Ranked passages with page, section, URL and score |
| `get_page` | `page` id or full docs URL | The whole page as text, with title and URL |
| `list_pages` | optional `prefix`, e.g. `tutorial/security` | Page ids, titles and URLs |

All three are **read-only** and declare it (`readOnlyHint`, `idempotentHint`, no open-world
access), return **typed structured output** (pydantic models, so clients get an output schema),
and the server sends **usage instructions** telling the model to search first, cite URLs, and
say when the docs don't cover something.

## Use it

**Claude Code**

```bash
# keyword search, no key needed
claude mcp add fastapi-docs -- uvx --from git+https://github.com/ahmedelalfiee/fastapi-docs-mcp fastapi-docs-mcp

# embedding search (better on questions worded differently from the docs)
claude mcp add fastapi-docs -e OPENAI_API_KEY=sk-... -- uvx --from "fastapi-docs-mcp[dense] @ git+https://github.com/ahmedelalfiee/fastapi-docs-mcp" fastapi-docs-mcp
```

**Claude Desktop / Cursor** (`claude_desktop_config.json` or `.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "fastapi-docs": {
      "command": "uvx",
      "args": ["--from", "fastapi-docs-mcp[dense] @ git+https://github.com/ahmedelalfiee/fastapi-docs-mcp", "fastapi-docs-mcp"],
      "env": { "OPENAI_API_KEY": "optional: enables embedding search" }
    }
  }
}
```

**From a clone**

```bash
pip install -e ".[dev]"        # add ",dense" for embedding search
python -m docs_mcp             # stdio server
python -m pytest -q            # 8 tests, no keys needed
```

Then ask your assistant something like *"Using the FastAPI docs, how do I cap the size of request
bodies?"* and it will call `search_docs` and answer with links.

## Configuration

| Variable | Effect |
|---|---|
| `OPENAI_API_KEY` | If set and the `dense` extra is installed, search uses `text-embedding-3-small`; otherwise it falls back to BM25 instead of failing |
| `DOCS_MCP_RETRIEVER` | Force `bm25` or `dense` |
| `DOCS_MCP_CACHE` | Where embeddings are cached (default `~/.cache/fastapi-docs-mcp/`). The docs are embedded once (< $0.01) |

## Design decisions

**1. Return passages, not answers.** There is no `answer` tool. The client is already an LLM; a
server that calls a second LLM would add cost and latency and hide the evidence. Returning
ranked passages with URLs lets the assistant read, combine and cite them itself.

**2. Errors the model can recover from.** Bad arguments fail schema validation with a readable
message, and an unknown page raises a `ToolError` that suggests the closest page ids (a typo like
`tutorial/request-file` returns *"Did you mean: tutorial/request-files…"*). Unexpected crashes
stay generic so internals don't leak into the conversation.

**3. Works with zero setup.** The docs snapshot ships inside the package and BM25 needs no key,
so `uvx` gets a working server in one command. Embeddings are an opt-in upgrade, measured to help
most on questions worded differently from the docs (e.g. *"force every request to use HTTPS"*,
where the docs say `HTTPSRedirectMiddleware`).

**4. Tested the way clients use it.** Every test goes through a real MCP client session, one
launches the actual entry point over **stdio** as a subprocess, and a quality test re-runs the
labelled questions through `search_docs` and fails CI if recall@5 drops below 0.85.

## Limitations

- The docs are a **snapshot** (FastAPI commit `05e6df7`, Nov 2026). Rebuild `docs_mcp/data/pages.jsonl`
  with rag-eval-lab's `scripts/build_corpus.py` to refresh it.
- English docs only; code examples included by reference in the docs (`{* ... *}`) are not indexed.
- The embedding cache is a JSON file, fine for ~1,000 chunks, not for large corpora.

## License

MIT. The bundled documentation is © Sebastián Ramírez under the MIT License (see `docs_mcp/data/NOTICE.md`).
