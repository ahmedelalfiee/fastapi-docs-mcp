"""Rebuild docs_mcp/data/pages.jsonl from a FastAPI docs checkout.

    python scripts/refresh_docs.py path/to/fastapi/docs/en/docs <commit-sha>

Same cleaning rules as rag-eval-lab (where retrieval on this corpus was evaluated), so a refresh
keeps the measured setup. Used by the monthly `refresh-docs` workflow.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs_mcp" / "data" / "pages.jsonl"
NOTICE = ROOT / "docs_mcp" / "data" / "NOTICE.md"

EXCLUDE = re.compile(
    r"^(_llm-test|agent-banner|translation-banner|release-notes|fastapi-people|newsletter|"
    r"external-links|management|translations|help-fastapi|contributing|about/|reference/)"
)
_ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}")
_INCLUDE = re.compile(r"^\{\*.*\*\}\s*$", re.M)
_ADMONITION = re.compile(r"^///.*$", re.M)
_TAG = re.compile(r"</?(dfn|abbr|span|div|img|a|br|details|summary)[^>]*>")
_SPACES = re.compile(r"\n{3,}")


def clean(md: str) -> str:
    md = _INCLUDE.sub("", md)
    md = _ADMONITION.sub("", md)
    md = _TAG.sub("", md)
    md = "\n".join(_ANCHOR.sub("", line) for line in md.splitlines())
    return _SPACES.sub("\n\n", md).strip()


def load_pages(docs_dir: Path) -> dict[str, str]:
    pages = {}
    for path in sorted(docs_dir.rglob("*.md")):
        rel = path.relative_to(docs_dir).with_suffix("").as_posix()
        if EXCLUDE.match(rel):
            continue
        text = clean(path.read_text(encoding="utf-8"))
        if len(text.split()) >= 30:
            pages[rel] = text
    return pages


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    docs, sha = Path(sys.argv[1]), sys.argv[2]
    pages = load_pages(docs)
    if len(pages) < 100:
        raise SystemExit(f"only {len(pages)} pages found: wrong path or upstream layout changed")
    with OUT.open("w", encoding="utf-8") as f:
        for page, text in pages.items():
            f.write(json.dumps({"page": page, "text": text}, ensure_ascii=False) + "\n")
    NOTICE.write_text(re.sub(r"commit `[0-9a-f]{7,40}`", f"commit `{sha}`", NOTICE.read_text(encoding="utf-8")),
                      encoding="utf-8")
    print(f"{len(pages)} pages, {sum(len(t.split()) for t in pages.values()):,} words, upstream {sha[:7]}")


if __name__ == "__main__":
    main()
