"""Academic citation generation, validation, deduplication, and formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.schemas.chat import Citation


def extract_snippet(text: str, max_length: int = 120) -> str:
    """Extract a clean, deterministic, word-preserved snippet of up to max_length characters."""
    if not text or max_length <= 0:
        return ""

    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_length:
        return cleaned

    if max_length <= 3:
        return cleaned[:max_length]

    budget = max_length - 3
    truncated = cleaned[:budget]

    last_space = truncated.rfind(" ")
    if last_space > int(budget * 0.7):
        truncated = truncated[:last_space]

    return f"{truncated.rstrip()}..."


def validate_page(filename: str, page: Any) -> int | None:
    """Validate page numbers strictly: PDF chunks must have positive int page; TXT/MD are None."""
    ext = Path(filename).suffix.lower()
    if ext != ".pdf":
        return None

    if isinstance(page, int) and page > 0:
        return page

    if isinstance(page, str) and page.isdigit() and int(page) > 0:
        return int(page)

    return None


def build_citations(chunks: list[dict[str, Any]]) -> list[Citation]:
    """Build deduplicated, validated, and ranked academic citation objects from retrieved chunks."""
    if not chunks:
        return []

    merged: dict[tuple[str, int | None], dict[str, Any]] = {}

    for chunk in chunks:
        doc_id = str(chunk.get("document_id", ""))
        filename = str(chunk.get("filename", "Unknown Document"))
        raw_page = chunk.get("page")
        valid_page = validate_page(filename, raw_page)
        chunk_id = str(chunk.get("chunk_id", ""))
        score = float(chunk.get("score", 0.0))
        text = str(chunk.get("text", ""))

        key = (doc_id, valid_page)
        if key not in merged or score > merged[key]["similarity_score"]:
            merged[key] = {
                "document_id": doc_id,
                "filename": filename,
                "page": valid_page,
                "chunk_id": chunk_id,
                "similarity_score": round(score, 4),
                "score": round(score, 4),
                "snippet": extract_snippet(text, max_length=120),
            }

    def sort_key(c: dict[str, Any]) -> tuple[float, str, int]:
        score_component = -c["similarity_score"]
        name_component = c["filename"].lower()
        page_component = c["page"] if c["page"] is not None else 999999
        return (score_component, name_component, page_component)

    sorted_citations = sorted(merged.values(), key=sort_key)
    return [Citation(**item) for item in sorted_citations]


def format_citation_text(citation: Citation) -> str:
    """Format a single citation into a clean academic string."""
    page_str = str(citation.page) if citation.page is not None else "N/A"
    return f'Document: {citation.filename}\nPage: {page_str}\nSnippet: "{citation.snippet}"'


def format_citations_block(citations: list[Citation]) -> str:
    """Format multiple citations into a readable string block."""
    if not citations:
        return ""
    return "\n\n".join(format_citation_text(c) for c in citations)
