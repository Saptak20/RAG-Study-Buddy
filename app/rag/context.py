"""Context formatting and boundary enforcement for RAG generation."""

from __future__ import annotations

from typing import Any


def format_retrieved_context(
    chunks: list[dict[str, Any]],
    max_context_chars: int = 4000,
) -> str:
    """Format retrieved candidate chunks into a clean, bounded context block for LLM prompts."""
    if not chunks or max_context_chars <= 0:
        return ""

    formatted_sources: list[str] = []
    current_length = 0

    for i, chunk in enumerate(chunks, start=1):
        filename = chunk.get("filename", "Unknown Document")
        page = chunk.get("page")
        page_str = str(page) if page is not None else "N/A"
        text = chunk.get("text", "").strip()

        source_header = f"[Source {i}]\nDocument: {filename}\nPage: {page_str}\nContent:\n"
        header_len = len(source_header)

        # If the header itself would exceed remaining budget, stop
        if current_length + header_len >= max_context_chars:
            break

        remaining_budget = max_context_chars - (current_length + header_len)
        if len(text) <= remaining_budget:
            source_block = f"{source_header}{text}"
            formatted_sources.append(source_block)
            current_length += len(source_block) + 2
        else:
            if remaining_budget > 50:
                truncated_text = text[: remaining_budget - 16].rstrip() + "... [truncated]"
                source_block = f"{source_header}{truncated_text}"
                formatted_sources.append(source_block)
            break

    return "\n\n".join(formatted_sources)
