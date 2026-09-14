import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.rag.citations import (
    build_citations,
    extract_snippet,
    format_citation_text,
    format_citations_block,
    validate_page,
)
from app.rag.llm import FakeLLMService
from app.schemas.chat import Citation
from tests.test_documents import make_token
from tests.test_retriever import create_indexed_doc


def test_snippet_generation_deterministic_and_word_preserved() -> None:
    # Short text
    short_text = "Photosynthesis converts light into chemical energy."
    snippet = extract_snippet(short_text, max_length=120)
    assert snippet == short_text

    # Long text (>120 characters)
    long_text = (
        "Database normalization is the process of structuring a relational database in accordance with a series of "
        "normal forms in order to reduce data redundancy and improve data integrity across tables."
    )
    long_snippet = extract_snippet(long_text, max_length=120)
    assert len(long_snippet) <= 120
    assert long_snippet.endswith("...")
    # Verify word boundary: should not end with a partial word before ellipsis
    prefix = long_snippet[:-3]
    assert not prefix.endswith(" ")
    assert long_snippet.startswith("Database normalization is")

    # Multi-line whitespace normalization
    messy_text = "  First line. \n\n   Second line with   tabs.\t\tThird line.  "
    clean_snippet = extract_snippet(messy_text, max_length=120)
    assert "\n" not in clean_snippet
    assert "\t" not in clean_snippet
    assert "First line. Second line with tabs. Third line." == clean_snippet


def test_validate_page_strictly() -> None:
    # PDF chunks must have valid positive integer page
    assert validate_page("lecture.pdf", 5) == 5
    assert validate_page("lecture.PDF", "18") == 18
    assert validate_page("lecture.pdf", 0) is None
    assert validate_page("lecture.pdf", -1) is None
    assert validate_page("lecture.pdf", "invalid") is None
    assert validate_page("lecture.pdf", None) is None

    # TXT and Markdown must always be None
    assert validate_page("notes.txt", 1) is None
    assert validate_page("guide.md", 2) is None
    assert validate_page("README.MD", 10) is None


def test_duplicate_citation_merging_by_document_and_page() -> None:
    chunks = [
        # Document 1, Page 4: 3 chunks
        {
            "chunk_id": "c1",
            "document_id": "doc-1",
            "filename": "dbms.pdf",
            "page": 4,
            "score": 0.72,
            "text": "Early paragraph on 1NF definition.",
        },
        {
            "chunk_id": "c2",
            "document_id": "doc-1",
            "filename": "dbms.pdf",
            "page": 4,
            "score": 0.94,
            "text": "Primary definition of 1NF eliminating repeating groups.",
        },
        {
            "chunk_id": "c3",
            "document_id": "doc-1",
            "filename": "dbms.pdf",
            "page": 4,
            "score": 0.81,
            "text": "Examples of 1NF violation and fixes.",
        },
        # Document 1, Page 5: 1 chunk (different page, must NOT merge with page 4)
        {
            "chunk_id": "c4",
            "document_id": "doc-1",
            "filename": "dbms.pdf",
            "page": 5,
            "score": 0.88,
            "text": "Second normal form removing partial key dependencies.",
        },
        # Document 2 (TXT): 2 chunks (both page=None, should merge into 1 TXT citation)
        {
            "chunk_id": "c5",
            "document_id": "doc-2",
            "filename": "summary.txt",
            "page": None,
            "score": 0.65,
            "text": "Brief overview of normalization.",
        },
        {
            "chunk_id": "c6",
            "document_id": "doc-2",
            "filename": "summary.txt",
            "page": None,
            "score": 0.78,
            "text": "Key takeaways on database design.",
        },
    ]

    citations = build_citations(chunks)

    # We expect exactly 3 citations: (doc-1, page 4), (doc-1, page 5), (doc-2, page None)
    assert len(citations) == 3

    # Verify doc-1 page 4 merged: kept highest score (0.94) and chunk c2's snippet
    c_page4 = next(c for c in citations if c.document_id == "doc-1" and c.page == 4)
    assert c_page4.chunk_id == "c2"
    assert c_page4.similarity_score == 0.94
    assert "Primary definition of 1NF" in c_page4.snippet

    # Verify doc-1 page 5 remained separate
    c_page5 = next(c for c in citations if c.document_id == "doc-1" and c.page == 5)
    assert c_page5.chunk_id == "c4"
    assert c_page5.similarity_score == 0.88

    # Verify doc-2 TXT merged with highest score (0.78)
    c_txt = next(c for c in citations if c.document_id == "doc-2")
    assert c_txt.page is None
    assert c_txt.chunk_id == "c6"
    assert c_txt.similarity_score == 0.78


def test_citation_ordering_score_then_filename_then_page() -> None:
    chunks = [
        {"chunk_id": "1", "document_id": "d1", "filename": "Zebra.pdf", "page": 10, "score": 0.60, "text": "Zebra text"},
        {"chunk_id": "2", "document_id": "d2", "filename": "Alpha.pdf", "page": 2, "score": 0.95, "text": "Alpha text"},
        {"chunk_id": "3", "document_id": "d3", "filename": "Beta.pdf", "page": 5, "score": 0.80, "text": "Beta text"},
        {"chunk_id": "4", "document_id": "d4", "filename": "Apple.pdf", "page": 1, "score": 0.80, "text": "Apple text"},
    ]

    citations = build_citations(chunks)
    assert len(citations) == 4

    # 1. Alpha.pdf (score 0.95)
    assert citations[0].filename == "Alpha.pdf"
    assert citations[0].similarity_score == 0.95

    # 2. Apple.pdf (score 0.80, 'apple' < 'beta')
    assert citations[1].filename == "Apple.pdf"
    assert citations[1].similarity_score == 0.80

    # 3. Beta.pdf (score 0.80)
    assert citations[2].filename == "Beta.pdf"
    assert citations[2].similarity_score == 0.80

    # 4. Zebra.pdf (score 0.60)
    assert citations[3].filename == "Zebra.pdf"
    assert citations[3].similarity_score == 0.60


def test_missing_metadata_fallback() -> None:
    chunks = [
        {
            # Missing document_id, filename, text, page invalid
            "page": "invalid_page_val",
            "score": "0.75",
        }
    ]
    citations = build_citations(chunks)
    assert len(citations) == 1
    c = citations[0]
    assert c.document_id == ""
    assert c.filename == "Unknown Document"
    assert c.page is None
    assert c.similarity_score == 0.75
    assert c.snippet == ""


def test_citation_formatting_helper() -> None:
    c_pdf = Citation(
        document_id="doc-1",
        filename="DBMS_Notes.pdf",
        page=18,
        chunk_id="c-18",
        similarity_score=0.91,
        snippet="Normalization reduces data redundancy.",
    )
    formatted = format_citation_text(c_pdf)
    assert "Document: DBMS_Notes.pdf" in formatted
    assert "Page: 18" in formatted
    assert 'Snippet: "Normalization reduces data redundancy."' in formatted

    c_txt = Citation(
        document_id="doc-2",
        filename="notes.txt",
        page=None,
        chunk_id="c-2",
        similarity_score=0.74,
        snippet="Plain text notes on SQL.",
    )
    formatted_txt = format_citation_text(c_txt)
    assert "Document: notes.txt" in formatted_txt
    assert "Page: N/A" in formatted_txt

    block = format_citations_block([c_pdf, c_txt])
    assert "DBMS_Notes.pdf" in block
    assert "notes.txt" in block


@pytest.mark.anyio
async def test_end_to_end_chat_citations_with_deduplication_and_anti_spoofing(chat_setup) -> None:
    database, settings, embedder, _ = chat_setup
    user_id = "student-cite"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-physics",
        filename="Physics_Mechanics.pdf",
        chunks_text=[
            ("Newton's second law states F = ma describing mass acceleration.", 22),
            ("Force equals rate of change of momentum over time.", 22),  # Same page! Must merge
            ("Newton's third law states action equals reaction.", 23),    # Different page! Must not merge
        ],
    )

    # Adversarial LLM attempts to fabricate a citation in its response text
    adversarial_llm = FakeLLMService(
        canned_response="Force is related to acceleration.\nSources:\n- Hallucinated_Book.pdf — Page 99"
    )
    app.dependency_overrides[app.dependency_overrides[type(adversarial_llm)] if False else None] = lambda: adversarial_llm
    from app.core.deps import get_llm
    app.dependency_overrides[get_llm] = lambda: adversarial_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/chat",
            json={"message": "State Newton's second and third laws."},
            headers=headers,
        )

    assert resp.status_code == 200
    data = resp.json()

    # Verify citations in response
    citations = data["citations"]
    assert len(citations) == 2  # Page 22 merged, page 23 separate

    pages = [c["page"] for c in citations]
    assert 22 in pages
    assert 23 in pages

    for c in citations:
        assert c["document_id"] == "doc-physics"
        assert c["filename"] == "Physics_Mechanics.pdf"
        assert "similarity_score" in c
        assert "snippet" in c
        assert len(c["snippet"]) > 0

    # Verify anti-spoofing: the LLM's text attempt did NOT pollute the structured citations
    assert not any("Hallucinated_Book.pdf" in c["filename"] for c in citations)

    # Verify cross-user isolation: User Bob cannot see these citations
    bob_token = make_token("user-bob-no-docs", settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        bob_resp = await client.post(
            "/chat",
            json={"message": "State Newton's second and third laws."},
            headers={"Authorization": f"Bearer {bob_token}"},
        )
    assert bob_resp.status_code == 200
    assert bob_resp.json()["citations"] == []
