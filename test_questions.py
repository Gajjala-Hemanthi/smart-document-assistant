
from langchain_core.documents import Document

from retrieval import retrieve_context


def test_retrieves_calibration_evidence():
    docs = [
        Document(
            page_content=(
                "The prototype has not been calibrated "
                "against laboratory instruments."
            ),
            metadata={"source": "Green_Campus.txt"},
        )
    ]

    results = retrieve_context(
        None,
        "What does the document say about calibration?",
        docs,
    )

    assert results
    assert "calibrated" in results[0].page_content.lower()


def test_retrieves_wifi_duration():
    docs = [
        Document(
            page_content="Wi-Fi was unavailable for 18 minutes.",
            metadata={"source": "Green_Campus.txt"},
        )
    ]

    results = retrieve_context(
        None,
        "How long was Wi-Fi unavailable?",
        docs,
    )

    assert results
    assert "18 minutes" in results[0].page_content


def test_empty_documents_return_no_results():
    results = retrieve_context(
        None,
        "What is the university name?",
        [],
    )

    assert results == []
