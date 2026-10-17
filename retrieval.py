
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "what", "which", "who", "whom",
    "when", "where", "why", "how", "can", "could", "would", "should",
    "tell", "me", "about", "according", "to", "document", "documents",
    "please", "give", "find", "show", "explain", "in", "on", "at",
    "of", "for", "and", "or", "with", "from", "it", "this", "that",
    "their", "there", "they", "he", "she", "we", "you", "your"
}


def normalize(text: str) -> str:
    """Normalize text for lexical matching."""
    text = str(text or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def retrieve_context(
    vector_store,
    question: str,
    source_docs: list[Document],
    k: int = 5,
) -> list[Document]:
    """
    Retrieve relevant document chunks using FAISS semantic search
    and lexical matching as a fallback.
    """
    if not question or not question.strip() or not source_docs:
        return []

    k = max(1, int(k))
    question_terms = {
    "calibrat" if word in {"calibration", "calibrated", "calibrate", "calibrating"}
    else word
    for word in normalize(question).split()
    if word not in STOP_WORDS and len(word) > 1
    }

    if not question_terms:
        return []

    candidates = []
    seen = set()

    # 1. Semantic retrieval using FAISS.
    if vector_store is not None:
        try:
            semantic_docs = vector_store.similarity_search(
                question,
                k=max(k * 3, 10),
            )

            for doc in semantic_docs:
                content = doc.page_content or ""
                key = (
                    content,
                    str(doc.metadata.get("source", "")),
                    str(doc.metadata.get("page", "")),
                )

                if key not in seen:
                    seen.add(key)
                    candidates.append(doc)

        except Exception:
            # Continue with lexical retrieval if semantic search fails.
            pass

    # 2. Include original source documents for lexical matching.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    for doc in source_docs:
        content = doc.page_content or ""

        if not content.strip():
            continue

        chunks = splitter.split_text(content) or [content]

        for chunk in chunks:
            key = (
                chunk,
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("page", "")),
            )

            if key not in seen:
                seen.add(key)
                candidates.append(
                    Document(
                        page_content=chunk,
                        metadata=dict(doc.metadata),
                    )
                )

    # 3. Rank candidates by lexical overlap.
    ranked = []

    for doc in candidates:
        text = normalize(doc.page_content)
        doc_terms = {"calibrat" if word in {"calibration", "calibrated", "calibrate", "calibrating"}
        else word
        for word in text.split()
        }

        overlap = question_terms.intersection(doc_terms)

        if not overlap:
            continue

        score = len(overlap) / len(question_terms)

        # Small bonus for exact phrases.
        normalized_question = normalize(question)

        if normalized_question and normalized_question in text:
            score += 0.5

        # Bonus for matching important numbers or durations.
        for term in question_terms:
            if term.isdigit() and term in doc_terms:
                score += 0.2

        ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)

    results = [doc for score, doc in ranked if score > 0]

    return results[:k]
