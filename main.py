from __future__ import annotations
from retrieval import retrieve_context as shared_retrieve_context

import math
import re
from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from io import BytesIO
from typing import Any

import streamlit as st

# Document loaders / spreadsheet readers
from pypdf import PdfReader
from openpyxl import load_workbook
import xlrd

# LangChain + local Ollama + FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


st.set_page_config(page_title="Smart Document Assistant", page_icon="📄", layout="wide")

EMBED_MODEL_DEFAULT = "nomic-embed-text"
LLM_MODEL_DEFAULT = "llama3.2:1b"
OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"


# ----------------------------- Text utilities -----------------------------

def display_value(value: Any, number_format: str | None = None) -> str:
    """Convert common spreadsheet/PDF values to readable text without needless .0."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        # Respect simple Excel zero-padding formats such as 00000 where possible.
        if number_format and re.fullmatch(r"0{2,}", number_format) and value.is_integer():
            return str(int(value)).zfill(len(number_format))
        if value.is_integer():
            return str(int(value))
    return str(value).strip()


def normalize(text: Any) -> str:
    text = str(text or "").strip().casefold()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^\w\s.%]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_filename(name: str) -> str:
    return name.replace("\n", " ").strip()


def make_unique_headers(raw_headers: list[str]) -> list[str]:
    """Keep headers dynamic, including blank and duplicated headers."""
    output: list[str] = []
    counts: Counter[str] = Counter()
    for index, raw in enumerate(raw_headers, start=1):
        base = raw.strip() or f"Column {index}"
        counts[base.casefold()] += 1
        if counts[base.casefold()] == 1:
            output.append(base)
        else:
            output.append(f"{base} ({counts[base.casefold()]})")
    return output


def row_to_text(headers: list[str], values: list[str]) -> str:
    pairs = [f"{header}: {value}" for header, value in zip(headers, values) if value]
    return " | ".join(pairs)


# ----------------------------- File extraction -----------------------------

def extract_pdf(file_name: str, file_bytes: bytes) -> tuple[list[Document], list[dict]]:
    docs: list[Document] = []
    warnings: list[dict] = []
    try:
        reader = PdfReader(BytesIO(file_bytes))
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                warnings.append({"source": file_name, "message": f"Page {page_number} could not be read: {exc}"})
                continue
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file_name, "page": page_number, "file_type": "PDF"}
                ))
        if not docs:
            warnings.append({
                "source": file_name,
                "message": "No selectable text was extracted. If this is a scanned PDF, OCR is needed."
            })
    except Exception as exc:
        warnings.append({"source": file_name, "message": f"Could not read PDF: {exc}"})
    return docs, warnings


def extract_txt(file_name: str, file_bytes: bytes) -> tuple[list[Document], list[dict]]:
    warnings: list[dict] = []
    text = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        warnings.append({"source": file_name, "message": "Could not decode text file."})
        return [], warnings
    if not text.strip():
        warnings.append({"source": file_name, "message": "Text file is empty."})
        return [], warnings
    # Split long text into manageable source units while preserving source name.
    blocks = re.split(r"\n\s*\n", text)
    docs = [
        Document(page_content=block.strip(), metadata={"source": file_name, "file_type": "TXT"})
        for block in blocks if block.strip()
    ]
    if not docs:
        docs = [Document(page_content=text, metadata={"source": file_name, "file_type": "TXT"})]
    return docs, warnings


def extract_xlsx(file_name: str, file_bytes: bytes) -> tuple[list[Document], list[dict], list[dict]]:
    """Return retrieval documents, structured rows, and warnings for XLSX."""
    docs: list[Document] = []
    records: list[dict] = []
    warnings: list[dict] = []
    try:
        # data_only reads cached formula results when the workbook has them.
        workbook = load_workbook(BytesIO(file_bytes), read_only=True, data_only=True)
        for sheet in workbook.worksheets:
            rows_iter = sheet.iter_rows()
            first_row = next(rows_iter, None)
            if first_row is None:
                continue
            raw_headers = [display_value(cell.value, cell.number_format) for cell in first_row]
            # Skip completely blank leading rows, if any.
            while first_row and not any(raw_headers):
                first_row = next(rows_iter, None)
                if first_row is None:
                    break
                raw_headers = [display_value(cell.value, cell.number_format) for cell in first_row]
            if first_row is None:
                continue
            headers = make_unique_headers(raw_headers)
            data_row_number = 2
            for cells in rows_iter:
                values = [display_value(cell.value, cell.number_format) for cell in cells]
                if not any(value.strip() for value in values):
                    data_row_number += 1
                    continue
                # Align shorter rows to the header width and ignore overflow columns only if no header exists.
                if len(values) < len(headers):
                    values += [""] * (len(headers) - len(values))
                values = values[:len(headers)]
                record = {
                    "source": file_name,
                    "sheet": sheet.title,
                    "row": data_row_number,
                    "headers": headers,
                    "values": values,
                    "data": dict(zip(headers, values)),
                }
                records.append(record)
                content = row_to_text(headers, values)
                if content:
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": file_name,
                            "sheet": sheet.title,
                            "row": data_row_number,
                            "file_type": "XLSX",
                        },
                    ))
                data_row_number += 1
        workbook.close()
    except Exception as exc:
        warnings.append({"source": file_name, "message": f"Could not read XLSX: {exc}"})
    return docs, records, warnings


def extract_xls(file_name: str, file_bytes: bytes) -> tuple[list[Document], list[dict], list[dict]]:
    docs: list[Document] = []
    records: list[dict] = []
    warnings: list[dict] = []
    try:
        workbook = xlrd.open_workbook(file_contents=file_bytes)
        for sheet in workbook.sheets():
            if sheet.nrows == 0:
                continue
            raw_headers = [display_value(sheet.cell_value(0, col)) for col in range(sheet.ncols)]
            headers = make_unique_headers(raw_headers)
            for row_idx in range(1, sheet.nrows):
                values = [display_value(sheet.cell_value(row_idx, col)) for col in range(sheet.ncols)]
                if not any(value.strip() for value in values):
                    continue
                record = {
                    "source": file_name,
                    "sheet": sheet.name,
                    "row": row_idx + 1,
                    "headers": headers,
                    "values": values,
                    "data": dict(zip(headers, values)),
                }
                records.append(record)
                content = row_to_text(headers, values)
                if content:
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": file_name,
                            "sheet": sheet.name,
                            "row": row_idx + 1,
                            "file_type": "XLS",
                        },
                    ))
    except Exception as exc:
        warnings.append({"source": file_name, "message": f"Could not read XLS: {exc}"})
    return docs, records, warnings


def extract_uploaded_files(uploaded_files) -> tuple[list[Document], list[dict], list[dict]]:
    all_docs: list[Document] = []
    all_records: list[dict] = []
    all_warnings: list[dict] = []
    for uploaded in uploaded_files:
        name = safe_filename(uploaded.name)
        data = uploaded.getvalue()
        suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        if suffix == "pdf":
            docs, warnings = extract_pdf(name, data)
            all_docs.extend(docs)
            all_warnings.extend(warnings)
        elif suffix == "txt":
            docs, warnings = extract_txt(name, data)
            all_docs.extend(docs)
            all_warnings.extend(warnings)
        elif suffix == "xlsx":
            docs, records, warnings = extract_xlsx(name, data)
            all_docs.extend(docs)
            all_records.extend(records)
            all_warnings.extend(warnings)
        elif suffix == "xls":
            docs, records, warnings = extract_xls(name, data)
            all_docs.extend(docs)
            all_records.extend(records)
            all_warnings.extend(warnings)
        else:
            all_warnings.append({"source": name, "message": "Unsupported file type. Use PDF, TXT, XLSX, or XLS."})
    return all_docs, all_records, all_warnings


# ----------------------------- Structured question answering -----------------------------

COUNT_WORDS = {"count", "number", "total", "how many", "quantity"}
AGGREGATION_WORDS = {
    "sum": ("sum", "total of", "add up", "combined"),
    "average": ("average", "mean"),
    "maximum": ("maximum", "max", "highest", "largest", "most expensive", "greatest"),
    "minimum": ("minimum", "min", "lowest", "smallest", "least expensive", "least"),
}


def question_mentions_column(question: str, headers: list[str]) -> str | None:
    qn = normalize(question)
    candidates = sorted(headers, key=lambda h: len(normalize(h)), reverse=True)
    for header in candidates:
        hn = normalize(header)
        if hn and (hn in qn or (len(hn.split()) == 1 and re.search(rf"\b{re.escape(hn)}\b", qn))):
            return header
    return None


def all_headers(records: list[dict]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for record in records:
        for header in record["headers"]:
            key = f'{record["source"]}::{record["sheet"]}::{header}'
            # Preserve first occurrence for normal lookup; columns vary by sheet/file.
            if header.casefold() not in seen:
                result.append(header)
                seen.add(header.casefold())
    return result


def parse_numeric(value: str) -> Decimal | None:
    if not value:
        return None
    candidate = value.strip().replace(",", "")
    candidate = re.sub(r"^[₹$€£]\s*", "", candidate)
    candidate = candidate.rstrip("%").strip()
    try:
        result = Decimal(candidate)
        if not result.is_finite():
            return None
        return result
    except (InvalidOperation, ValueError):
        return None


def is_count_question(question: str) -> bool:
    q = normalize(question)
    return (
        "how many" in q
        or "count" in q
        or "number of" in q
        or "total number" in q
        or q.startswith("total ")
        or "how much" in q
    )


def record_scope(question: str, records: list[dict]) -> list[dict]:
    q = normalize(question)
    # If a file or sheet name is mentioned, narrow to it.
    scoped = records
    for token_source in sorted({r["source"] for r in records}, key=len, reverse=True):
        if normalize(token_source) in q:
            scoped = [r for r in scoped if r["source"] == token_source]
            break
    for token_sheet in sorted({r["sheet"] for r in scoped}, key=len, reverse=True):
        if normalize(token_sheet) in q:
            scoped = [r for r in scoped if r["sheet"] == token_sheet]
            break
    return scoped


def requested_output_column(question: str, headers: list[str], lookup_column: str | None = None) -> str | None:
    """Infer the field the user wants returned, separately from the lookup field."""
    qn = normalize(question)
    # A user can mention both the key column and the desired answer column.
    # Prefer output cues such as "what name", "which size", or "give me the email".
    cue_words = ("name", "email", "phone", "address", "size", "date", "timestamp", "price", "amount", "status", "value")
    for cue in cue_words:
        if re.search(rf"\b{cue}\b", qn):
            possible = []
            for header in headers:
                h_words = normalize(header).split()
                if cue in h_words:
                    possible.append(header)
            if len(possible) == 1:
                return possible[0]
            if possible:
                # Prefer the shortest matching header if one is clearly the simple field.
                possible.sort(key=lambda h: len(normalize(h)))
                if len(possible) == 1 or len(normalize(possible[0])) < len(normalize(possible[1])):
                    return possible[0]
    return None


def find_exact_cell_matches(question: str, records: list[dict], lookup_column: str | None = None) -> list[dict]:
    """Find exact values and safe short suffixes of identifiers in spreadsheet rows.

    Partial matching is enabled only when the question explicitly names a lookup column
    and provides a short alphanumeric fragment (for example, an ID suffix such as B9).
    It never uses the fragment against every arbitrary cell in the workbook.
    """
    q = normalize(question)
    matches = []
    unique_values = sorted(
        {value for record in records for value in record["values"] if len(value.strip()) >= 3},
        key=len,
        reverse=True,
    )
    for value in unique_values:
        vn = normalize(value)
        if vn and re.search(rf"(?<!\w){re.escape(vn)}(?!\w)", q):
            for record in records:
                if any(normalize(cell) == vn for cell in record["values"]):
                    matches.append(record)
            if matches:
                return matches

    if lookup_column:
        # Pull a short token after a lookup-column phrase, such as "roll number B9".
        header_words = normalize(lookup_column)
        q_match = re.search(rf"{re.escape(header_words)}\s+(?:is\s+|of\s+|:)?([a-z0-9][a-z0-9_-]{{1,7}})\b", q)
        fragment = q_match.group(1) if q_match else None
        if fragment and len(fragment) <= 8 and re.search(r"[a-z]", fragment) and re.search(r"\d", fragment):
            fragment_key = re.sub(r"[^a-z0-9]", "", fragment.casefold())
            for record in records:
                cell = record["data"].get(lookup_column, "")
                cell_key = re.sub(r"[^a-z0-9]", "", normalize(cell))
                if cell_key and cell_key.endswith(fragment_key):
                    matches.append(record)
            if matches:
                return matches
    return []


def direct_structured_answer(question: str, records: list[dict]) -> dict | None:
    """Answer safe, deterministic spreadsheet questions without asking the LLM to count."""
    if not records:
        return None
    scoped = record_scope(question, records)
    if not scoped:
        return None
    qn = normalize(question)
    headers = all_headers(scoped)
    column = question_mentions_column(question, headers)

    # Count rows matching a value mentioned in the question, e.g. "How many
    # participants are marked Present?" The column/value are discovered from
    # the actual table; neither status labels nor column names are hardcoded.
    if is_count_question(question):
        matched_values = set()
        for record in scoped:
            for value in record["values"]:
                value_norm = normalize(value)
                if len(value_norm) >= 2 and re.search(
                    rf"(?<!\\w){re.escape(value_norm)}(?!\\w)", qn
                ):
                    matched_values.add(value_norm)

        # If a mentioned value belongs to one column, count rows matching it
        # in that column. If several columns contain it, avoid double-counting
        # rows and count each row once.
        if matched_values:
            value_column_counts = []
            for header in headers:
                matching = [
                    r for r in scoped
                    if normalize(r["data"].get(header, "")) in matched_values
                ]
                if matching:
                    value_column_counts.append((header, matching))
            if value_column_counts:
                # Prefer the column with the most matching rows; this also
                # handles repeated status values in the same column.
                value_column_counts.sort(key=lambda item: len(item[1]), reverse=True)
                matched_header, matched_rows = value_column_counts[0]
                # Only use this path if the question seems to count a category/status,
                # not merely if a common word accidentally appears in the question.
                category_intent = any(
                    phrase in qn for phrase in (
                        "marked", "status", "whose", "with ", "are ", "is ",
                        "listed as", "labelled", "labeled", "registered as",
                        "belong to", "in the", "have ", "has "
                    )
                )
                if category_intent:
                    answer = (
                        f"**{len(matched_rows)}** records match **{matched_header} = "
                        f"{', '.join(sorted(matched_values))}**."
                    )
                    return {
                        "answer": answer,
                        "records": matched_rows[:10],
                        "method": "Exact spreadsheet category count",
                    }

        # Count records/responses/entries, or count non-empty/unique values in a column.
        if column:
            values = [r["data"].get(column, "").strip() for r in scoped]
            nonempty = [v for v in values if v]
            if any(word in qn for word in ("unique", "distinct", "different")):
                answer = f"There are {len(set(normalize(v) for v in nonempty))} unique non-empty values in **{column}**."
            elif any(word in qn for word in ("filled", "non empty", "nonempty", "answered")):
                answer = f"**{len(nonempty)}** rows have a non-empty value in **{column}** (out of {len(scoped)} records)."
            else:
                answer = f"**{len(scoped)}** records are in scope. The **{column}** column has {len(nonempty)} non-empty values."
        else:
            answer = f"**{len(scoped)}** non-empty spreadsheet records are in scope."
        return {"answer": answer, "records": scoped[:10], "method": "Exact spreadsheet count"}

    # Aggregations: choose a named column, or infer a likely numeric column.
    aggregation = None
    for agg, phrases in AGGREGATION_WORDS.items():
        if any(phrase in qn for phrase in phrases):
            aggregation = agg
            break
    if aggregation:
        candidate_headers = [column] if column else headers
        best = None
        for header in candidate_headers:
            if not header:
                continue
            parsed = [(r, parse_numeric(r["data"].get(header, ""))) for r in scoped]
            valid = [(r, value) for r, value in parsed if value is not None]
            if valid:
                best = (header, valid)
                if column:
                    break
        if best:
            header, valid = best
            vals = [v for _, v in valid]
            if aggregation == "sum":
                result = sum(vals, Decimal("0"))
                answer = f"**Sum of {header}: {result.normalize():f}** (calculated from {len(vals)} numeric values)."
            elif aggregation == "average":
                result = sum(vals, Decimal("0")) / Decimal(len(vals))
                answer = f"**Average of {header}: {result.quantize(Decimal('0.01'))}** (calculated from {len(vals)} numeric values)."
            elif aggregation == "maximum":
                result = max(vals)
                rows = [r for r, v in valid if v == result]
                answer = f"**Highest {header}: {result.normalize():f}**\n\nMatching record(s):\n" + "\n".join(
                    f"- {row_to_text(r['headers'], r['values'])} ({r['source']} / {r['sheet']}, row {r['row']})"
                    for r in rows[:10]
                )
            else:
                result = min(vals)
                rows = [r for r, v in valid if v == result]
                answer = f"**Lowest {header}: {result.normalize():f}**\n\nMatching record(s):\n" + "\n".join(
                    f"- {row_to_text(r['headers'], r['values'])} ({r['source']} / {r['sheet']}, row {r['row']})"
                    for r in rows[:10]
                )
            return {"answer": answer, "records": [r for r, _ in valid[:10]], "method": f"Python {aggregation}"}

    # Exact value lookups work with any identifier/column, not a fixed header name.
    matches = find_exact_cell_matches(question, scoped, lookup_column=column)
    if matches:
        requested_column = requested_output_column(question, headers, lookup_column=column) or column
        # If the question asks for a field and a matching row contains that field, answer only it.
        if requested_column:
            found_values = []
            for record in matches:
                value = record["data"].get(requested_column, "")
                if value:
                    found_values.append((value, record))
            if found_values and not any(normalize(requested_column) == normalize(v) for v, _ in found_values):
                unique = list(dict.fromkeys(v for v, _ in found_values))
                if len(unique) == 1:
                    answer = f"**{requested_column}: {unique[0]}**"
                else:
                    answer = f"Matching values in **{requested_column}**:\n" + "\n".join(f"- {v}" for v in unique[:20])
            else:
                answer = "I found these matching record(s):\n" + "\n".join(
                    f"- {row_to_text(r['headers'], r['values'])} ({r['source']} / {r['sheet']}, row {r['row']})"
                    for r in matches[:10]
                )
        else:
            answer = "I found these matching record(s):\n" + "\n".join(
                f"- {row_to_text(r['headers'], r['values'])} ({r['source']} / {r['sheet']}, row {r['row']})"
                for r in matches[:10]
            )
        return {"answer": answer, "records": matches[:10], "method": "Exact spreadsheet lookup"}

    # List the values in a requested column.
    list_intent = any(phrase in qn for phrase in ("list ", "show ", "display ", "what are the", "which are the", "give me all"))
    if column and list_intent:
        values = list(dict.fromkeys(r["data"].get(column, "").strip() for r in scoped if r["data"].get(column, "").strip()))
        if values:
            answer = f"**Values in {column}** ({len(values)} unique values shown):\n" + "\n".join(f"- {v}" for v in values[:100])
            return {"answer": answer, "records": scoped[:10], "method": "Spreadsheet column listing"}

    return None


# ----------------------------- Retrieval and generation -----------------------------

def build_vector_store(documents: list[Document], embedding_model: str, ollama_url: str):
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_url)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    if not chunks:
        return None, []
    # Batch to reduce memory spikes with local embedding models.
    vector_store = None
    batch_size = 32
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        partial = FAISS.from_documents(batch, embeddings)
        if vector_store is None:
            vector_store = partial
        else:
            vector_store.merge_from(partial)
    return vector_store, chunks


def format_source(meta: dict) -> str:
    bits = [str(meta.get("source", "Unknown source"))]
    if meta.get("sheet"):
        bits.append(f"sheet {meta['sheet']}")
    if meta.get("page"):
        bits.append(f"page {meta['page']}")
    if meta.get("row"):
        bits.append(f"row {meta['row']}")
    return " — ".join(bits)


def retrieve_context(vector_store, question: str, k: int = 5) -> list[Document]:
    """Combine FAISS semantic results with a lexical fallback over extracted text.

    This prevents a valid sentence from being lost solely because the embedding
    model ranked unrelated chunks higher or the FAISS index is unavailable.
    """
    k = max(1, min(int(k), 12))
    semantic_results: list[Document] = []
    if vector_store is not None:
        try:
            # Fetch more candidates than we display so lexical ranking can rescue
            # exact terms that semantic search ranked too low.
            semantic_results = vector_store.similarity_search(question, k=min(max(k * 4, 12), 40))
        except Exception as exc:
            st.warning(f"Semantic search failed; trying keyword search instead: {exc}")

    # Re-split the extracted source documents so fallback search works even when
    # embeddings/FAISS failed or returned no useful candidates.
    source_docs = getattr(st.session_state, "documents", []) or []
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        lexical_docs = splitter.split_documents(source_docs) if source_docs else []
    except Exception:
        lexical_docs = source_docs

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "what", "when", "where", "who", "why", "how",
        "which", "and", "or", "to", "of", "in", "on", "for", "from", "with",
        "it", "this", "that", "these", "those", "tell", "me", "about", "please",
        "show", "give", "find", "document", "source", "evidence", "say",
    }
    query_terms = [
        term for term in re.findall(r"[a-z0-9]+", normalize(question))
        if len(term) > 2 and term not in stop_words
    ]
    # Prefix matching handles forms such as calibration/calibrated/calibrating.
    def term_matches(query_term: str, content_term: str) -> bool:
        if query_term == content_term:
            return True
        if len(query_term) >= 6 and len(content_term) >= 6:
            return query_term[:6] == content_term[:6]
        return False

    combined: list[Document] = []
    seen: set[tuple[str, str]] = set()
    for doc in semantic_results + lexical_docs:
        key = (str(doc.metadata), doc.page_content)
        if key not in seen:
            seen.add(key)
            combined.append(doc)

    ranked: list[tuple[float, int, Document]] = []
    for position, doc in enumerate(combined):
        content_terms = set(re.findall(r"[a-z0-9]+", normalize(doc.page_content)))
        if query_terms:
            matched = sum(any(term_matches(qt, ct) for ct in content_terms) for qt in query_terms)
            # A matching distinctive term is enough to retain evidence; rank by
            # overlap first, then prefer the existing FAISS order for ties.
            score = matched / len(query_terms)
        else:
            score = 0.0
        if doc in semantic_results:
            score += 0.01
        if score > 0:
            ranked.append((score, -position, doc))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if ranked:
        return [item[2] for item in ranked[:k]]
    # If no lexical overlap exists, preserve semantic results rather than returning
    # an empty context. If both methods found nothing, the caller can show its warning.
    return semantic_results[:k]


def generate_answer(
    question: str,
    retrieved: list[Document],
    chat_history: list[dict],
    model_name: str,
    ollama_url: str,
) -> str:
    if not retrieved:
        return (
            "I couldn't find relevant text in the processed documents for this question. "
            "Try using the wording from the document, or check that the file contains selectable text."
        )
    context_parts = []
    for index, doc in enumerate(retrieved, start=1):
        context_parts.append(f"[Source {index}: {format_source(doc.metadata)}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)
    # Keep prompt bounded for small local models.
    context = context[:14000]
    history_text = ""
    for item in chat_history[-4:]:
        role = "User" if item.get("role") == "user" else "Assistant"
        history_text += f"{role}: {item.get('content', '')[:700]}\n"
    prompt = f"""You are a careful document question-answering assistant.
Answer using only the supplied source context. Do not invent details.
If the answer is not supported by the context, clearly say the documents do not provide enough information.
Be direct and specific. For lists, use bullets. When useful, mention the source filename and page/sheet.
Conversation history (may help interpret follow-up questions, but is not evidence):
{history_text or "(none)"}

Source context:
{context}

Question: {question}

Answer:"""
    try:
        llm = ChatOllama(model=model_name, base_url=ollama_url, temperature=0)
        response = llm.invoke(prompt)
        return str(response.content).strip()
    except Exception as exc:
        return (
            f"I couldn't generate an answer with Ollama: {exc}\n\n"
            "Check that Ollama is running and the selected model is available."
        )


def records_as_documents(records: list[dict]) -> list[Document]:
    docs = []
    for record in records:
        content = row_to_text(record["headers"], record["values"])
        if content:
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": record["source"],
                    "sheet": record["sheet"],
                    "row": record["row"],
                    "file_type": "spreadsheet",
                },
            ))
    return docs


# ----------------------------- Streamlit UI -----------------------------

st.title("📄 Smart Document Assistant")
st.caption("Ask questions across multiple PDF, TXT, XLSX, and XLS files using local Ollama models, FAISS, and Python.")

with st.sidebar:
    st.header("⚙️ Settings")
    ollama_url = st.text_input("Ollama URL", value=OLLAMA_BASE_URL_DEFAULT)
    embedding_model = st.text_input("Embedding model", value=EMBED_MODEL_DEFAULT)
    llm_model = st.text_input("Language model", value=LLM_MODEL_DEFAULT)
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=5)
    st.caption("Exact spreadsheet lookups and arithmetic use Python. PDF/TXT semantic questions use FAISS + Ollama.")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "xlsx", "xls"],
    accept_multiple_files=True,
    help="Upload multiple documents together. Spreadsheet headers are detected dynamically.",
)

if "documents" not in st.session_state:
    st.session_state.documents = []
if "records" not in st.session_state:
    st.session_state.records = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_names" not in st.session_state:
    st.session_state.processed_names = []
if "warnings" not in st.session_state:
    st.session_state.warnings = []
if "messages" not in st.session_state:
    st.session_state.messages = []

left, right = st.columns([1, 1])
with left:
    process_clicked = st.button("🔄 Process / Rebuild index", type="primary", disabled=not uploaded_files)
with right:
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

if process_clicked and uploaded_files:
    with st.spinner("Extracting files and building the local search index..."):
        documents, records, warnings = extract_uploaded_files(uploaded_files)
        # Include row-level spreadsheet docs in FAISS too, so fuzzy spreadsheet questions can retrieve evidence.
        documents.extend(records_as_documents(records))
        st.session_state.documents = documents
        st.session_state.records = records
        st.session_state.warnings = warnings
        st.session_state.processed_names = [f.name for f in uploaded_files]
        st.session_state.vector_store = None
        st.session_state.chunks = []
        st.session_state.messages = []
        if documents:
            try:
                vector_store, chunks = build_vector_store(documents, embedding_model, ollama_url)
                st.session_state.vector_store = vector_store
                st.session_state.chunks = chunks
            except Exception as exc:
                st.error(
                    f"Text was extracted, but the FAISS index could not be built: {exc}\n\n"
                    "Check Ollama is running, the embedding model is pulled, and enough memory is available."
                )
        st.rerun()

if st.session_state.warnings:
    with st.expander("⚠️ Extraction warnings", expanded=False):
        for warning in st.session_state.warnings:
            st.write(f"**{warning['source']}** — {warning['message']}")

if st.session_state.documents:
    st.success(
        f"Processed {len(st.session_state.processed_names)} uploaded file(s) · "
        f"{len(st.session_state.documents)} extracted text units · "
        f"{len(st.session_state.records)} spreadsheet records · "
        f"{len(st.session_state.chunks)} search chunks."
    )
    with st.expander("📚 Processed documents and extracted text", expanded=False):
        names = sorted(set(st.session_state.processed_names))
        for name in names:
            st.markdown(f"**{name}**")
        preview_docs = st.session_state.documents[:8]
        for i, doc in enumerate(preview_docs, start=1):
            st.markdown(f"**Text unit {i} — {format_source(doc.metadata)}**")
            st.text(doc.page_content[:1200])
        if len(st.session_state.documents) > len(preview_docs):
            st.caption(f"Showing 8 of {len(st.session_state.documents)} extracted text units.")

    if st.session_state.records:
        with st.expander("📊 Spreadsheet overview", expanded=False):
            st.write(f"Structured rows available for exact lookup/counting: **{len(st.session_state.records)}**")
            overview = []
            seen_sheets = set()
            for record in st.session_state.records:
                key = (record["source"], record["sheet"])
                if key not in seen_sheets:
                    seen_sheets.add(key)
                    overview.append({
                        "File": record["source"],
                        "Sheet": record["sheet"],
                        "Detected columns": ", ".join(record["headers"]),
                    })
            st.dataframe(overview, use_container_width=True, hide_index=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("🔎 Sources and evidence", expanded=True):
                    for source in message["sources"]:
                        st.markdown(f"**{source['label']}**")
                        st.text(source["text"][:1800])
            elif message["role"] == "assistant":
                st.caption("No source evidence was saved for this response.")

    question = st.chat_input("Ask a question about your uploaded documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Finding evidence..."):
                # Structured operations come first so counts and exact cell matches do not depend on LLM guessing.
                structured = direct_structured_answer(question, st.session_state.records)
                if structured:
                    answer = structured["answer"]
                    sources = []
                    for record in structured.get("records", []):
                        sources.append({
                            "label": f'{record["source"]} — sheet {record["sheet"]}, row {record["row"]} ({structured["method"]})',
                            "text": row_to_text(record["headers"], record["values"]),
                        })
                    if sources:
                        with st.expander("🔎 Sources and evidence", expanded=True):
                            for source in sources[:10]:
                                st.markdown(f'**{source["label"]}**')
                                st.text(source["text"][:1800])
                    st.markdown(answer)
                else:
                    retrieved = shared_retrieve_context(st.session_state.vector_store,question,getattr(st.session_state, "documents", []) or [],top_k,)
                    # Temporary retrieval debugging: inspect exactly what reaches the LLM.
                    with st.expander("🛠️ Debug: Retrieved document chunks", expanded=True):
                        st.write(f"Retrieved chunks: {len(retrieved)}")
                        if not retrieved:
                            st.warning("No document chunks were retrieved.")
                        else:
                            for i, doc in enumerate(retrieved, start=1):
                                st.markdown(f"**Chunk {i} — {format_source(doc.metadata)}**")
                                st.text(doc.page_content)
                    answer = generate_answer(
                        question,
                        retrieved,
                        st.session_state.messages[:-1],
                        llm_model,
                        ollama_url,
                    )
                    sources = [
                        {"label": format_source(doc.metadata), "text": doc.page_content}
                        for doc in retrieved
                    ]
                    st.markdown(answer)
                    if sources:
                        with st.expander("🔎 Sources and evidence", expanded=True):
                            for source in sources:
                                st.markdown(f'**{source["label"]}**')
                                st.text(source["text"][:1800])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources if 'sources' in locals() else [],
        })
else:
    st.info("Upload one or more documents, then click **Process / Rebuild index** to start asking questions.")

st.divider()
st.caption("Notes: scanned PDFs require OCR; XLSX formula cells need saved cached values to show calculated results. Verify important answers against the displayed source evidence.")
