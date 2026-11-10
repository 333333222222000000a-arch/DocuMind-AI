"""Deterministic retrieval and citation metrics, independent of LLM generation."""

import re


def _chunk_id(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("chunk_id") or item.get("metadata", {}).get("chunk_id")
    return getattr(item, "metadata", {}).get("chunk_id")


def precision_at_k(ranked_ids, relevant_ids, k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be a positive integer")
    relevant = set(relevant_ids)
    return sum(chunk_id in relevant for chunk_id in ranked_ids[:k]) / k


def recall_at_k(ranked_ids, relevant_ids, k: int = 10):
    relevant = set(relevant_ids)
    if not relevant:
        return None
    return sum(chunk_id in relevant for chunk_id in ranked_ids[:k]) / len(relevant)


def reciprocal_rank(ranked_ids, relevant_ids) -> float:
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(ranked_ids, 1):
        if chunk_id in relevant:
            return 1 / rank
    return 0.0


def unsupported_numeric_claims(answer: str, evidence: str):
    """Return answer numerals absent from retrieved evidence (a narrow heuristic)."""
    answer_without_citations = re.sub(r"\[S\d+\]", " ", answer, flags=re.IGNORECASE)
    answer_numbers = set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?", answer_without_citations))
    evidence_numbers = set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?", evidence))
    return sorted(answer_numbers - evidence_numbers)


def evaluate_retrieval_case(case, ranked_chunks, k: int = 10):
    ranked = [_chunk_id(chunk) for chunk in ranked_chunks]
    relevant = set(case.get("relevant_chunk_ids", []))
    source_by_id = {chunk["chunk_id"]: (chunk.get("source"), chunk.get("page"))
                    for chunk in case.get("chunks", [])}
    expected_citations = {source_by_id[cid] for cid in relevant if cid in source_by_id}
    retrieved_citations = {source_by_id[cid] for cid in ranked[:k] if cid in source_by_id}
    citation_precision = (
        len(expected_citations & retrieved_citations) / len(retrieved_citations)
        if retrieved_citations else (1.0 if not expected_citations else 0.0)
    )
    citation_recall = (
        len(expected_citations & retrieved_citations) / len(expected_citations)
        if expected_citations else None
    )
    return {
        "case_id": case["id"],
        "precision_at_k": precision_at_k(ranked, relevant, k),
        "recall_at_k": recall_at_k(ranked, relevant, k),
        "mrr": reciprocal_rank(ranked, relevant),
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "unanswerable_retrieval": not relevant and bool(ranked[:k]),
    }


def evaluate_retrieval_cases(cases, retrieve, k: int = 10):
    """Evaluate a callable ``retrieve(question, chunks)`` over labeled cases."""
    results = [
        evaluate_retrieval_case(case, retrieve(case["question"], case.get("chunks", [])), k)
        for case in cases
    ]
    answerable = [result for result in results if result["recall_at_k"] is not None]
    unanswerable = [result for result in results if result["recall_at_k"] is None]

    def mean(key, rows):
        return sum(row[key] for row in rows) / len(rows) if rows else None

    return {
        "k": k,
        "case_results": results,
        "mean_precision_at_k": mean("precision_at_k", results),
        "mean_recall_at_k": mean("recall_at_k", answerable),
        "mrr": mean("mrr", answerable),
        "mean_citation_precision": mean("citation_precision", results),
        "mean_citation_recall": mean("citation_recall", answerable),
        "unanswerable_retrieval_rate": mean(
            "value",
            [{"value": float(row["unanswerable_retrieval"])} for row in unanswerable],
        ) if unanswerable else None,
    }
