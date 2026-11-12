"""LangChain answer-generation chain construction and invocation."""

import json
import math
import re
import unicodedata

from .config import (
    MAX_PROMPT_TOKEN_BUDGET,
    PROMPT_TEMPLATE,
    PROMPT_TOKEN_SAFETY_MARGIN,
)
from .citations import display_page_number
from .vector_store import deduplicate_ranked_documents


_INVISIBLE_CITATION_CHARS = "\u200b\u200c\u200d\u2060\ufeff"
_CITATION_GAP = rf"[\s{_INVISIBLE_CITATION_CHARS}]*"
_CITATION_PATTERN = re.compile(
    rf"(?<![\w])(?:\[{_CITATION_GAP}S{_CITATION_GAP}(\d+){_CITATION_GAP}\]"
    rf"|【{_CITATION_GAP}S{_CITATION_GAP}(\d+){_CITATION_GAP}】"
    rf"|\({_CITATION_GAP}S{_CITATION_GAP}(\d+){_CITATION_GAP}\))",
    re.IGNORECASE,
)
_CITATION_CANDIDATE_PATTERN = re.compile(
    rf"(?:\[|【|\(){_CITATION_GAP}S{_CITATION_GAP}[:#-]?{_CITATION_GAP}\d+{_CITATION_GAP}(?:\]|】|\))",
    re.IGNORECASE,
)
_NO_CITATION_ANSWER = (
    "I could not verify a source-cited answer from the retrieved passages. "
    "Please rephrase the question or ask for a specific detail."
)
_ARABIC_QUERY_STOPWORDS = {
    "انا", "انت", "انتم", "هو", "هي", "هم", "هن", "هذا", "هذه", "ذلك", "تلك",
    "من", "في", "على", "عن", "الى", "الي", "مع", "ما", "ماذا", "كيف", "هل",
    "و", "او", "ثم", "ب", "ل", "اللي", "حيث", "عاوز", "عايز", "اريد", "اعرف",
}
_ENGLISH_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "these", "this",
    "to", "what", "when", "where", "which", "with",
}
_QUERY_STOPWORDS = _ARABIC_QUERY_STOPWORDS | _ENGLISH_QUERY_STOPWORDS
_ARABIC_TECHNICAL_ALIASES = {
    "معالج": "__cpu__", "بروسيسور": "__cpu__", "processor": "__cpu__", "cpu": "__cpu__",
    "رام": "__memory__", "رامات": "__memory__", "ذاكرة": "__memory__", "ذاكره": "__memory__",
    "memory": "__memory__", "ram": "__memory__",
    "كارت": "__gpu__", "كرت": "__gpu__", "شاشة": "__gpu__", "graphics": "__gpu__", "gpu": "__gpu__",
    "تخزين": "__storage__", "قرص": "__storage__", "storage": "__storage__", "ssd": "__storage__", "nvme": "__storage__",
}
_CONTEXT_MMR_RELEVANCE_WEIGHT = 0.75
_CONTEXT_MMR_DIVERSITY_WEIGHT = 0.25
_LEXICAL_RELEVANCE_WEIGHT = 0.8
_FAISS_RANK_WEIGHT = 0.2
_BM25_K1 = 1.5
_BM25_B = 0.75
_CITATION_REPAIR_TEMPLATE = """
Review the draft answer against the supplied document context and return a corrected answer.
Use only facts supported by that context. Attach exact citations like [S1] or [S2]
to every factual claim, using only the source IDs shown in the context. Do not invent
source IDs or add facts. If the context does not support the answer, respond with
NO_ANSWER: followed by a brief explanation. Answer in the same language as the question.

Document context:
{context}

Question: {input}

Draft answer to verify:
{draft}

Allowed source IDs: {allowed_source_ids}

Corrected answer:
""".strip()


def _context_passage(document, citation_number: int) -> str:
    source = str(document.metadata.get("source") or "filename unavailable")
    source = source.replace("\r", " ").replace("\n", " ")
    page = display_page_number(document.metadata)
    page_label = str(page) if page is not None else "not available"
    return f"[S{citation_number}] Source: {source} | Page: {page_label}\n{document.page_content}"


def _citation_number(match):
    return int(next(value for value in match.groups() if value is not None))


def _citation_diagnostics(
    references, valid, rejected, document_count, candidate_count, reason, *, abstained=False,
):
    unique_references = sorted(set(references))
    rejected_references = unique_references if abstained else rejected
    rejection_reason = (
        "ignored because the model explicitly abstained"
        if abstained else f"outside selected range S1-S{document_count}"
    )
    rejected_ids = [
        {"id": f"S{value}", "reason": rejection_reason}
        for value in rejected_references
    ]
    unparsed_candidate_count = max(0, candidate_count - len(references))
    if unparsed_candidate_count:
        rejected_ids.append({
            "id": "unparsed citation marker",
            "reason": (
                "ignored because the model explicitly abstained"
                if abstained else "malformed or unsupported citation marker syntax"
            ),
        })
    return {
        "extracted_ids": [f"S{value}" for value in unique_references],
        "valid_ids": [] if abstained else [f"S{value}" for value in valid],
        "rejected_ids": rejected_ids,
        "citation_candidate_count": candidate_count,
        "unparsed_candidate_count": unparsed_candidate_count,
        "validation_reason": reason,
        "abstained": abstained,
    }


def _citation_validation(answer, documents):
    """Validate response markers against this attempt's selected context only."""
    answer = str(answer or "")
    normalized = unicodedata.normalize("NFKC", answer)
    no_answer = re.match(r"^NO_ANSWER\s*:\s*", normalized.lstrip(), re.IGNORECASE)
    candidate_count = len(_CITATION_CANDIDATE_PATTERN.findall(normalized))
    matches = list(_CITATION_PATTERN.finditer(normalized))
    references = [_citation_number(match) for match in matches]
    valid = sorted({value for value in references if 1 <= value <= len(documents)})
    rejected = sorted({value for value in references if value < 1 or value > len(documents)})

    if no_answer:
        answer_body = normalized.lstrip()[no_answer.end():].strip()
        answer_body = _CITATION_CANDIDATE_PATTERN.sub(" ", answer_body)
        answer_body = re.sub(r"[ \t]{2,}", " ", answer_body)
        answer_body = re.sub(r"\s+([.,;:!?،؛])", r"\1", answer_body).strip()
        return answer_body, [], _citation_diagnostics(
            references, valid, rejected, len(documents), candidate_count,
            "model explicitly abstained", abstained=True,
        )

    # NFKC handles compatibility punctuation; the pattern tolerates invisible
    # formatting characters inside a marker. The returned answer uses this
    # normalized representation so displayed markers match validated markers.
    unparsed_candidate_count = max(0, candidate_count - len(references))
    if not valid or rejected or unparsed_candidate_count:
        if not references:
            reason = (
                "malformed or unsupported citation marker syntax"
                if unparsed_candidate_count else "no supported citation marker detected"
            )
        elif unparsed_candidate_count:
            reason = "malformed or unsupported citation marker syntax"
        elif rejected and valid:
            reason = "one or more citation IDs are outside selected context"
        else:
            reason = "all citation IDs are outside selected context"
        return _NO_CITATION_ANSWER, [], _citation_diagnostics(
            references, valid, rejected, len(documents), candidate_count, reason,
        )

    cited_documents = []
    for index in valid:
        document = documents[index - 1]
        cited_documents.append(document.__class__(
            page_content=document.page_content,
            metadata={**document.metadata, "citation_id": f"S{index}"},
        ))
    normalized = _CITATION_PATTERN.sub(
        lambda match: f"[S{_citation_number(match)}]"
        if _citation_number(match) in valid else "",
        normalized,
    )
    reason = (
        "valid source IDs found"
        if not rejected else "some citation IDs are outside selected context"
    )
    return normalized.strip(), cited_documents, _citation_diagnostics(
        references, valid, rejected, len(documents), candidate_count, reason,
    )


def _cited_source_documents(answer, documents):
    """Backward-compatible citation validator returning answer and sources."""
    parsed_answer, cited_documents, _ = _citation_validation(answer, documents)
    return parsed_answer, cited_documents


def _log_citation_diagnostics(logger, phase, response, document_ids, prompt_ids, validation):
    if not logger:
        return
    details = {
        "phase": phase,
        "selected_source_ids": document_ids,
        "source_ids_present_in_prompt_context": prompt_ids,
        "all_source_ids_present": document_ids == prompt_ids,
        "response_type": type(response).__name__,
        "response_length": len(str(response or "")),
        "detected_citation_markers": validation["extracted_ids"],
        "citation_candidate_count": validation["citation_candidate_count"],
        "unparsed_candidate_count": validation["unparsed_candidate_count"],
        "validation_reason": validation["validation_reason"],
        "extracted_citation_ids": validation["extracted_ids"],
        "valid_ids": validation["valid_ids"],
        "rejected_ids": validation["rejected_ids"],
        "abstained": validation["abstained"],
    }
    logger("Citation diagnostics " + json.dumps(details, ensure_ascii=True, separators=(",", ":")))


def estimate_prompt_token_upper_bound(
    documents,
    question: str,
    prompt_template: str = PROMPT_TEMPLATE,
    safety_margin: int = PROMPT_TOKEN_SAFETY_MARGIN,
    additional_values=None,
) -> int:
    """Conservatively estimate tokens using UTF-8 bytes plus message overhead.

    This dependency-free upper bound avoids downloading an unrelated tokenizer
    at runtime and is intentionally conservative for Arabic text.
    """
    if safety_margin < 0:
        raise ValueError("safety_margin must be non-negative")
    context = "\n\n".join(
        _context_passage(document, index)
        for index, document in enumerate(documents, 1)
    )
    prompt_values = {"context": context, "input": question}
    if additional_values:
        prompt_values.update(additional_values)
    prompt = prompt_template.format(**prompt_values)
    return len(prompt.encode("utf-8")) + safety_margin


def _retrieval_tokens(text):
    """Normalize Arabic/English words for a lightweight deterministic BM25 pass."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = normalized.replace("ـ", "")
    normalized = re.sub(r"[\u064b-\u065f\u0670]", "", normalized)
    normalized = normalized.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    # PDF tables often glue Arabic labels to Latin model names (e.g. المعالجIntel).
    normalized = re.sub(r"(?<=[\u0600-\u06ff])(?=[a-z0-9])", " ", normalized)
    normalized = re.sub(r"(?<=[a-z0-9])(?=[\u0600-\u06ff])", " ", normalized)
    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    normalized_tokens = []
    for token in tokens:
        if token.startswith("و") and len(token) > 4:
            token = token[1:]
        if token.startswith("ال") and len(token) > 5:
            token = token[2:]
        token = _ARABIC_TECHNICAL_ALIASES.get(token, token)
        if token not in _QUERY_STOPWORDS and len(token) > 1:
            normalized_tokens.append(token)
    return normalized_tokens


def _bm25_relevance_scores(documents, question):
    """Return normalized lexical relevance without changing FAISS's rank order."""
    query_terms = set(_retrieval_tokens(question))
    tokenized = [_retrieval_tokens(document.page_content) for document in documents]
    if not query_terms or not tokenized:
        return [0.0] * len(documents)

    document_frequency = {
        term: sum(term in set(tokens) for tokens in tokenized)
        for term in query_terms
    }
    average_length = sum(map(len, tokenized)) / len(tokenized) or 1.0
    raw_scores = []
    for tokens in tokenized:
        frequencies = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        length = len(tokens)
        score = 0.0
        for term in query_terms:
            tf = frequencies.get(term, 0)
            if not tf:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log1p(
                (len(documents) - df + 0.5) / (df + 0.5)
            )
            denominator = tf + _BM25_K1 * (
                1 - _BM25_B + _BM25_B * length / average_length
            )
            score += inverse_document_frequency * tf * (_BM25_K1 + 1) / denominator
        raw_scores.append(score)

    maximum = max(raw_scores, default=0.0)
    if maximum <= 0:
        return [0.0] * len(documents)
    return [score / maximum for score in raw_scores]


def _rank_relevance_scores(documents, question):
    """Blend normalized query term evidence with the unchanged FAISS rank prior."""
    lexical_scores = _bm25_relevance_scores(documents, question)
    if not documents:
        return []
    scores = []
    for index, lexical in enumerate(lexical_scores, 1):
        rank_prior = 1.0 / math.log2(index + 1)
        scores.append(
            _LEXICAL_RELEVANCE_WEIGHT * lexical + _FAISS_RANK_WEIGHT * rank_prior
        )
    return scores


def _chunk_term_similarity(left, right):
    left_terms = set(_retrieval_tokens(left.page_content))
    right_terms = set(_retrieval_tokens(right.page_content))
    if not left_terms or not right_terms:
        return 0.0
    # Overlap coefficient catches near-copies that add a short sentence while
    # keeping broadly related passages less penalized than exact Jaccard alone.
    return len(left_terms & right_terms) / min(len(left_terms), len(right_terms))


def select_context_documents(
    documents,
    question: str,
    prompt_template: str = PROMPT_TEMPLATE,
    max_prompt_tokens: int = MAX_PROMPT_TOKEN_BUDGET,
):
    """Select a budgeted, relevance-aware diverse subset of ranked FAISS results.

    FAISS/MMR output order is retained as the rank prior and final order. A
    second deterministic MMR pass uses BM25 query evidence to prevent long,
    weakly matching early passages from exhausting the prompt budget before a
    concise, directly relevant passage can be selected.
    """
    if not documents:
        return []
    relevance = _rank_relevance_scores(documents, question)
    selected_indices = []
    remaining = set(range(len(documents)))

    while remaining:
        feasible = []
        for index in remaining:
            candidate_indices = sorted([*selected_indices, index])
            candidate = [documents[position] for position in candidate_indices]
            if estimate_prompt_token_upper_bound(
                candidate, question, prompt_template
            ) <= max_prompt_tokens:
                redundancy = max(
                    (_chunk_term_similarity(documents[index], documents[chosen])
                     for chosen in selected_indices),
                    default=0.0,
                )
                if redundancy >= 0.9:
                    continue
                marginal_score = (
                    _CONTEXT_MMR_RELEVANCE_WEIGHT * relevance[index]
                    - _CONTEXT_MMR_DIVERSITY_WEIGHT * redundancy
                )
                feasible.append((marginal_score, relevance[index], -index, index))
        if not feasible:
            break
        chosen = max(feasible)[-1]
        selected_indices.append(chosen)
        remaining.remove(chosen)

    selected_indices.sort()
    return [documents[index] for index in selected_indices]


def answer_question(
    retriever,
    llm,
    question: str,
    prompt_template: str = PROMPT_TEMPLATE,
    *,
    max_prompt_tokens: int = MAX_PROMPT_TOKEN_BUDGET,
    logger=None,
):
    retrieved_documents = retriever.invoke(question)
    if not retrieved_documents:
        return "The uploaded documents do not contain relevant information to answer this question.", []

    unique_documents = deduplicate_ranked_documents(retrieved_documents)

    documents = select_context_documents(
        unique_documents,
        question,
        prompt_template,
        max_prompt_tokens,
    )
    estimated_tokens = estimate_prompt_token_upper_bound(documents, question, prompt_template)
    if logger:
        logger(
            f"Retrieval returned {len(retrieved_documents)} chunks; "
            f"duplicate filtering kept {len(unique_documents)}; "
            f"context selection kept {len(documents)}; "
            f"prompt token upper-bound estimate {estimated_tokens}/{max_prompt_tokens}"
        )
    if not documents:
        return "Relevant passages were found but exceeded the safe prompt budget. Please ask a more focused question.", []

    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    prompted_documents = [
        document.__class__(
            page_content=_context_passage(document, index),
            metadata=dict(document.metadata),
        )
        for index, document in enumerate(documents, 1)
    ]
    source_ids = [f"S{index}" for index in range(1, len(documents) + 1)]
    prompt_source_ids = [
        source_id for source_id, document in zip(source_ids, prompted_documents)
        if document.page_content.startswith(f"[{source_id}]")
    ]
    answer = document_chain.invoke({"context": prompted_documents, "input": question})
    answer = answer or ""
    parsed_answer, cited_documents, validation = _citation_validation(answer, documents)
    _log_citation_diagnostics(
        logger, "initial", answer, source_ids, prompt_source_ids, validation,
    )
    if cited_documents or re.match(r"^NO_ANSWER\s*:", answer.lstrip(), re.IGNORECASE):
        if logger:
            logger(f"Citation validation: {len(cited_documents)} valid source IDs on initial response")
        return parsed_answer, cited_documents

    if logger:
        logger("Citation validation failed on initial response; checking correction prompt budget")
    repair_values = {
        "draft": answer,
        "allowed_source_ids": ", ".join(source_ids),
    }
    repair_estimate = estimate_prompt_token_upper_bound(
        documents,
        question,
        _CITATION_REPAIR_TEMPLATE,
        additional_values=repair_values,
    )
    if repair_estimate > max_prompt_tokens:
        if logger:
            logger(
                "Citation correction skipped because its prompt estimate exceeds the configured budget "
                f"({repair_estimate}/{max_prompt_tokens})"
            )
        return _NO_CITATION_ANSWER, []

    if logger:
        logger(f"Citation correction prompt estimate {repair_estimate}/{max_prompt_tokens}; requesting one correction")
    repair_prompt = ChatPromptTemplate.from_template(_CITATION_REPAIR_TEMPLATE)
    repair_chain = create_stuff_documents_chain(llm, repair_prompt)
    corrected = repair_chain.invoke({
        "context": prompted_documents,
        "input": question,
        **repair_values,
    })
    corrected = corrected or ""
    corrected_answer, cited_documents, validation = _citation_validation(corrected, documents)
    _log_citation_diagnostics(
        logger, "correction", corrected, source_ids, prompt_source_ids, validation,
    )
    if logger:
        logger("Citation correction inputs " + json.dumps({
            "allowed_source_ids": source_ids,
            "allowed_ids_passed_to_chain": ", ".join(source_ids),
            "all_context_ids_present": source_ids == prompt_source_ids,
        }, separators=(",", ":")))
    if logger:
        logger(f"Citation validation after correction: {len(cited_documents)} valid source IDs")
    if not cited_documents:
        return _NO_CITATION_ANSWER, []
    return corrected_answer, cited_documents
