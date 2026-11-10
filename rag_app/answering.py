"""LangChain answer-generation chain construction and invocation."""

import re

from .config import (
    MAX_PROMPT_TOKEN_BUDGET,
    PROMPT_TEMPLATE,
    PROMPT_TOKEN_SAFETY_MARGIN,
)
from .citations import display_page_number
from .vector_store import deduplicate_ranked_documents


def _context_passage(document, citation_number: int) -> str:
    source = str(document.metadata.get("source") or "filename unavailable")
    source = source.replace("\r", " ").replace("\n", " ")
    page = display_page_number(document.metadata)
    page_label = str(page) if page is not None else "not available"
    return f"[S{citation_number}] Source: {source} | Page: {page_label}\n{document.page_content}"


def _cited_source_documents(answer, documents):
    no_answer = re.match(r"^NO_ANSWER\s*:\s*", answer.lstrip(), re.IGNORECASE)
    if no_answer:
        return answer.lstrip()[no_answer.end():].strip(), []

    references = [int(value) for value in re.findall(r"\[S(\d+)\]", answer, re.IGNORECASE)]
    valid = sorted({value for value in references if 1 <= value <= len(documents)})
    if not valid:
        return (
            "I could not verify a source-cited answer from the retrieved passages. "
            "Please rephrase the question or ask for a specific detail.",
            [],
        )

    cited_documents = []
    for index in valid:
        document = documents[index - 1]
        cited_documents.append(document.__class__(
            page_content=document.page_content,
            metadata={**document.metadata, "citation_id": f"S{index}"},
        ))
    answer = re.sub(
        r"\[S(\d+)\]",
        lambda match: match.group(0) if int(match.group(1)) in valid else "",
        answer,
        flags=re.IGNORECASE,
    )
    return answer.strip(), cited_documents


def estimate_prompt_token_upper_bound(
    documents,
    question: str,
    prompt_template: str = PROMPT_TEMPLATE,
    safety_margin: int = PROMPT_TOKEN_SAFETY_MARGIN,
) -> int:
    """Conservatively estimate tokens using UTF-8 bytes plus message overhead.

    This dependency-free upper bound avoids downloading an unrelated tokenizer
    at runtime and is intentionally conservative for Arabic text.
    """
    context = "\n\n".join(
        _context_passage(document, index)
        for index, document in enumerate(documents, 1)
    )
    prompt = prompt_template.format(context=context, input=question)
    return len(prompt.encode("utf-8")) + safety_margin


def select_context_documents(
    documents,
    question: str,
    prompt_template: str = PROMPT_TEMPLATE,
    max_prompt_tokens: int = MAX_PROMPT_TOKEN_BUDGET,
):
    """Keep the highest-ranked diverse chunks that fit the prompt budget."""
    selected = []
    for document in documents:
        candidate = [*selected, document]
        if estimate_prompt_token_upper_bound(candidate, question, prompt_template) <= max_prompt_tokens:
            selected.append(document)
    return selected


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
    answer = document_chain.invoke({"context": prompted_documents, "input": question})
    answer = answer or "No answer generated"
    return _cited_source_documents(answer, documents)
