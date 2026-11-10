"""Embedding model construction."""

from .config import EMBEDDING_MODEL_NAME


def get_embeddings(model_name: str = EMBEDDING_MODEL_NAME):
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)
