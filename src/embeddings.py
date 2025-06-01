from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import EMBEDDING_MODEL_NAME

def get_embeddings():
    """Initialize and return the embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME) 