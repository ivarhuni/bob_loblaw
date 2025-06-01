import os
from langchain_community.vectorstores import FAISS
from .config import VECTOR_STORE_PATH
from .embeddings import get_embeddings

def get_vector_store():
    """Initialize and return the vector store."""
    embeddings = get_embeddings()
    
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings)
    else:
        return None

def save_vector_store(vector_store):
    """Save the vector store to disk."""
    if not os.path.exists(os.path.dirname(VECTOR_STORE_PATH)):
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH))
    vector_store.save_local(VECTOR_STORE_PATH) 