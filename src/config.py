import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Model configuration
MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Vector store configuration
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store")

# Embedding model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM configuration
LLM_CONFIG = {
    "n_ctx": 4096,
    "n_batch": 512,
    "n_threads": 8,
    "n_gpu_layers": 0
}

# System prompt
SYSTEM_PROMPT = """You are a helpful AI assistant specialized in Icelandic law. 
You provide accurate, clear, and concise answers based on the provided context. 
Always cite your sources when possible."""

# Required Python packages
REQUIRED_PACKAGES = [
    "requests",
    "langchain",
    "langchain_community",
    "sentence_transformers",
    "faiss",
    "bs4",
    "tqdm",
    "llama-cpp-python"
] 