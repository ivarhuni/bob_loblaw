from langchain_community.llms import LlamaCpp
from .config import MODEL_PATH, LLM_CONFIG, SYSTEM_PROMPT

def get_llm():
    """Initialize and return the LLM model."""
    return LlamaCpp(
        model_path=MODEL_PATH,
        system_prompt=SYSTEM_PROMPT,
        **LLM_CONFIG
    ) 