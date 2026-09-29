"""
Configuration module for the RAG Agentic AI Chatbot.

Loads environment variables and defines project-wide constants.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_config_val(key: str, default=None):
    """
    Retrieve configuration value from:
    1. os.environ (from .env or environment variables)
    2. Streamlit secrets (st.secrets) if running within Streamlit Cloud
    """
    val = os.getenv(key)
    if val:
        return val

    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            val = str(st.secrets[key])
            os.environ[key] = val
            return val
    except Exception:
        pass

    return default


# --- API Keys ---
OPENAI_API_KEY = get_config_val("OPENAI_API_KEY")
PINECONE_API_KEY = get_config_val("PINECONE_API_KEY")

# --- Pinecone Configuration ---
PINECONE_INDEX_NAME = get_config_val("PINECONE_INDEX_NAME", "agentic-ai-index")
PINECONE_DIMENSION = 1536  # Matches OpenAI text-embedding-3-small output dimension
PINECONE_METRIC = "cosine"

# --- OpenAI Model Configuration ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# --- Document Chunking Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval Configuration ---
RETRIEVER_TOP_K = 3

# --- File Paths ---
PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Ebook-Agentic-AI.pdf")


# --- Validation ---
def validate_config():
    """Validate that all required configuration values are present."""
    global OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = get_config_val("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        PINECONE_API_KEY = get_config_val("PINECONE_API_KEY")
    if not PINECONE_INDEX_NAME:
        PINECONE_INDEX_NAME = get_config_val("PINECONE_INDEX_NAME", "agentic-ai-index")

    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Please set them in your Streamlit Secrets (on Streamlit Cloud) or in your .env file."
        )


