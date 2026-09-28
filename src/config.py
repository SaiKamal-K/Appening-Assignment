"""
Configuration module for the RAG Agentic AI Chatbot.

Loads environment variables and defines project-wide constants.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- Pinecone Configuration ---
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
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
    global OPENAI_API_KEY, PINECONE_API_KEY
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Please set them in your .env file. See .env.example for reference."
        )

