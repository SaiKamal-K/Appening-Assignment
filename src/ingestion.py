"""
Document Ingestion Pipeline for the RAG Agentic AI Chatbot.

This module handles:
1. Loading the PDF document using PyPDFLoader.
2. Splitting the document into manageable text chunks using RecursiveCharacterTextSplitter.
3. Creating a Pinecone index (if it doesn't already exist).
4. Generating OpenAI embeddings for each chunk and upserting them into Pinecone.
"""

import os
import sys
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import (
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION,
    PINECONE_METRIC,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_PATH,
    validate_config,
    get_config_val,
)


def load_pdf(pdf_path: str):
    """
    Load a PDF file and return a list of Document objects (one per page).

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of LangChain Document objects.

    Raises:
        FileNotFoundError: If the PDF file does not exist at the given path.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at '{pdf_path}'. "
            f"Please download it and place it under data/Ebook-Agentic-AI.pdf"
        )
    print(f"[Ingestion] Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"[Ingestion] Loaded {len(docs)} pages from the PDF.")
    return docs


def chunk_documents(docs, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """
    Split documents into smaller text chunks for embedding.

    Args:
        docs: List of LangChain Document objects.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    print(f"[Ingestion] Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[Ingestion] Created {len(chunks)} text chunks.")
    return chunks


def ensure_pinecone_index(index_name: str = PINECONE_INDEX_NAME):
    """
    Create the Pinecone index if it does not already exist.

    Uses a serverless spec on AWS us-east-1 (free tier compatible).

    Args:
        index_name: Name of the Pinecone index to create or verify.
    """
    pinecone_key = get_config_val("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_key)
    
    # Safely extract existing index names across different Pinecone SDK versions
    raw_indexes = pc.list_indexes()
    if hasattr(raw_indexes, "names"):
        existing_indexes = raw_indexes.names()
    else:
        existing_indexes = [
            idx.name if hasattr(idx, "name") else (idx["name"] if isinstance(idx, dict) else str(idx))
            for idx in raw_indexes
        ]

    if index_name in existing_indexes:
        print(f"[Ingestion] Pinecone index '{index_name}' already exists.")
        return

    print(f"[Ingestion] Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=PINECONE_DIMENSION,
        metric=PINECONE_METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for index to be ready (safe for both dict and IndexStatus object models)
    while True:
        status = pc.describe_index(index_name).status
        is_ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
        if is_ready:
            break
        print("[Ingestion] Waiting for index to become ready...")
        time.sleep(2)
    print(f"[Ingestion] Pinecone index '{index_name}' is ready.")


def upsert_to_pinecone(chunks, index_name: str = PINECONE_INDEX_NAME):
    """
    Generate embeddings for document chunks and upsert them into Pinecone.

    Args:
        chunks: List of LangChain Document objects (text chunks).
        index_name: Name of the target Pinecone index.

    Returns:
        PineconeVectorStore instance connected to the populated index.
    """
    print(f"[Ingestion] Generating embeddings and upserting {len(chunks)} chunks to Pinecone...")
    openai_key = get_config_val("OPENAI_API_KEY")
    pinecone_key = get_config_val("PINECONE_API_KEY")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=openai_key)
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=pinecone_key,
    )
    print("[Ingestion] Upsert complete.")
    return vector_store


def run_ingestion(pdf_path: str = PDF_PATH, index_name: str = PINECONE_INDEX_NAME):
    """
    Execute the full ingestion pipeline end-to-end:
        1. Validate configuration.
        2. Load the PDF.
        3. Chunk the document text.
        4. Ensure the Pinecone index exists.
        5. Upsert embeddings into Pinecone.

    Args:
        pdf_path: Path to the source PDF file.
        index_name: Pinecone index name.

    Returns:
        PineconeVectorStore instance.
    """
    validate_config()

    # Step 1: Load PDF
    docs = load_pdf(pdf_path)

    # Step 2: Chunk documents
    chunks = chunk_documents(docs)

    # Step 3: Ensure Pinecone index exists
    ensure_pinecone_index(index_name)

    # Step 4: Upsert embeddings
    vector_store = upsert_to_pinecone(chunks, index_name)

    print("[Ingestion] ✅ Ingestion pipeline completed successfully.")
    return vector_store


# --- CLI Entry Point ---
if __name__ == "__main__":
    run_ingestion()
