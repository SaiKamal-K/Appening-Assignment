"""
FastAPI Application for the RAG Agentic AI Chatbot.

Exposes a POST /chat endpoint that accepts a user query,
invokes the LangGraph RAG workflow, and returns:
- The grounded answer
- The retrieved context chunks
- A confidence score
"""

import os
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import PINECONE_INDEX_NAME, validate_config
from src.graph import build_rag_graph


# ---------------------------------------------------------------------------
# App Initialization & Lifespan
# ---------------------------------------------------------------------------

_rag_graph = None


def get_graph():
    """Retrieve or initialize the compiled RAG workflow graph."""
    global _rag_graph
    if _rag_graph is None:
        validate_config()
        _rag_graph = build_rag_graph(index_name=PINECONE_INDEX_NAME)
    return _rag_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the RAG graph during application startup if credentials are set."""
    try:
        get_graph()
    except Exception as e:
        print(f"[App] RAG graph pre-warming deferred (will initialize on first request): {e}")
    yield


app = FastAPI(
    title="Agentic AI RAG Chatbot API",
    description=(
        "A Retrieval-Augmented Generation (RAG) chatbot API that answers questions "
        "strictly grounded in the Agentic AI eBook. Built with LangGraph, Pinecone, "
        "and OpenAI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS for local development / Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Schema for the incoming chat request."""
    query: str = Field(
        ...,
        min_length=1,
        description="The user's question to ask the RAG chatbot.",
        examples=["What is Agentic AI according to the eBook?"],
    )


class QueryResponse(BaseModel):
    """Schema for the chat response."""
    final_answer: str = Field(
        ..., description="The LLM-generated answer grounded in the retrieved context."
    )
    retrieved_context: list[str] = Field(
        ..., description="The document chunks retrieved from Pinecone."
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence / relevance score (0.0 – 1.0)."
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Agentic AI RAG Chatbot API",
        "version": "1.0.0",
        "graph_initialized": _rag_graph is not None,
    }


@app.post("/chat", response_model=QueryResponse, tags=["Chat"])
async def chat_endpoint(request: QueryRequest):
    """
    Accept a user query, run the RAG pipeline, and return a grounded answer
    along with the retrieved context chunks and a confidence score.
    """
    try:
        graph = get_graph()
    except EnvironmentError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Configuration error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize RAG workflow: {str(e)}",
        )

    try:
        initial_state = {
            "question": request.query,
            "context": [],
            "answer": "",
            "score": 0.0,
        }
        result = graph.invoke(initial_state)

        return QueryResponse(
            final_answer=result["answer"],
            retrieved_context=result["context"],
            confidence_score=result["score"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your query: {str(e)}",
        )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
    )
