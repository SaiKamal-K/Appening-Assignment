"""
LangGraph RAG Workflow for the Agentic AI Chatbot.

This module defines:
- AgentState: The typed state schema flowing through the graph.
- retrieve_node: Queries Pinecone for the top-k relevant document chunks.
- generate_node: Uses the retrieved context to produce a grounded answer via the LLM.
- build_rag_graph(): Assembles and compiles the LangGraph StateGraph.
"""

import os
from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import (
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVER_TOP_K,
    validate_config,
    get_config_val,
)


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """State that flows through each node of the RAG graph."""
    question: str          # The user's input query
    context: List[str]     # Retrieved document chunks
    answer: str            # Generated response
    score: float           # Confidence / relevance score


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a strict, document-grounded assistant. Your sole knowledge \
source is the "Agentic AI eBook" context provided below.

RULES:
1. Answer the user's question using ONLY the information present in the Context section.
2. If the context does not contain sufficient information to answer the question, respond \
exactly with: "I cannot answer this question based on the provided document."
3. Do NOT use any external knowledge, assumptions, or information outside the context.
4. Cite relevant details from the context to support your answer.
5. Keep answers clear, concise, and well-structured.

Context:
{context}

Question: {question}"""


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_rag_graph(index_name: str = PINECONE_INDEX_NAME):
    """
    Build and compile the LangGraph RAG workflow.

    The graph consists of two sequential nodes:
        START -> retrieve -> generate -> END

    Args:
        index_name: Name of the Pinecone index to query.

    Returns:
        A compiled LangGraph StateGraph ready for .invoke() calls.
    """
    validate_config()

    openai_key = get_config_val("OPENAI_API_KEY")
    pinecone_key = get_config_val("PINECONE_API_KEY")
    target_index = index_name or get_config_val("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME)

    # Ensure environment variables are populated for all underlying SDKs
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if pinecone_key:
        os.environ["PINECONE_API_KEY"] = pinecone_key

    # Initialize components with validated string API keys
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=openai_key,
    )
    # Guarantee sync client is present (prevents 'Sync client is not available' error)
    if not getattr(embeddings, "client", None):
        import openai
        embeddings.client = openai.OpenAI(api_key=openai_key).embeddings

    vectorstore = PineconeVectorStore(
        index_name=target_index,
        embedding=embeddings,
        pinecone_api_key=pinecone_key,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=openai_key,
    )

    # ------------------------------------------------------------------
    # Node: retrieve
    # ------------------------------------------------------------------
    def retrieve_node(state: AgentState) -> dict:
        """
        Query the Pinecone vector store for the most relevant document chunks
        matching the user's question.

        Returns:
            dict with 'context' key containing list of chunk texts.
        """
        question = state["question"]
        docs = retriever.invoke(question)
        context_texts = [doc.page_content for doc in docs]
        return {"context": context_texts}

    # ------------------------------------------------------------------
    # Node: generate
    # ------------------------------------------------------------------
    def generate_node(state: AgentState) -> dict:
        """
        Generate a grounded answer using the LLM, constrained to the
        retrieved context chunks.

        Computes a confidence heuristic:
        - 0.95 when context chunks are available and the model provides an answer.
        - 0.6 when context exists but the model indicates insufficient information / refusal.
        - 0.0 when no context was retrieved at all.

        Returns:
            dict with 'answer' and 'score' keys.
        """
        context_str = "\n\n---\n\n".join(state["context"])
        question = state["question"]

        prompt = SYSTEM_PROMPT.format(context=context_str, question=question)
        response = llm.invoke(prompt)
        answer_text = str(response.content) if not isinstance(response.content, str) else response.content

        # Refusal phrases indicating the document lacks relevant information
        refusal_markers = [
            "cannot answer",
            "not contain",
            "does not contain",
            "not mentioned",
            "no information",
            "insufficient information",
        ]
        is_refusal = any(marker in answer_text.lower() for marker in refusal_markers)

        # Confidence heuristic
        if len(state["context"]) == 0:
            confidence = 0.0
        elif is_refusal:
            confidence = 0.6  # Model flagged insufficient context in document
        else:
            confidence = round(min(0.95, 0.7 + 0.05 * len(state["context"])), 2)

        return {"answer": answer_text, "score": confidence}

    # ------------------------------------------------------------------
    # Assemble StateGraph
    # ------------------------------------------------------------------
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
