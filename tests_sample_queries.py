"""
Sample Test Queries for the RAG Agentic AI Chatbot.

This script runs 6 benchmark queries against the compiled LangGraph RAG pipeline
and prints the results (answer, retrieved context chunks, and confidence score)
for manual verification of response quality and document grounding.

Usage:
    python tests_sample_queries.py
"""

import json
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import PINECONE_INDEX_NAME, validate_config
from src.graph import build_rag_graph


# ---------------------------------------------------------------------------
# Benchmark Queries
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = [
    {
        "id": 1,
        "query": "What is Agentic AI according to the eBook?",
        "expected_behavior": "Should provide a definition/explanation from the eBook.",
    },
    {
        "id": 2,
        "query": "How do AI agents differ from traditional automation systems?",
        "expected_behavior": "Should contrast AI agents with traditional automation based on eBook content.",
    },
    {
        "id": 3,
        "query": "What are the core components of an Agentic Architecture?",
        "expected_behavior": "Should list and describe architectural components from the eBook.",
    },
    {
        "id": 4,
        "query": "What role does memory play in Agentic AI workflows?",
        "expected_behavior": "Should explain the role of memory as described in the eBook.",
    },
    {
        "id": 5,
        "query": "What are the key challenges in deploying Agentic AI systems?",
        "expected_behavior": "Should discuss deployment challenges mentioned in the eBook.",
    },
    {
        "id": 6,
        "query": "Who won the 2022 FIFA World Cup?",
        "expected_behavior": (
            "VALIDATION TEST — The chatbot should refuse to answer or state that "
            "the provided document does not contain this information."
        ),
    },
]


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------

def run_tests():
    """Execute all sample queries and print formatted results."""
    try:
        validate_config()
    except EnvironmentError as e:
        print(f"\n❌ Configuration Error: {e}\n")
        return 1

    print("=" * 80)
    print("  RAG Agentic AI Chatbot — Sample Query Tests")
    print("=" * 80)
    print()

    # Build graph once
    print("[Test] Building RAG graph...")
    try:
        graph = build_rag_graph(index_name=PINECONE_INDEX_NAME)
        print("[Test] Graph compiled successfully.\n")
    except Exception as e:
        print(f"\n❌ Failed to build RAG graph: {e}\n")
        return 1

    results = []

    for test_case in SAMPLE_QUERIES:
        query_id = test_case["id"]
        query = test_case["query"]
        expected = test_case["expected_behavior"]

        print(f"{'─' * 80}")
        print(f"  Query #{query_id}: {query}")
        print(f"  Expected: {expected}")
        print(f"{'─' * 80}")

        # Invoke the RAG pipeline
        initial_state = {
            "question": query,
            "context": [],
            "answer": "",
            "score": 0.0,
        }

        try:
            result = graph.invoke(initial_state)

            answer = result["answer"]
            context = result["context"]
            score = result["score"]

            print(f"\n  📝 Answer:\n    {answer}\n")
            print(f"  📊 Confidence Score: {score:.2f}")
            print(f"  📄 Retrieved Chunks ({len(context)}):")
            for i, chunk in enumerate(context, 1):
                preview = chunk[:200].replace("\n", " ")
                print(f"    [{i}] {preview}...")
            print()

            results.append({
                "query_id": query_id,
                "query": query,
                "answer": answer,
                "confidence_score": score,
                "num_chunks_retrieved": len(context),
                "status": "PASS",
            })

        except Exception as e:
            print(f"\n  ❌ ERROR: {e}\n")
            results.append({
                "query_id": query_id,
                "query": query,
                "error": str(e),
                "status": "FAIL",
            })

    # Summary
    print("=" * 80)
    print("  TEST SUMMARY")
    print("=" * 80)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"  Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print()

    for r in results:
        status_icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"  {status_icon} Query #{r['query_id']}: {r['query'][:60]}")
        if r["status"] == "PASS":
            print(f"      Score: {r['confidence_score']:.2f} | Chunks: {r['num_chunks_retrieved']}")
        else:
            print(f"      Error: {r.get('error', 'Unknown')}")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
