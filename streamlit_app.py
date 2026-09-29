"""
Streamlit UI for the RAG Agentic AI Chatbot.

Provides a lightweight chat interface with a side panel displaying
retrieved context chunks and confidence scores.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
from src.config import PINECONE_INDEX_NAME, validate_config
from src.graph import build_rag_graph


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .context-chunk {
        background-color: #f0f2f6;
        border-left: 4px solid #1E88E5;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .score-high { background-color: #C8E6C9; color: #2E7D32; }
    .score-medium { background-color: #FFF9C4; color: #F57F17; }
    .score-low { background-color: #FFCDD2; color: #C62828; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Initialize RAG Graph (cached for performance)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_rag_graph():
    """Build and cache the RAG graph so it persists across Streamlit reruns."""
    validate_config()
    return build_rag_graph(index_name=PINECONE_INDEX_NAME)


try:
    graph = get_rag_graph()
except EnvironmentError as e:
    st.error(f"⚠️ Configuration Error: {e}")
    st.warning("Missing API Keys detected. If deploying on Streamlit Cloud, add your keys to Streamlit Secrets:")
    st.markdown("""
    ### 🔑 Setting up Secrets on Streamlit Community Cloud:
    1. In your Streamlit Cloud dashboard, open your app settings (**⋮** menu &rarr; **Settings**).
    2. Go to the **Secrets** tab on the left.
    3. Paste your credentials:
    ```toml
    OPENAI_API_KEY = "sk-proj-your-key-here"
    PINECONE_API_KEY = "pcsk_your-key-here"
    PINECONE_INDEX_NAME = "agentic-ai-index"
    ```
    4. Click **Save**. Your app will automatically reboot and start answering questions!
    """)
    st.stop()
except Exception as e:
    st.error(f"⚠️ Initialization Error: {e}")
    st.info("Failed to connect to vector store or models. Please check your credentials and internet connection.")
    st.stop()


# ---------------------------------------------------------------------------
# Chat History
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_context" not in st.session_state:
    st.session_state.last_context = []

if "last_score" not in st.session_state:
    st.session_state.last_score = 0.0


# ---------------------------------------------------------------------------
# Sidebar: Retrieved Context
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📚 Retrieved Context")
    st.markdown("---")

    if st.session_state.last_context:
        # Confidence badge
        score = st.session_state.last_score
        if score >= 0.8:
            badge_class = "score-high"
        elif score >= 0.5:
            badge_class = "score-medium"
        else:
            badge_class = "score-low"

        st.markdown(
            f'<span class="score-badge {badge_class}">Confidence: {score:.0%}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        for i, chunk in enumerate(st.session_state.last_context, 1):
            st.markdown(f"**Chunk {i}**")
            st.markdown(
                f'<div class="context-chunk">{chunk[:500]}{"..." if len(chunk) > 500 else ""}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Ask a question to see retrieved context here.")

    st.markdown("---")
    st.caption("Built with LangGraph · Pinecone · OpenAI")


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">🤖 Agentic AI RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions about the Agentic AI eBook — answers are strictly grounded in the document.</div>',
    unsafe_allow_html=True,
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_query := st.chat_input("Ask a question about Agentic AI..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Run RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching the document and generating answer..."):
            initial_state = {
                "question": user_query,
                "context": [],
                "answer": "",
                "score": 0.0,
            }
            try:
                result = graph.invoke(initial_state)

                answer = result["answer"]
                context = result["context"]
                score = result["score"]

                # Update sidebar state
                st.session_state.last_context = context
                st.session_state.last_score = score

                # Display answer
                st.markdown(answer)

                # Show inline confidence
                st.caption(f"📊 Confidence: {score:.0%} · 📄 {len(context)} chunks retrieved")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as e:
                err_text = str(e)
                if "insufficient_quota" in err_text or "credit_balance_exhausted" in err_text:
                    st.error("⚠️ **OpenAI Quota Limit:** Your OpenAI account has exhausted its available credits. Please check your billing at [platform.openai.com](https://platform.openai.com/settings/organization/billing/).")
                elif "unauthorized" in err_text.lower() or "401" in err_text:
                    st.error("⚠️ **Authentication Error:** Invalid API Key provided for OpenAI or Pinecone. Please verify your keys in Streamlit Secrets.")
                else:
                    st.error(f"⚠️ **Error:** {err_text}")
