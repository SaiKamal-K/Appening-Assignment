# 🤖 RAG Agentic AI Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions strictly grounded in the [Agentic AI eBook](https://drive.google.com/file/d/15VLphKcY23_fpYxN62UEQRri_psRVfP9/view?usp=sharing). Built with **LangGraph**, **Pinecone**, **OpenAI**, and **FastAPI** / **Streamlit**.

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Document Ingestion](#document-ingestion)
- [Running the Application](#running-the-application)
- [API Usage](#api-usage)
- [Sample Test Queries](#sample-test-queries)
- [Technology Stack](#technology-stack)

---

## ✨ Features

- **Strict Document Grounding** — Answers only from the ingested eBook; refuses out-of-scope questions.
- **Stateful RAG Pipeline** — LangGraph orchestrates retrieval → generation in a compiled execution graph.
- **Vector Search** — Pinecone stores and retrieves semantically relevant document chunks.
- **Confidence Scoring** — Every response includes a relevance/confidence score (0.0–1.0).
- **Dual Interface** — FastAPI REST endpoint **and** Streamlit chat UI.
- **Context Transparency** — Retrieved chunks are returned alongside the answer for full traceability.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────────┐
│  User Query  │────▶│             LangGraph StateGraph                │
└─────────────┘     │                                                  │
                    │  ┌───────────┐     ┌────────────────┐           │
                    │  │ RETRIEVE  │────▶│   GENERATE     │           │
                    │  │           │     │                │           │
                    │  │ Pinecone  │     │ OpenAI LLM     │           │
                    │  │ Top-K     │     │ Strict Prompt  │           │
                    │  │ Search    │     │ + Confidence   │           │
                    │  └───────────┘     └────────────────┘           │
                    └──────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
              ┌──────────┐      ┌──────────────┐      ┌────────────┐
              │  Answer   │      │  Context      │      │  Score     │
              │  (text)   │      │  (chunks)     │      │  (float)   │
              └──────────┘      └──────────────┘      └────────────┘
```

### RAG Pipeline Flow

1. **User submits a query** via FastAPI `/chat` endpoint or Streamlit UI.
2. **Retrieve Node** — Queries Pinecone vector store for the top-*k* most relevant document chunks using cosine similarity over OpenAI embeddings.
3. **Generate Node** — Feeds the retrieved context + user question into OpenAI's LLM with a strict grounding system prompt. Computes a confidence heuristic.
4. **Response** — Returns the generated answer, retrieved context chunks, and confidence score.

---

## 📁 Project Structure

```
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf       # Downloaded source document
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Environment setup & constants
│   ├── ingestion.py                # PDF loading, splitting & Pinecone indexing
│   └── graph.py                    # LangGraph workflow definition & state logic
│
├── app.py                          # FastAPI application
├── streamlit_app.py                # Streamlit chat UI (alternative)
├── tests_sample_queries.py         # Script with 6 benchmark test queries
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Production Docker container definition
├── .dockerignore                   # Docker build exclusions
├── Procfile                        # PaaS process file (Render, Railway, Heroku)
├── .env.example                    # Template for environment variables
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 📦 Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python**  | 3.10 or higher |
| **RAM**     | Minimum 8 GB |
| **OS**      | macOS, Linux, or Windows (WSL2 recommended) |
| **OpenAI API Key** | For embeddings (`text-embedding-3-small`) and LLM (`gpt-4o-mini`) |
| **Pinecone API Key** | Free tier account for vector index hosting |

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/rag-agentic-ai.git
cd rag-agentic-ai
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX_NAME=agentic-ai-index
```

### 5. Download the Source PDF

Download the [Agentic AI eBook PDF](https://drive.google.com/file/d/15VLphKcY23_fpYxN62UEQRri_psRVfP9/view?usp=sharing) and place it at:

```
data/Ebook-Agentic-AI.pdf
```

---

## 📥 Document Ingestion

Run the ingestion pipeline to parse the PDF, chunk the text, generate embeddings, and upsert them into Pinecone:

```bash
python -m src.ingestion
```

**What this does:**
1. Loads the PDF using `PyPDFLoader`
2. Splits it into ~1000-character chunks with 200-character overlap using `RecursiveCharacterTextSplitter`
3. Creates the Pinecone index (dimension=1536, cosine metric) if it doesn't exist
4. Generates embeddings via OpenAI `text-embedding-3-small` and upserts all chunks into Pinecone

> ⚠️ **Run this only once** (or when you want to re-index the document). The ingestion process may take a few minutes depending on PDF size.

---

## ▶️ Running the Application

### Option A: FastAPI Server

```bash
python app.py
```

Or with uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: **http://localhost:8000**

Interactive API docs: **http://localhost:8000/docs**

### Option B: Streamlit UI

```bash
streamlit run streamlit_app.py
```

Opens a browser with the chat interface at: **http://localhost:8501**

---

## 🚢 Deployment Guide

### Option 1: Docker (Recommended for Containers)

Build and run the Docker image locally or deploy to AWS ECS, GCP Cloud Run, or Azure:

```bash
# 1. Build the Docker container
docker build -t rag-agentic-ai .

# 2. Run the container with your environment variables
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your-openai-api-key" \
  -e PINECONE_API_KEY="your-pinecone-api-key" \
  -e PINECONE_INDEX_NAME="agentic-ai-index" \
  rag-agentic-ai
```

To run the Streamlit interface instead in Docker:
```bash
docker run -p 8501:8501 \
  -e OPENAI_API_KEY="your-openai-api-key" \
  -e PINECONE_API_KEY="your-pinecone-api-key" \
  rag-agentic-ai streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

### Option 2: Render / Railway / Fly.io

1. Connect your GitHub repository.
2. The included `Procfile` and `Dockerfile` are automatically detected:
   ```
   web: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
3. Set the Environment Variables in the platform dashboard:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME`
4. Deploy! The health check endpoint at `/` ensures zero-downtime health verification.

### Option 3: Streamlit Community Cloud

1. Push your repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and create a new app pointing to `streamlit_app.py`.
3. In Advanced Settings, add your secrets under **Secrets**:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key"
   PINECONE_API_KEY = "your-pinecone-api-key"
   PINECONE_INDEX_NAME = "agentic-ai-index"
   ```
4. Click **Deploy**.

---

## 🔌 API Usage

### Health Check

```bash
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Agentic AI RAG Chatbot API",
  "version": "1.0.0",
  "graph_initialized": true
}
```

### Chat Endpoint

```bash
POST /chat
Content-Type: application/json

{
  "query": "What is Agentic AI according to the eBook?"
}
```

**Response:**
```json
{
  "final_answer": "According to the eBook, Agentic AI refers to...",
  "retrieved_context": [
    "Chunk 1 text...",
    "Chunk 2 text...",
    "Chunk 3 text..."
  ],
  "confidence_score": 0.85
}
```

### Example with cURL

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the core components of an Agentic Architecture?"}'
```

---

## 🧪 Sample Test Queries

Run the benchmark test suite:

```bash
python tests_sample_queries.py
```

This executes 6 queries and prints formatted results:

| # | Query | Expected Behavior |
|---|-------|-------------------|
| 1 | What is Agentic AI according to the eBook? | Definition from eBook |
| 2 | How do AI agents differ from traditional automation systems? | Contrast based on eBook |
| 3 | What are the core components of an Agentic Architecture? | List components from eBook |
| 4 | What role does memory play in Agentic AI workflows? | Explain memory's role |
| 5 | What are the key challenges in deploying Agentic AI systems? | Discuss challenges |
| 6 | Who won the 2022 FIFA World Cup? | **Refuse / state insufficient context** |

Query #6 is a **validation test** — the chatbot should refuse to answer since this information is not in the eBook.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | Stateful RAG workflow graph |
| **Vector Store** | Pinecone | Embedding storage & similarity search |
| **Embeddings** | OpenAI `text-embedding-3-small` | Document chunk vectorization |
| **LLM** | OpenAI `gpt-4o-mini` | Grounded response generation |
| **PDF Parsing** | PyPDF / LangChain `PyPDFLoader` | Text extraction from PDF |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` | Document chunking |
| **API Framework** | FastAPI + Uvicorn | REST API backend |
| **UI Framework** | Streamlit | Chat interface |
| **Config** | python-dotenv | Environment variable management |

---

## 📄 License

This project is created as part of the Appening AI assignment.
