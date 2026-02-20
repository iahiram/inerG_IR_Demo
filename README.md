# Customer Service Knowledge Assistant (inerG IR Demo)

A state-of-the-art information retrieval and agentic search system designed for customer service knowledge bases. This project leverages **LangGraph**, **Qdrant**, and **Google Gemini** to provide accurate, context-aware responses to customer queries using a Retrieval-Augmented Generation (RAG) architecture.

## 🚀 Key Features

-   **Hybrid Search**: Combines dense (HuggingFace) and sparse (BM25 via FastEmbed) embeddings for superior retrieval accuracy in Qdrant.
-   **Agentic Workflow**: Managed by LangGraph for multi-node reasoning and message history handling.
-   **Interactive UI**: A premium Streamlit-based dashboard for performing queries and visualizing retrieved documents with their metadata and scores.
-   **Automated Ingestion**: streamlined script to process and upsert document chunks into the vector database.
-   **Reranking**: Integrated support for cross-encoder reranking to refine search results (optional).

## 🛠️ Technology Stack

-   **Backend**: Python 3.12, LangChain, LangGraph
-   **Vector Database**: Qdrant (Hybrid Search)
-   **LLM**: Google Gemini (via `langchain-google-genai`)
-   **Embeddings**: 
    -   Dense: `all-MiniLM-L6-v2` (default)
    -   Sparse: `Qdrant/bm25`
-   **Frontend**: Streamlit
-   **Containerization**: Docker & Docker Compose

## 📋 Prerequisites

-   Python 3.12+
-   Docker and Docker Compose
-   Google Gemini API Key

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd inerG_IR_Demo
```

### 2. Environment Variables
Create a `.env` file in the root directory and populate it with the following:
```env
# Models
GEMINI_KEY=your_google_gemini_api_key
GEMINI_MODEL_NAME=gemini-2.5-flash
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
SPARSE_MODEL_NAME=Qdrant/bm25

# Qdrant
QDRANT_URL=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=customer_service_knowledge

# Settings
USE_RERANKER=false
TOP_K=5
```

### 3. Running with Docker (Recommended)
The project is containerized for easy deployment.
```bash
docker-compose up --build
```
This will start:
-   **Qdrant**: Available at `http://localhost:6333`
-   **App**: Streamlit interface at `http://localhost:8501`

### 4. Local Development
If you prefer to run it outside Docker:
1. Install dependencies (using [uv]):
   ```bash
   uv sync
   ```
   
2. Start Qdrant (using Docker):
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```
3. Ingest data:
   ```bash
   # Using uv
   uv run ingest_data.py
   # Or using python directly
   python ingest_data.py
   ```
4. Run the app:
   ```bash
   # Using uv
   uv run streamlit run streamlit_app.py
   # Or using streamlit directly
   streamlit run streamlit_app.py
   ```

## 📂 Project Structure

```text
├── src/
│   ├── architectures/     # LangGraph nodes and graph builders
│   ├── configurations/    # Prompts, model settings, and configs
│   ├── data/              # Raw data chunks for ingestion
│   ├── models/            # LLM and VectorStore client initializations
│   └── qdrant/            # Qdrant utility functions
├── ingest_data.py         # Data ingestion pipeline
├── main.py                # Core logic and graph execution
├── streamlit_app.py       # Streamlit UI
├── docker-compose.yml     # Infrastructure orchestration
└── pyproject.toml         # Python project configuration
```

## 🔍 Usage

### Data Ingestion
The `ingest_data.py` script reads from `src/data/chunks.json` and upserts them into Qdrant. It automatically creates the collection with the correct dimensions and hybrid search configuration if it doesn't exist.

### Querying
1. Open the Streamlit app (`http://localhost:8501`).
2. Enter a customer query (e.g., "Why is my tracking not updating?").
3. View the generated answer and the source documents used for retrieval.

---
