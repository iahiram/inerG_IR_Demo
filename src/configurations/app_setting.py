import os
from dotenv import load_dotenv
load_dotenv()


EMBEDDING_MODEL_NAME=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
QDRANT_COLLECTION_NAME=os.getenv("QDRANT_COLLECTION_NAME", "customer_service_knowledge")

SPARSE_MODEL_NAME=os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm25")


QDRANT_URL=os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT=int(os.getenv("QDRANT_PORT", 6333))

GEMINI_KEY = os.getenv("GEMINI_KEY")

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", 0))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", 2048))
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", 10))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", 2))

TOP_K = int(os.getenv("TOP_K", 5))
USE_RERANKER=bool(os.getenv("USE_RERANKER", True))