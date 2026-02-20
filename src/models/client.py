from qdrant_client import QdrantClient
from src.configurations.app_setting import (
    EMBEDDING_MODEL_NAME,
    QDRANT_URL,
    QDRANT_PORT,
    SPARSE_MODEL_NAME,
    GEMINI_KEY,
    GEMINI_MODEL_NAME,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_TOKENS,
    GEMINI_TIMEOUT,
    GEMINI_MAX_RETRIES,
    QDRANT_COLLECTION_NAME,
    RERANKER_MODEL_NAME,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
import logging
import yaml
from sentence_transformers import CrossEncoder


logging.basicConfig(level=logging.INFO)

with open("src/configurations/prompts.yaml", "r") as f:
    prompt_data = yaml.safe_load(f)

with open("src/configurations/main_config.yaml", "r") as f:
    main_config = yaml.safe_load(f)

mymemory = MemorySaver()
logging.info("mymemory")
agent_instances = {}

qdrant_client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
sparse_model = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)
reranking_model = CrossEncoder(RERANKER_MODEL_NAME, trust_remote_code=True)


def get_vectorstore():
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embedding_model,
        retrieval_mode=RetrievalMode.HYBRID,  # Dense + sparse
        vector_name="dense_vector",
        sparse_vector_name="sparse_vector",
        sparse_embedding=sparse_model,
    )


llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    temperature=GEMINI_TEMPERATURE,
    max_tokens=GEMINI_MAX_TOKENS,
    timeout=GEMINI_TIMEOUT,
    max_retries=GEMINI_MAX_RETRIES,
    google_api_key=GEMINI_KEY,
)


def get_qdrant_client() -> QdrantClient:
    return qdrant_client


def get_embedding_model() -> HuggingFaceEmbeddings:
    return embedding_model


def get_sparse_model() -> FastEmbedSparse:
    return sparse_model


def get_reranking_model() -> CrossEncoder:
    return reranking_model


def get_llm() -> ChatGoogleGenerativeAI:
    return llm


logging.info("Client setup completed successfully")
