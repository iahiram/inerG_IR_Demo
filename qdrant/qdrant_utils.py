

import os
import uuid
import hashlib
import logging
import json
from qdrant_client.http.models import (
    PointStruct,
    SparseVector,
    VectorParams,
    Distance,
    SparseVectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from configurations.app_setting import (
    QDRANT_COLLECTION_NAME,
    
)


from client import qdrant_client, embedding_model,sparse_model,get_vectorstore

def upsert_circular_to_qdrant(json_data, collection_name: str = QDRANT_COLLECTION_NAME):
    """
    Upserts a single circular entry (question/answer) to Qdrant.
    """
    logging.info(f"Starting upsert for circular number: {json_data['metadata']['circular_number']} page: {json_data['metadata']['page']}")
    if not qdrant_client.collection_exists(collection_name):
        try:
            logging.info(f"Collection '{collection_name}' does not exist. Attempting to create it.")
            # Get embedding dimension
            test_embedding = embedding_model.embed_query("test")
            vector_size = len(test_embedding)
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense_vector": VectorParams(size=vector_size, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse_vector": SparseVectorParams(
                        index=None,
                        modifier=None,
                    )
                },
            )
            logging.info(f"Created collection '{collection_name}' with vector size {vector_size}")
        except Exception as e:
            logging.warning(f"Collection creation failed (might exist): {e}")

    
    try:
        logging.info(f"Checking for duplicates in collection '{collection_name}' for circular number: {json_data['metadata']['circular_number']} page: {json_data['metadata']['page']}")
        duplicate_check, _ = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="page_content",
                        match=MatchValue(value=json_data["page_content"]),
                    ),
                ]
            ),
            limit=1,
        )

        if duplicate_check:
            logging.info(
                f"Duplicate circular found for circular number: {json_data['metadata']['circular_number']} page: {json_data['metadata']['page']}... Overwriting existing entry."
            )
            

    except Exception as e:
        logging.error(f"Error checking for duplicates: {e}")
        pass


    try:
        # Create a deterministic ID based on content
        logging.info(f"Generating point ID for circular number: {json_data['metadata']['circular_number']} page: {json_data['metadata']['page']}")
        point_id = str(
            uuid.UUID(
                bytes=hashlib.md5(json_data["page_content"].encode("utf-8")).digest()
            )
        )
        dense_vector_values = embedding_model.embed_query(json_data["page_content"])


        # Sparse embedding
        sparse_result = sparse_model.embed_query(json_data["page_content"])
        sparse_vector_struct = {
            "indices": sparse_result.indices,
            "values": sparse_result.values,
        }
        logging.info(f"Generated point ID: {point_id} for circular number: {json_data['metadata']['circular_number']} page: {json_data['metadata']['page']}")
        point = PointStruct(
            id=point_id,
            vector={
                "dense_vector": dense_vector_values,
                "sparse_vector": sparse_vector_struct,
            },
            payload={
                "metadata": json_data["metadata"],
                "page_content": json_data["page_content"],
            },
        )

        # 5. Upsert
        logging.info(f"Upserting point ID: {point_id} to collection '{collection_name}' for circular number: {json_data['metadata']['circular_number']} page: {json_data['metadata']['page']}")
        qdrant_client.upsert(collection_name=collection_name, points=[point], wait=True)
        logging.info(f"Successfully upserted point {point_id} to Qdrant.")

    except Exception as e:
        logging.error(f"Error processing/upserting item to Qdrant: {e}")


def get_retrieved_data(query,content_type):
    """
    Retrieves relevant data from Qdrant based on the query and content type.
    """
    try:
        vectorstore = get_vectorstore()

        # Build filter based on content type if provided
        filters = []
        if content_type:
            filters.append(
                FieldCondition(
                    key="metadata.content_type",
                    match=MatchValue(value=content_type),
                )
            )

        q_filter = Filter(must=filters) if filters else None
        retrieved_points  = vectorstore.similarity_search_with_score(
            query=query,
            filter=q_filter,
            k=5
        )


        logging.info(f"Retrieved {len(retrieved_points)} points from Qdrant for query: '{query}' with content type: '{content_type}'")
        return retrieved_points

    except Exception as e:
        logging.error(f"Error during retrieval from Qdrant: {e}")
        return []


if __name__ == "__main__":
    json_data=json.load(open("/home/jarvis/inerg_demo/data/chunks.json"))
    upsert_circular_to_qdrant(json_data=json_data)