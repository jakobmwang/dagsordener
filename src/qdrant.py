"""Qdrant operations for storing and retrieving chunks."""

import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Datatype,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "dagsordener")
DENSE_DIM = 1024  # BGE-M3 dimension


def get_client(url: str | None = None) -> QdrantClient:
    """Get Qdrant client."""
    return QdrantClient(url=url or URL)


def _ensure_payload_indexes(client: QdrantClient):
    desired = {
        "meeting_id": PayloadSchemaType.KEYWORD,
        "udvalg": PayloadSchemaType.KEYWORD,
        "sagsnummer": PayloadSchemaType.KEYWORD,
        "datetime": PayloadSchemaType.DATETIME,
    }
    collection_info = client.get_collection(COLLECTION)
    payload_schema = collection_info.payload_schema or {}
    for field, schema_type in desired.items():
        existing = payload_schema.get(field)
        if existing is None:
            client.create_payload_index(COLLECTION, field, field_schema=schema_type)
        elif existing.data_type != schema_type:
            print(
                f"  Payload index '{field}' is {existing.data_type}, "
                f"expected {schema_type}. Consider rebuilding the index."
            )


def ensure_collection(client: QdrantClient):
    """Create collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION not in collections:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                    datatype=Datatype.FLOAT16,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            },
        )
        _ensure_payload_indexes(client)
    else:
        _ensure_payload_indexes(client)


def meeting_exists(client: QdrantClient, meeting_id: str) -> bool:
    """Check if any punkt from this meeting already exists."""
    result = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="meeting_id", match=MatchValue(value=meeting_id))]
        ),
        limit=1,
    )
    return len(result[0]) > 0


def upsert_punkter(client: QdrantClient, punkter: list[dict], embeddings: dict):
    """Upsert punkter with their embeddings."""
    points = []

    for i, punkt in enumerate(punkter):
        dense = embeddings["dense"][i]
        sparse = embeddings["sparse"][i]

        point = PointStruct(
            id=punkt["punkt_id"],
            vector={
                "dense": dense,
                "sparse": SparseVector(
                    indices=sparse["indices"],
                    values=sparse["values"],
                ),
            },
            payload={
                "url": punkt["url"],
                "index": punkt["index"],
                "title": punkt["title"],
                "sagsnummer": punkt["sagsnummer"],
                "content_md": punkt["content_md"],
                "links": punkt["links"],  # indexed URLs from content
                "meeting_id": punkt["meeting_id"],
                "udvalg": punkt["udvalg"],
                "datetime": punkt["datetime"],
                "sted": punkt["sted"],
                "type": punkt["type"],
            },
        )
        points.append(point)

    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    return len(points)
