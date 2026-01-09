"""Qdrant operations for storing and retrieving chunks."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

COLLECTION = "dagsordener"
DENSE_DIM = 1024  # BGE-M3 dimension


def get_client(url: str = "http://localhost:6333") -> QdrantClient:
    """Get Qdrant client."""
    return QdrantClient(url=url)


def ensure_collection(client: QdrantClient):
    """Create collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION not in collections:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            },
        )
        # Create payload indexes for filtering
        client.create_payload_index(COLLECTION, "meeting_id", "keyword")
        client.create_payload_index(COLLECTION, "udvalg", "keyword")
        client.create_payload_index(COLLECTION, "sagsnummer", "keyword")
        client.create_payload_index(COLLECTION, "datetime", "keyword")


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
