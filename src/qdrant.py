"""Qdrant operations for storing and retrieving chunks."""

import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Bm25Config,
    Distance,
    Datatype,
    Document,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dagsordener")
DENSE_DIM = 1024  # BGE-M3 dimension
BM25_CONFIG = Bm25Config(language="danish", ascii_folding=True)


def get_client(url: str = QDRANT_URL) -> QdrantClient:
    """Get Qdrant client."""
    return QdrantClient(url=url)


def da_expand(text: str) -> str:
    return (
        text.replace("\u00c5", "Aa")
        .replace("\u00e5", "aa")
        .replace("\u00c6", "Ae")
        .replace("\u00e6", "ae")
        .replace("\u00d8", "Oe")
        .replace("\u00f8", "oe")
    )


def ensure_collection(client: QdrantClient, collection: str):
    """Create collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]

    if collection not in collections:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                    datatype=Datatype.FLOAT16,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
                "bm25": SparseVectorParams(),
            },
        )
    else:
        collection_info = client.get_collection(collection)
        sparse_vectors = collection_info.config.params.sparse_vectors or {}
        if "bm25" not in sparse_vectors:
            client.update_collection(
                collection_name=collection,
                sparse_vectors_config={"bm25": SparseVectorParams()},
            )

    desired = {
        "meeting_id": PayloadSchemaType.KEYWORD,
        "udvalg": PayloadSchemaType.KEYWORD,
        "sagsnummer": PayloadSchemaType.KEYWORD,
        "datetime": PayloadSchemaType.DATETIME,
    }

    collection_info = client.get_collection(collection)
    payload_schema = collection_info.payload_schema or {}
    for field, schema_type in desired.items():
        existing = payload_schema.get(field)
        if existing is None:
            client.create_payload_index(collection, field, field_schema=schema_type)
        elif existing.data_type != schema_type:
            print(
                f"  Payload index '{field}' is {existing.data_type}, "
                f"expected {schema_type}. Consider rebuilding the index."
            )


def meeting_exists(client: QdrantClient, collection: str, meeting_id: str) -> bool:
    """Check if any punkt from this meeting already exists."""
    result = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="meeting_id", match=MatchValue(value=meeting_id))]
        ),
        limit=1,
    )
    return len(result[0]) > 0


def upsert_punkter(client: QdrantClient, collection: str, punkter: list[dict], embeddings: dict):
    """Upsert punkter with their embeddings."""
    points = []

    for i, punkt in enumerate(punkter):
        dense = embeddings["dense"][i]
        sparse = embeddings["sparse"][i]
        chunk_text = punkt["chunk_text"]
        expanded_text = da_expand(chunk_text)
        bm25_text = f"{chunk_text}\n{expanded_text}" if expanded_text != chunk_text else chunk_text
        bm25_doc = Document(
            text=bm25_text,
            model="Qdrant/Bm25",
            options=BM25_CONFIG,
        )

        point = PointStruct(
            id=punkt["punkt_id"],
            vector={
                "dense": dense,
                "sparse": SparseVector(
                    indices=sparse["indices"],
                    values=sparse["values"],
                ),
                "bm25": bm25_doc,
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
        client.upsert(collection_name=collection, points=points)

    return len(points)
