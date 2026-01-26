"""Search and retrieve agenda items from Qdrant."""

from datetime import date, datetime, time
from typing import Iterable

from qdrant_client import QdrantClient, models

from src.embedder import Embedder
from src.qdrant import BM25_CONFIG, QDRANT_COLLECTION, da_expand, get_client

DEFAULT_LIMIT = 8
PREFETCH_LIMIT = 128


# =============================================================================
# Main functions - used by the agent and search engine
# =============================================================================


def search(
    query: str,
    *,
    udvalg: str | list[str] | None = None,
    sagsnummer: str | list[str] | None = None,
    date_after: str | None = None,
    date_before: str | None = None,
    limit: int = DEFAULT_LIMIT,
    prefetch_limit: int = PREFETCH_LIMIT,
    offset: int = 0,
) -> dict:
    """
    Search agenda items with semantic search and optional filters.

    Uses hybrid search (dense + sparse + BM25 vectors) with RRF fusion.

    Args:
        query: Search text, e.g. "cykelstier i Aarhus" or "budget 2024"
        udvalg: Filter by committee, e.g. "Byrådet" or ["Byrådet", "Teknisk Udvalg"]
        sagsnummer: Filter by case number, e.g. "SAG-2024-12345"
        date_after: Only items on or after this date (inclusive), format: "YYYY-MM-DD", e.g. "2024-01-01"
        date_before: Only items on or before this date (inclusive), format: "YYYY-MM-DD", e.g. "2024-12-31"
        limit: Number of results (default 8, use higher for more context)
        offset: Start from this offset (for pagination, next page: offset + limit)

    Returns:
        Dict with items (list of points), total (match count), has_more (bool).
        Each item has: punkt_id, score, payload (title, content_md, udvalg, sagsnummer, datetime, ...)

    Example:
        >>> search("cykelstier", udvalg="Teknisk Udvalg", limit=5)
        {"items": [...], "total": 42, "has_more": True}
        >>> search("budget", date_after="2024-01-01", offset=10)  # page 2
    """
    query_text = query.strip()
    if not query_text:
        raise ValueError("Query cannot be empty - use get_case() or get_meeting() to filter without search text")

    client = get_client()
    query_filter = build_filter(
        udvalg=udvalg,
        sagsnummer=sagsnummer,
        date_after=date_after,
        date_before=date_before,
    )

    total = client.count(
        collection_name=QDRANT_COLLECTION,
        count_filter=query_filter,
        exact=True,
    ).count

    if total == 0:
        return {"items": [], "total": 0, "has_more": False}

    embedder = Embedder()
    emb = embedder.embed_single(query_text)
    dense = emb["dense"]
    sparse = models.SparseVector(indices=emb["sparse"]["indices"], values=emb["sparse"]["values"])
    expanded_query = da_expand(query_text)
    bm25_text = f"{query_text} {expanded_query}" if expanded_query != query_text else query_text
    bm25_query = models.Document(
        text=bm25_text,
        model="Qdrant/Bm25",
        options=BM25_CONFIG,
    )

    prefetch_limit = min(prefetch_limit, total)
    fetch_limit = limit + offset

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        prefetch=[
            models.Prefetch(query=dense, using="dense", limit=prefetch_limit, filter=query_filter),
            models.Prefetch(query=sparse, using="sparse", limit=prefetch_limit, filter=query_filter),
            models.Prefetch(query=bm25_query, using="bm25", limit=prefetch_limit, filter=query_filter),
        ],
        query_filter=query_filter,
        limit=fetch_limit,
        with_payload=True,
    )

    points = results.points
    if not points:
        return {"items": [], "total": total, "has_more": False}

    ordered = [record_to_dict(p) for p in points]

    items = ordered[offset : offset + limit]
    has_more = offset + limit < len(ordered) or len(ordered) < total

    return {"items": items, "total": total, "has_more": has_more}


def get_case(sagsnummer: str) -> list[dict]:
    """
    Get all agenda items for a case, sorted by date (newest first).

    Use this function to get an overview of an entire case across meetings.

    Args:
        sagsnummer: Case number, e.g. "SAG-2024-12345"

    Returns:
        List of items sorted by datetime (newest first).
        Each item has: punkt_id, payload (title, content_md, udvalg, datetime, ...)

    Example:
        >>> get_case("SAG-2024-12345")
        [{"punkt_id": "...", "payload": {"title": "...", "datetime": "2024-06-15T10:00:00", ...}}, ...]
    """
    if not sagsnummer or not sagsnummer.strip():
        raise ValueError("sagsnummer is required")

    client = get_client()
    query_filter = build_filter(sagsnummer=sagsnummer.strip())

    records = scroll_all(client, query_filter)
    records.sort(key=lambda r: (r["payload"].get("datetime", ""), r["payload"].get("index", 0)), reverse=True)

    return records


def get_meeting(meeting_id: str) -> list[dict]:
    """
    Get all agenda items from a meeting, sorted by item number.

    Use this function to see the full agenda for a specific meeting.

    Args:
        meeting_id: Meeting ID from the system

    Returns:
        List of items sorted by index (item number on the agenda).
        Each item has: punkt_id, payload (title, content_md, index, udvalg, datetime, ...)

    Example:
        >>> get_meeting("meeting-123")
        [{"punkt_id": "...", "payload": {"title": "1. Godkendelse af dagsorden", "index": 0, ...}}, ...]
    """
    if not meeting_id or not meeting_id.strip():
        raise ValueError("meeting_id is required")

    client = get_client()
    query_filter = build_filter(meeting_id=meeting_id.strip())

    records = scroll_all(client, query_filter)
    records.sort(key=lambda r: r["payload"].get("index", 0))

    return records


def get_point(punkt_id: str) -> dict | None:
    """
    Get a specific agenda item.

    Use this function when you have a punkt_id from search results and want to read the full content.

    Args:
        punkt_id: Item ID (from search results)

    Returns:
        Dict with punkt_id and payload, or None if not found.
        Payload contains: title, content_md, udvalg, sagsnummer, datetime, url, links, ...

    Example:
        >>> get_point("abc-123")
        {"punkt_id": "abc-123", "payload": {"title": "...", "content_md": "...", ...}}
    """
    if not punkt_id or not punkt_id.strip():
        return None

    client = get_client()
    records = client.retrieve(collection_name=QDRANT_COLLECTION, ids=[punkt_id.strip()], with_payload=True)

    if not records:
        return None

    return record_to_dict(records[0])


# =============================================================================
# Helper functions
# =============================================================================


def build_filter(
    *,
    udvalg: str | Iterable[str] | None = None,
    sagsnummer: str | Iterable[str] | None = None,
    meeting_id: str | Iterable[str] | None = None,
    date_after: str | None = None,
    date_before: str | None = None,
) -> models.Filter | None:
    """Build Qdrant filter from parameters."""
    conditions: list[models.Condition] = []

    for key, value in [("udvalg", udvalg), ("sagsnummer", sagsnummer), ("meeting_id", meeting_id)]:
        condition = match_condition(key, value)
        if condition:
            conditions.append(condition)

    if date_after or date_before:
        dt_range = models.DatetimeRange(
            gte=parse_datetime(date_after) if date_after else None,
            lte=parse_datetime(date_before, end_of_day=True) if date_before else None,
        )
        conditions.append(models.FieldCondition(key="datetime", range=dt_range))

    return models.Filter(must=conditions) if conditions else None


def match_condition(key: str, value: str | Iterable[str] | None) -> models.FieldCondition | None:
    """Build match condition for a field."""
    if value is None:
        return None
    if isinstance(value, str):
        return models.FieldCondition(key=key, match=models.MatchValue(value=value)) if value else None
    values = [v for v in value if v]
    if not values:
        return None
    return models.FieldCondition(key=key, match=models.MatchAny(any=values))


def parse_datetime(value: str, *, end_of_day: bool = False) -> datetime:
    """Parse ISO 8601 date/datetime string."""
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = f"{cleaned[:-1]}+00:00"
    if "T" not in cleaned and " " not in cleaned:
        parsed_date = date.fromisoformat(cleaned)
        return datetime.combine(parsed_date, time.max if end_of_day else time.min)
    return datetime.fromisoformat(cleaned)


def record_to_dict(record, score: float | None = None) -> dict:
    """Convert Qdrant record to dict."""
    return {
        "punkt_id": str(record.id),
        "score": score if score is not None else getattr(record, "score", None),
        "payload": record.payload or {},
    }


def scroll_all(client: QdrantClient, query_filter: models.Filter | None) -> list[dict]:
    """Fetch all items matching a filter via scroll."""
    records = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=query_filter,
            limit=256,
            offset=offset,
            with_payload=True,
        )
        records.extend(record_to_dict(r) for r in batch)
        if not batch or next_offset is None:
            break
        offset = next_offset
    return records
