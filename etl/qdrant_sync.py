#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from typing import Iterable
from urllib.parse import urljoin

from FlagEmbedding import BGEM3FlagModel
from playwright.sync_api import TimeoutError as PWTimeout
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from etl.extract.html_meeting import MeetingData, browser_context, load_meeting
from etl.extract.listings import collect_year_entries

DEFAULT_COLLECTION = 'dagsordener'
DEFAULT_MODEL = 'BAAI/bge-m3'


class BgeM3Embedder:
    def __init__(self, model_name: str, *, use_fp16: bool, device: str | None = None):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)

    def embed(self, texts: list[str]) -> tuple[list[list[float]], list[dict[str, list[float]]]]:
        result = self.model.encode(texts, batch_size=8, max_length=8192)
        dense = result['dense_vecs']
        sparse = result['sparse_vecs']
        return dense, sparse


def _hash_fallback(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:32]


def _build_chunk_text(meeting: MeetingData, item) -> str:
    lines = []
    if item.title:
        lines.append(f'# {item.title}')
    if meeting.committee:
        lines.append(f'Udvalg: {meeting.committee}')
    if meeting.datetime_text:
        lines.append(f'Dato: {meeting.datetime_text}')
    if meeting.place:
        lines.append(f'Sted: {meeting.place}')
    if item.case_number:
        lines.append(f'Sagsnummer: {item.case_number}')
    lines.append(f'Møde URL: {meeting.meeting_url}')
    if item.index is not None:
        lines.append(f'Punkt: {item.index}')
    lines.append('')
    lines.append(item.markdown)
    return '\n'.join(lines).strip()


def _sparse_to_qdrant(sparse: dict[str, list[float]]) -> qmodels.SparseVector:
    indices = sparse.get('indices', [])
    values = sparse.get('values', [])
    return qmodels.SparseVector(indices=list(indices), values=list(values))


def ensure_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    collections = client.get_collections().collections
    if any(col.name == name for col in collections):
        return

    client.create_collection(
        collection_name=name,
        vectors_config={
            'dense': qmodels.VectorParams(size=dense_dim, distance=qmodels.Distance.COSINE)
        },
        sparse_vectors_config={
            'sparse': qmodels.SparseVectorParams(index=qmodels.SparseIndexParams(on_disk=False))
        },
    )


def _meeting_points(meeting: MeetingData) -> tuple[list[str], list[qmodels.PointStruct]]:
    texts: list[str] = []
    points: list[qmodels.PointStruct] = []
    for item in meeting.items:
        chunk_text = _build_chunk_text(meeting, item)
        if not chunk_text:
            continue
        item_id = item.item_id or _hash_fallback(chunk_text)
        payload = {
            'meeting_id': meeting.meeting_id,
            'meeting_url': meeting.meeting_url,
            'committee': meeting.committee,
            'datetime': meeting.datetime_text,
            'place': meeting.place,
            'kind': meeting.kind,
            'item_id': item.item_id,
            'item_index': item.index,
            'item_title': item.title,
            'case_number': item.case_number,
            'content': item.markdown,
        }
        texts.append(chunk_text)
        points.append(
            qmodels.PointStruct(
                id=item_id,
                vector={},
                payload=payload,
            )
        )
    return texts, points


def upsert_meeting(
    client: QdrantClient,
    embedder: BgeM3Embedder,
    meeting: MeetingData,
    collection: str,
) -> int:
    texts, points = _meeting_points(meeting)
    if not texts:
        return 0

    dense_vecs, sparse_vecs = embedder.embed(texts)
    if not dense_vecs:
        return 0

    ensure_collection(client, collection, len(dense_vecs[0]))

    for point, dense_vec, sparse_vec in zip(points, dense_vecs, sparse_vecs):
        point.vector = {
            'dense': dense_vec,
            'sparse': _sparse_to_qdrant(sparse_vec),
        }

    client.upsert(collection_name=collection, points=points)
    return len(points)


def _extract_frontpage_links(
    frontpage_url: str,
    *,
    limit: int,
    headless: bool,
) -> list[str]:
    with browser_context(headless=headless) as ctx:
        page = ctx.new_page()
        page.set_default_timeout(15000)
        page.goto(frontpage_url)
        try:
            page.wait_for_selector('#resultater a.searchresult', state='visible', timeout=20000)
        except PWTimeout:
            page.wait_for_selector('a.list-group-item.searchresult', timeout=20000)

        anchors = page.locator('#resultater a.searchresult')
        if anchors.count() == 0:
            anchors = page.locator('a.list-group-item.searchresult')

        hrefs: list[str] = []
        count = anchors.count()
        for i in range(min(limit, count)):
            href = (anchors.nth(i).get_attribute('href') or '').strip()
            if href:
                hrefs.append(href)

    return [urljoin(frontpage_url, href) for href in hrefs]


def _extract_links_for_year(frontpage_url: str, year: int, *, headless: bool) -> list[str]:
    entries = collect_year_entries(frontpage_url, year, headless=headless)
    return [entry.url for entry in entries]


def _iter_meeting_urls(
    frontpage_url: str,
    *,
    limit: int | None,
    year: int | None,
    headless: bool,
) -> Iterable[str]:
    if year is not None:
        return _extract_links_for_year(frontpage_url, year, headless=headless)
    if limit is None:
        limit = 20
    return _extract_frontpage_links(frontpage_url, limit=limit, headless=headless)


def sync_meetings(
    frontpage_url: str,
    *,
    collection: str,
    qdrant_url: str,
    qdrant_api_key: str | None,
    model_name: str,
    limit: int | None,
    year: int | None,
    include_agendas: bool,
    headless: bool,
    use_fp16: bool,
    device: str | None,
) -> None:
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embedder = BgeM3Embedder(model_name, use_fp16=use_fp16, device=device)

    for url in _iter_meeting_urls(
        frontpage_url,
        limit=limit,
        year=year,
        headless=headless,
    ):
        meeting = load_meeting(url, headless=headless)
        if meeting.kind != 'referat' and not include_agendas:
            print(f'[skip] {meeting.meeting_id} kind={meeting.kind}')
            continue
        count = upsert_meeting(client, embedder, meeting, collection)
        print(f'[upsert] {meeting.meeting_id} -> {count} points')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Sync dagsordener to Qdrant')
    parser.add_argument('--url', required=True, help='Frontpage URL')
    parser.add_argument('--collection', default=DEFAULT_COLLECTION, help='Qdrant collection name')
    parser.add_argument('--qdrant-url', default='http://localhost:6333', help='Qdrant URL')
    parser.add_argument('--qdrant-api-key', default=None, help='Qdrant API key')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Embedding model')
    parser.add_argument('--limit', type=int, default=20, help='How many frontpage items to sync')
    parser.add_argument('--year', type=int, help='Sync all meetings for a given year')
    parser.add_argument('--include-agendas', action='store_true', help='Include dagsorden entries')
    parser.add_argument('--headful', action='store_true', help='Show browser')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16 embeddings')
    parser.add_argument('--device', default=None, help='Override device (e.g. cpu/cuda)')
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sync_meetings(
        args.url,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        model_name=args.model,
        limit=None if args.year else args.limit,
        year=args.year,
        include_agendas=args.include_agendas,
        headless=not args.headful,
        use_fp16=not args.no_fp16,
        device=args.device,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
