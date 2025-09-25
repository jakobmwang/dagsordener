#!/usr/bin/env python3
"""
Backfill fetcher: crawl historical meetings for a city.

What it does:
- Loads the supplied frontpage URL and discovers available meeting years
- Visits each year (oldest first), scrolling until all meetings are listed
- Ingests meetings that are not yet present under <out_root>/<agenda|minutes>/<meeting-id>

Usage:
  python -m etl.extract.backfill \
    --url https://dagsordener.aarhus.dk/ \
    --out data/raw/meetings/aarhus
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from etl.extract.ingestion import DEFAULT_RPS, ingest_meeting, parse_meeting_id_from_url
from etl.extract.listings import ListingEntry, collect_year_entries, list_available_years


def ingest_links(
    entries: Iterable[ListingEntry],
    *,
    out_root: Path,
    with_audio: bool,
    headless: bool,
    rps: float,
) -> list[str]:
    """Ingest each meeting URL unless it already exists. Returns ingested IDs."""
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ingested: list[str] = []
    for entry in entries:
        url = entry.url
        mid = parse_meeting_id_from_url(url)
        if mid:
            existing_meta = None
            for kind_dir in ('agenda', 'minutes', 'other'):
                meta_path = out_root / kind_dir / mid / 'meta.json'
                if meta_path.exists():
                    existing_meta = meta_path
                    break
            if existing_meta:
                print(f'[skip] {mid} already ingested ({existing_meta.parent})')
                continue
        try:
            meta = ingest_meeting(
                meeting_url=url,
                out_root=str(out_root),
                with_audio=with_audio,
                headless=headless,
                rps=rps,
            )
            ingested.append(meta.meeting_id)
            print(
                f'[ok] {meta.meeting_id} -> {meta.items_count} items, '
                f'{meta.attachments_count_total} attachments'
            )
        except Exception as exc:  # keep going on errors
            print(f'[error] ingest failed for url={url}: {exc}')
    return ingested


def _main():
    ap = argparse.ArgumentParser(description='Backfill historical meetings for a city')
    ap.add_argument('--url', required=True, help='Frontpage URL (e.g. https://dagsordener.aarhus.dk/)')
    ap.add_argument('--out', required=True, help='Output city root (e.g. data/raw/meetings/aarhus)')
    ap.add_argument('--no-audio', action='store_true', help='Do not download audio MP3s')
    ap.add_argument('--headful', action='store_true', help='Show browser (not headless)')
    ap.add_argument('--rps', type=float, default=DEFAULT_RPS, help='Max requests per second for downloads')
    ap.add_argument(
        '--print-only',
        action='store_true',
        help='Only print discovered links; do not ingest',
    )
    args = ap.parse_args()

    years = list_available_years(args.url, headless=not args.headful)
    if not years:
        print('[warn] No years discovered; nothing to backfill')
        return

    # Backfill from oldest to newest to respect chronology.
    years_sorted = sorted(years)
    print(f'[info] Backfill years: {", ".join(map(str, years_sorted))}')

    for year in years_sorted:
        entries = collect_year_entries(args.url, year, headless=not args.headful)
        if not entries:
            print(f'[info] Year {year}: no entries discovered')
            continue
        if args.print_only:
            for entry in entries:
                print(entry.url)
            continue
        print(f'[info] Year {year}: processing {len(entries)} meetings')
        ingest_links(
            entries,
            out_root=Path(args.out),
            with_audio=not args.no_audio,
            headless=not args.headful,
            rps=args.rps,
        )


if __name__ == '__main__':
    _main()
