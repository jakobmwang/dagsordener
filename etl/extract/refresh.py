#!/usr/bin/env python3
"""
Refresh fetcher: re-ingest recent meetings for a city.

What it does:
- Loads the provided frontpage URL and discovers available meeting years
- Walks years from newest to oldest until meetings fall outside the refresh window
- Re-ingests every meeting discovered within the window, forcing fresh downloads

Usage:
  python -m etl.extract.refresh \
    --url https://dagsordener.aarhus.dk/ \
    --out data/raw/meetings/aarhus \
    --days 30
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Iterable

from etl.extract.ingestion import DEFAULT_RPS, ingest_meeting
from etl.extract.listings import ListingEntry, collect_year_entries, list_available_years


def parse_date(text: str | None) -> dt.date | None:
    if not text:
        return None

    text = text.strip()
    patterns = [
        '%Y-%m-%d',
        '%d-%m-%Y',
        '%d.%m.%Y',
        '%d/%m/%Y',
    ]
    for pattern in patterns:
        try:
            return dt.datetime.strptime(text, pattern).date()
        except ValueError:
            continue

    match = re.search(r'\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|\d{2}\.\d{2}\.\d{4}|\d{2}/\d{2}/\d{4}', text)
    if match:
        token = match.group(0)
        for pattern in patterns:
            try:
                return dt.datetime.strptime(token, pattern).date()
            except ValueError:
                continue

    # Try to extract a date token and parse
    tokens = ['-', '.', '/']
    for sep in tokens:
        parts = text.split(sep)
        if len(parts) == 3 and all(part.strip().isdigit() for part in parts):
            guess = sep.join(part.strip() for part in parts)
            for pattern in patterns:
                try:
                    return dt.datetime.strptime(guess, pattern).date()
                except ValueError:
                    continue
    return None


def refresh_entries(
    entries: Iterable[ListingEntry],
    *,
    out_root: Path,
    with_audio: bool,
    headless: bool,
    rps: float,
    threshold_date: dt.date,
) -> bool:
    """Return True if refresh should continue; False to stop (threshold reached)."""
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        listing_date = parse_date(entry.raw_date)
        if listing_date and listing_date < threshold_date:
            print(
                f'[stop] listing {listing_date.isoformat()} older than threshold {threshold_date.isoformat()}'
            )
            return False

        meta = ingest_meeting(
            meeting_url=entry.url,
            out_root=str(out_root),
            with_audio=with_audio,
            headless=headless,
            rps=rps,
            force=True,
        )
        print(
            f'[refresh] {meta.meeting_id} -> {meta.items_count} items, '
            f'{meta.attachments_count_total} attachments'
        )

        meta_date = parse_date(meta.datetime_text)
        if meta_date and meta_date < threshold_date:
            print(
                f'[stop] meta {meta_date.isoformat()} older than threshold {threshold_date.isoformat()}'
            )
            return False

    return True


def _main():
    ap = argparse.ArgumentParser(description='Refresh recent meetings for a city')
    ap.add_argument('--url', required=True, help='Frontpage URL (e.g. https://dagsordener.aarhus.dk/)')
    ap.add_argument('--out', required=True, help='Output city root (e.g. data/raw/meetings/aarhus)')
    ap.add_argument('--days', type=int, default=30, help='How many days back to refresh (default 30)')
    ap.add_argument('--no-audio', action='store_true', help='Do not download audio MP3s')
    ap.add_argument('--headful', action='store_true', help='Show browser (not headless)')
    ap.add_argument('--rps', type=float, default=DEFAULT_RPS, help='Max requests per second for downloads')
    args = ap.parse_args()

    threshold = dt.date.today() - dt.timedelta(days=args.days)
    years = list_available_years(args.url, headless=not args.headful)
    if not years:
        print('[warn] No years discovered; nothing to refresh')
        return

    years_sorted = sorted(years, reverse=True)
    print(f'[info] Refresh threshold: {threshold.isoformat()} (days={args.days})')
    print(f'[info] Evaluating years (newest first): {", ".join(map(str, years_sorted))}')

    for year in years_sorted:
        entries = collect_year_entries(args.url, year, headless=not args.headful)
        if not entries:
            print(f'[info] Year {year}: no entries discovered')
            continue
        print(f'[info] Year {year}: refreshing {len(entries)} meetings')
        should_continue = refresh_entries(
            entries,
            out_root=Path(args.out),
            with_audio=not args.no_audio,
            headless=not args.headful,
            rps=args.rps,
            threshold_date=threshold,
        )
        if not should_continue:
            break


if __name__ == '__main__':
    _main()
