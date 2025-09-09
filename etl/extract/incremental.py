#!/usr/bin/env python3
"""
Incremental fetcher: visit frontpage and ingest latest meetings.

What it does:
- Opens https://dagsordener.aarhus.dk/ (Playwright; dynamic DOM)
- Extracts the 10 meeting links shown on the frontpage list
- Calls etl.extract.ingestion.ingest_meeting() for each
- Skips meetings that already exist under <out_root>/<meeting-id>/meta.json

Usage:
  python -m etl.extract.incremental \
    --out data/raw/meetings \
    --limit 10

Notes:
- This script focuses only on the frontpage latest list (incremental).
- Backfill/crawling across the site will be handled separately.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout

from etl.extract.ingestion import (
    BASE,
    DEFAULT_RPS,
    browser_context,
    ingest_meeting,
    parse_meeting_id_from_url,
)


def normalize_url(href: str) -> str:
    if not href:
        return href
    if href.startswith('http://') or href.startswith('https://'):
        return href
    # site uses root-relative hrefs
    return BASE.rstrip('/') + href


def extract_frontpage_links(limit: int = 10, headless: bool = True) -> list[str]:
    """Return up to `limit` absolute meeting URLs from the frontpage.

    Selectors are based on current DOM where latest meetings are rendered as
    anchors with class `list-group-item searchresult` inside `#resultater`.
    """
    with browser_context(headless=headless) as ctx:
        page = ctx.new_page()
        page.set_default_timeout(15000)
        page.goto(BASE + '/')
        # Wait for the dynamic list to appear
        try:
            page.wait_for_selector('#resultater a.searchresult', state='visible', timeout=20000)
        except PWTimeout:
            # fallback to a broader selector
            page.wait_for_selector('a.list-group-item.searchresult', timeout=20000)

        anchors = page.locator('#resultater a.searchresult')
        if anchors.count() == 0:
            anchors = page.locator('a.list-group-item.searchresult')

        hrefs: list[str] = []
        count = anchors.count()
        for i in range(min(limit, count)):
            a = anchors.nth(i)
            href = (a.get_attribute('href') or '').strip()
            if href:
                hrefs.append(normalize_url(href))
        return hrefs


def ingest_links(
    links: Iterable[str],
    *,
    out_root: Path,
    with_audio: bool,
    headless: bool,
    rps: float,
) -> list[str]:
    """Ingest each meeting URL unless it already exists. Returns list of ingested IDs."""
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ingested: list[str] = []
    for url in links:
        mid = parse_meeting_id_from_url(url)
        if mid:
            meta_path = out_root / mid / 'meta.json'
            if meta_path.exists():
                print(f'[skip] {mid} already ingested')
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
        except Exception as e:  # keep going on errors
            print(f'[error] ingest failed for url={url}: {e}')
    return ingested


def _main():
    ap = argparse.ArgumentParser(description='Incremental frontpage fetch (latest 10 meetings)')
    ap.add_argument(
        '--out', default='data/raw/meetings', help='Output root (default: data/raw/meetings)'
    )
    ap.add_argument(
        '--limit', type=int, default=10, help='How many frontpage links to process (default 10)'
    )
    ap.add_argument('--no-audio', action='store_true', help='Do not download audio MP3s')
    ap.add_argument('--headful', action='store_true', help='Show browser (not headless)')
    ap.add_argument(
        '--rps', type=float, default=DEFAULT_RPS, help='Max requests per second for downloads'
    )
    ap.add_argument(
        '--print-only', action='store_true', help='Only print discovered links; do not ingest'
    )
    args = ap.parse_args()

    links = extract_frontpage_links(limit=args.limit, headless=not args.headful)
    if args.print_only:
        for u in links:
            print(u)
        return

    ingest_links(
        links,
        out_root=Path(args.out),
        with_audio=not args.no_audio,
        headless=not args.headful,
        rps=args.rps,
    )


if __name__ == '__main__':
    _main()
