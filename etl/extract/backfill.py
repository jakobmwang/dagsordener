#!/usr/bin/env python3
"""
Backfill fetcher: visit the year-filtered list and ingest all meetings.

What it does:
- Loads https://dagsordener.aarhus.dk/?request.kriterie.udvalgId=&request.kriterie.moedeDato=<YEAR>
- Scrolls to the bottom repeatedly to trigger lazy loading until no new results appear
- Extracts all meeting links on the page
- Calls etl.extract.ingestion.ingest_meeting() for each, skipping existing (meta.json present)

Usage:
  python -m etl.extract.backfill \
    --year 2025 \
    --out data/raw/meetings
"""

from __future__ import annotations

import argparse
import datetime as dt
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
    return BASE.rstrip('/') + href


def _year_url(year: int) -> str:
    return f'{BASE}/?request.kriterie.udvalgId=&request.kriterie.moedeDato={year}'


def extract_year_links(year: int, *, headless: bool = True) -> list[str]:
    """Return all meeting URLs for the given year by scrolling to load more."""
    with browser_context(headless=headless) as ctx:
        page = ctx.new_page()
        page.set_default_timeout(20000)
        page.goto(_year_url(year))

        # Primary locator used on the site; keep fallback for robustness
        def anchors_locator():
            loc = page.locator('#resultater a.searchresult')
            return loc if loc.count() > 0 else page.locator('a.list-group-item.searchresult')

        # Scroll loop: continue until no increase in count for a few iterations
        stable_rounds = 0
        prev_count = -1
        while True:
            loc = anchors_locator()
            count = loc.count()
            if count == prev_count:
                stable_rounds += 1
            else:
                stable_rounds = 0
            if stable_rounds >= 3:
                break

            # Scroll to bottom and give time for new results to load
            page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            try:
                page.wait_for_load_state('networkidle', timeout=3000)
            except PWTimeout:
                page.wait_for_timeout(500)
            prev_count = count

        # Collect hrefs
        loc = anchors_locator()
        hrefs: list[str] = []
        for i in range(loc.count()):
            a = loc.nth(i)
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
    ap = argparse.ArgumentParser(description='Backfill a year of meetings')
    ap.add_argument('--year', type=int, default=dt.date.today().year, help='Year to backfill')
    ap.add_argument(
        '--out', default='data/raw/meetings', help='Output root (default: data/raw/meetings)'
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

    links = extract_year_links(args.year, headless=not args.headful)
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
