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
    """Return all meeting URLs for the given year by loading all results.

    Strategy:
    - Load year-filtered page
    - Prefer clicking a visible "Vis flere" (Load more) button until it disappears or is disabled
    - Fallback: scroll a results container (or window) and wait for count increases
    - Stop when height and count stabilize across several iterations or after a max loop guard
    """
    with browser_context(headless=headless) as ctx:
        page = ctx.new_page()
        page.set_default_timeout(30000)
        page.goto(_year_url(year))
        # Give initial content time to render
        try:
            page.wait_for_selector(
                '#resultater a.searchresult, a.list-group-item.searchresult', timeout=10000
            )
        except PWTimeout:
            # continue; we'll rely on scroll/idle waits below
            pass

        # Helper: current anchors locator and count
        def anchors_locator():
            loc = page.locator('#resultater a.searchresult')
            return loc if loc.count() > 0 else page.locator('a.list-group-item.searchresult')

        def anchors_count() -> int:
            return anchors_locator().count()

        # Try to find a "Vis flere" element that loads more results
        def load_more_once() -> bool:
            # Common patterns: text=Vis flere, buttons/links with "visMere" or "load more"
            for sel in [
                "button:has-text('Vis flere')",
                "a:has-text('Vis flere')",
                '#visMere',
                '.load-more',
            ]:
                btn = page.locator(sel).first
                if btn and btn.count() > 0 and btn.is_enabled() and btn.is_visible():
                    btn.click()
                    return True
            return False

        # Loop until stable or iteration cap; track height and count
        stable_rounds = 0
        prev_count = -1
        prev_height = page.evaluate('() => document.body.scrollHeight')
        max_loops = 400
        for _ in range(max_loops):
            before = anchors_count()

            # Prefer clicking explicit load-more if available
            clicked = False
            try:
                clicked = load_more_once()
            except PWTimeout:
                clicked = False

            if not clicked:
                # Fallback: scroll results container or window (multiple nudges)
                try:
                    page.evaluate(
                        '() => {'
                        "const el = document.querySelector('#resultater');"
                        ' if (el) { el.scrollTo(0, el.scrollHeight); }'
                        ' else { window.scrollTo(0, document.body.scrollHeight); }'
                        '}'
                    )
                except Exception:
                    pass
                # Nudge with End key as well
                try:
                    page.keyboard.press('End')
                except Exception:
                    pass

            # Wait longer for new content (site can be slow)
            try:
                page.wait_for_load_state('networkidle', timeout=5000)
            except PWTimeout:
                page.wait_for_timeout(1200)

            # Wait for either count increase or height increase
            try:
                page.wait_for_function(
                    ' (prev) => {'
                    " const q = document.querySelectorAll('#resultater a.searchresult, "
                    "a.list-group-item.searchresult');"
                    ' return q.length > prev;'
                    ' }',
                    arg=before,
                    timeout=4000,
                )
            except PWTimeout:
                # As a fallback, detect DOM height growth
                try:
                    page.wait_for_function(
                        '(h) => document.body.scrollHeight > h',
                        arg=prev_height,
                        timeout=4000,
                    )
                except PWTimeout:
                    pass

            now_count = anchors_count()
            now_height = page.evaluate('() => document.body.scrollHeight')
            if now_count == prev_count and now_height == prev_height:
                stable_rounds += 1
            else:
                stable_rounds = 0
            if stable_rounds >= 5:
                break
            prev_count = now_count
            prev_height = now_height

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
