"""CLI entry point for dagsordener sync."""

import argparse
import re

from src.scraper import (
    create_browser_context,
    close_browser,
    get_meeting_links,
    get_meeting_html,
    get_available_years,
    get_year_meeting_links,
)
from src.parser import parse_meeting
from src.embedder import Embedder
from src.qdrant import URL, get_client, ensure_collection, meeting_exists, upsert_punkter


def process_meeting(page, url: str, embedder: Embedder, qdrant_client) -> int:
    """Process a single meeting. Returns number of punkter upserted."""
    # Extract meeting_id from URL
    match = re.search(r"id=([a-f0-9-]+)", url)
    if not match:
        return 0
    meeting_id = match.group(1)

    # Check if already exists
    if meeting_exists(qdrant_client, meeting_id):
        print(f"  Skip (exists): {meeting_id[:8]}...")
        return 0

    # Fetch and parse
    try:
        html = get_meeting_html(page, url)
    except Exception as e:
        print(f"  Error fetching {meeting_id[:8]}: {e}")
        return 0

    punkter = parse_meeting(html, url)

    if punkter is None:
        print(f"  Skip (not referat): {meeting_id[:8]}...")
        return 0

    if not punkter:
        print(f"  Skip (no punkter): {meeting_id[:8]}...")
        return 0

    # Embed
    texts = [p["chunk_text"] for p in punkter]
    embeddings = embedder.embed(texts)

    # Upsert
    count = upsert_punkter(qdrant_client, punkter, embeddings)
    print(f"  Upserted {count} punkter from {punkter[0]['udvalg']}")

    return count


def run_incremental(args):
    """Check frontpage for new meetings and process them."""
    print("Running incremental sync...")

    playwright, browser, page = create_browser_context()
    qdrant_client = get_client(args.qdrant_url)
    ensure_collection(qdrant_client)
    embedder = Embedder()

    try:
        urls = get_meeting_links(page)
        print(f"Found {len(urls)} meetings on frontpage")

        total = 0
        for url in urls:
            total += process_meeting(page, url, embedder, qdrant_client)

        print(f"Done. Upserted {total} punkter total.")

    finally:
        close_browser(playwright, browser)


def run_backfill(args):
    """Process all meetings from all years."""
    print("Running backfill sync...")

    playwright, browser, page = create_browser_context()
    qdrant_client = get_client(args.qdrant_url)
    ensure_collection(qdrant_client)
    embedder = Embedder()

    try:
        if args.year:
            years = [args.year]
        else:
            years = get_available_years(page)

        print(f"Processing years: {years}")

        total = 0
        for year in years:
            print(f"\n=== Year {year} ===")
            urls = get_year_meeting_links(page, year)
            print(f"Found {len(urls)} meetings")

            for url in urls:
                total += process_meeting(page, url, embedder, qdrant_client)

        print(f"\nDone. Upserted {total} punkter total.")

    finally:
        close_browser(playwright, browser)


def main():
    parser = argparse.ArgumentParser(description="Sync dagsordener to Qdrant")
    parser.add_argument(
        "--mode",
        choices=["incremental", "backfill"],
        default="incremental",
        help="Sync mode: incremental (frontpage) or backfill (all years)",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Specific year to backfill (only with --mode backfill)",
    )
    parser.add_argument(
        "--qdrant-url",
        default=URL,
        help="Qdrant URL",
    )
    args = parser.parse_args()

    if args.mode == "incremental":
        run_incremental(args)
    else:
        run_backfill(args)


if __name__ == "__main__":
    main()
