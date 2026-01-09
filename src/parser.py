"""Parse meeting HTML into structured dicts."""

import re
from datetime import datetime
from bs4 import BeautifulSoup
from markdownify import markdownify as md

BASE_URL = "https://dagsordener.aarhus.dk"


def parse_meeting(html: str, meeting_url: str) -> list[dict] | None:
    """
    Parse meeting HTML and return list of punkt dicts.
    Returns None if not a 'referat' (only process referater).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Extract meeting ID from URL
    match = re.search(r"id=([a-f0-9-]+)", meeting_url)
    if not match:
        return None
    meeting_id = match.group(1)

    # Check if referat or dagsorden
    title_el = soup.select_one("h1.dato")
    if not title_el:
        return None

    title_text = title_el.get_text(strip=True).lower()
    if "referat" in title_text:
        meeting_type = "referat"
    elif "dagsorden" in title_text:
        meeting_type = "dagsorden"
    else:
        meeting_type = "unknown"

    # Only process referater
    if meeting_type != "referat":
        return None

    # Extract meeting metadata
    udvalg = _get_text(soup, "span.udvalg")
    dato_text = _get_text(soup, "span.dato")
    sted = _get_text(soup, "span.sted")
    dt = _parse_danish_datetime(dato_text)

    # Parse each punkt
    punkter = []
    rows = soup.select("tr.punktrow")

    for row in rows:
        punkt = _parse_punkt_row(row, meeting_id, udvalg, dt, sted, meeting_type)
        if punkt:
            punkter.append(punkt)

    return punkter


def _parse_punkt_row(row, meeting_id: str, udvalg: str, dt: datetime | None,
                      sted: str, meeting_type: str) -> dict | None:
    """Parse a single punkt row into a dict."""
    # Get punkt ID from row id="punktrow_{uuid}"
    row_id = row.get("id", "")
    match = re.search(r"punktrow_([a-f0-9-]+)", row_id)
    if not match:
        return None
    punkt_id = match.group(1)

    # Get index from label
    index_el = row.select_one("span.label")
    index = int(index_el.get_text(strip=True)) if index_el else 0

    # Get title
    title_el = row.select_one("h2.overskrift")
    title = title_el.get_text(strip=True) if title_el else ""

    # Get sagsnummer (may not exist)
    sags_el = row.select_one("span.sagsnummer")
    sagsnummer = sags_el.get_text(strip=True) if sags_el else None

    # Get content from .details .punkt
    details = row.select_one(".details")
    if not details:
        return None

    # Clone to avoid modifying original
    content_soup = BeautifulSoup(str(details), "html.parser")

    # Remove dropdown menus and other UI noise
    for el in content_soup.select(".dropdown, .dropdown-menu, .expand"):
        el.decompose()

    # Convert audio embeds to links
    for audio in content_soup.select("audio"):
        source = audio.select_one("source")
        if source and source.get("src"):
            src = source["src"]
            link = content_soup.new_tag("a", href=src)
            link.string = "Lydfil"
            audio.replace_with(link)

    # Convert to markdown
    content_md = md(str(content_soup), heading_style="ATX", strip=["script", "style"])
    content_md = _clean_markdown(content_md)

    # Build canonical URL
    punkt_url = f"{BASE_URL}/vis?id={meeting_id}&punktid={punkt_id}"

    # Build chunk text for embedding
    chunk_parts = []
    chunk_parts.append(f"Udvalg: {udvalg}")
    if dt:
        chunk_parts.append(f"Dato: {dt.strftime('%Y-%m-%d kl. %H:%M')}")
    if sted:
        chunk_parts.append(f"Sted: {sted}")
    if sagsnummer:
        chunk_parts.append(f"Sagsnummer: {sagsnummer}")
    chunk_parts.append("")
    chunk_parts.append(f"# {title}")
    chunk_parts.append("")
    chunk_parts.append(content_md)

    chunk_text = "\n".join(chunk_parts)

    return {
        "punkt_id": punkt_id,
        "url": punkt_url,
        "index": index,
        "title": title,
        "sagsnummer": sagsnummer,
        "content_md": content_md,
        "chunk_text": chunk_text,
        "meeting_id": meeting_id,
        "udvalg": udvalg,
        "datetime": dt.isoformat() if dt else None,
        "sted": sted,
        "type": meeting_type,
    }


def _get_text(soup: BeautifulSoup, selector: str) -> str:
    """Get text from first matching element."""
    el = soup.select_one(selector)
    return el.get_text(strip=True) if el else ""


def _parse_danish_datetime(text: str) -> datetime | None:
    """Parse Danish datetime string like 'onsdag den 1. oktober 2025 kl. 15.45'"""
    if not text:
        return None

    # Danish month names
    months = {
        "januar": 1, "februar": 2, "marts": 3, "april": 4,
        "maj": 5, "juni": 6, "juli": 7, "august": 8,
        "september": 9, "oktober": 10, "november": 11, "december": 12
    }

    # Pattern: day. month year kl. HH.MM
    pattern = r"(\d{1,2})\.\s+(\w+)\s+(\d{4})\s+kl\.\s*(\d{1,2})\.(\d{2})"
    match = re.search(pattern, text.lower())
    if not match:
        return None

    day = int(match.group(1))
    month_name = match.group(2)
    year = int(match.group(3))
    hour = int(match.group(4))
    minute = int(match.group(5))

    month = months.get(month_name)
    if not month:
        return None

    return datetime(year, month, day, hour, minute)


def _clean_markdown(text: str) -> str:
    """Clean up markdown text."""
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text
