"""Parse meeting HTML into structured dicts."""

import re
from datetime import datetime
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
from bs4.element import Tag
from markdownify import markdownify as md

BASE_URL = "https://dagsordener.aarhus.dk"


def normalize_url(url: str, base: str = BASE_URL) -> str:
    """Normalize URL: make absolute, clean up query params."""
    if not url:
        return ""
    # Make absolute
    full = urljoin(base, url)
    # Parse and rebuild clean
    parsed = urlparse(full)
    # Remove trailing slashes, lowercase scheme/host
    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path.rstrip("/") or "/",
        parsed.params,
        parsed.query,
        ""  # drop fragment
    ))


def is_empty_content(el: Tag) -> bool:
    """True if element has no meaningful text content."""
    text = el.get_text().replace("\xa0", " ").strip()
    return not text


def clean_html_before_markdown(soup: BeautifulSoup) -> None:
    """Remove empty elements and demote headings to plain paragraphs."""
    # Headings are usually low-signal ("Beslutninger", "Bilag") and often empty; treat as plain text.
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if not isinstance(h, Tag):
            continue
        if is_empty_content(h):
            h.decompose()

    for tag in soup.find_all(["p", "li"]):
        if not isinstance(tag, Tag):
            continue
        if is_empty_content(tag):
            tag.decompose()


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
    udvalg = get_text(soup, "span.udvalg")
    dato_text = get_text(soup, "span.dato")
    sted = get_text(soup, "span.sted")
    dt = parse_danish_datetime(dato_text)

    # Parse each punkt
    punkter = []
    rows = soup.select("tr.punktrow")

    for row in rows:
        punkt = parse_punkt_row(row, meeting_id, udvalg, dt, sted, meeting_type)
        if punkt:
            punkter.append(punkt)

    return punkter


def parse_punkt_row(row, meeting_id: str, udvalg: str, dt: datetime | None,
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

    # Remove dropdown menus, badges, sagsnummer (extracted as metadata), and other UI noise
    for el in content_soup.select(".dropdown, .dropdown-menu, .expand, .label, .badge, .pdf-label, .lydfilnavn, .sagsnummer"):
        el.decompose()

    # Convert audio embeds to links
    for audio in content_soup.select("audio"):
        src_attr = audio.get("src")
        if not src_attr:
            source_tag = audio.select_one("source")
            if source_tag:
                src_attr = source_tag.get("src")

        href = None
        if isinstance(src_attr, str):
            href = src_attr
        elif isinstance(src_attr, list) and src_attr:
            href = src_attr[0]

        if href:
            link = content_soup.new_tag("a", href=href)
            link.string = "Lydfil"
            audio.replace_with(link)

    # Extract and index all links (removes URL noise from embeddings)
    links_list = []
    for a in content_soup.select("a[href]"):
        href_attr = a.get("href")

        href = None
        if isinstance(href_attr, str):
            href = href_attr
        elif isinstance(href_attr, list) and href_attr:
            href = href_attr[0]

        if href and not href.startswith("#") and not href.startswith("javascript:"):
            normalized = normalize_url(href)
            if normalized not in links_list:
                links_list.append(normalized)
            idx = links_list.index(normalized)
            # Replace with markdown-style indexed link: [text](idx)
            link_text = a.get_text(strip=True) or f"Link {idx}"
            a.replace_with(f"[{link_text}]({idx})")

    clean_html_before_markdown(content_soup)

    # Convert to markdown
    content_md = md(str(content_soup), heading_style="ATX", strip=["script", "style"])
    content_md = promote_emphasis_headers(content_md)
    content_md = clean_markdown(content_md)

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
        "links": links_list,  # URLs extracted, not in embedding
        "meeting_id": meeting_id,
        "udvalg": udvalg,
        "datetime": dt.isoformat() if dt else None,
        "sted": sted,
        "type": meeting_type,
    }


def get_text(soup: BeautifulSoup, selector: str) -> str:
    """Get text from first matching element."""
    el = soup.select_one(selector)
    return el.get_text(strip=True) if el else ""


def parse_danish_datetime(text: str) -> datetime | None:
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


def clean_markdown(text: str) -> str:
    """Clean up markdown text."""
    # Remove per-line trailing white space and excessive blank lines
    text = re.sub(r"[^\S\n]*\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip document-level leading/trailing whitespace
    text = text.strip()
    return text


def promote_emphasis_headers(text: str) -> str:
    """Convert bold/italic-only lines into H5 headers."""
    pattern = re.compile(r"(?m)^\s*\*{1,2}([^\n]+?)\*{1,2}\s*$")

    def replace(match: re.Match) -> str:
        content = match.group(1).strip()
        return f"##### {content}" if content else match.group(0)

    return pattern.sub(replace, text)
