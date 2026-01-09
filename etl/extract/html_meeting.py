from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

USER_AGENT = 'dagsordener-html-ingester/0.1'

RE_GUID = re.compile(
    r'([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})'
)


@dataclass
class MeetingItem:
    item_id: str
    index: int | None
    title: str | None
    case_number: str | None
    html: str
    markdown: str


@dataclass
class MeetingData:
    meeting_id: str
    meeting_url: str
    kind: str | None
    committee: str | None
    place: str | None
    datetime_text: str | None
    items: list[MeetingItem]


@contextlib.contextmanager
def browser_context(headless: bool = True):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=USER_AGENT,
            java_script_enabled=True,
            viewport={'width': 1366, 'height': 900},
        )
        try:
            yield context
        finally:
            context.close()
            browser.close()


def parse_meeting_id_from_url(url: str) -> str | None:
    match = re.search(r'[?&]id=' + RE_GUID.pattern, url)
    if match:
        return match.group(1)
    match = RE_GUID.search(url)
    return match.group(1) if match else None


def canonical_meeting_url(url: str) -> tuple[str, str]:
    meeting_id = parse_meeting_id_from_url(url)
    if not meeting_id:
        raise ValueError(f'Cannot find meeting id in URL: {url!r}')
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f'Meeting URL must be absolute (got: {url!r})')
    origin = f'{parsed.scheme}://{parsed.netloc}'
    return meeting_id, f'{origin}/vis?id={meeting_id}'


def _absolute_origin(url: str) -> str:
    parsed = urlparse(url)
    return f'{parsed.scheme}://{parsed.netloc}'


def fetch_meeting_html(meeting_url: str, *, headless: bool = True) -> str:
    with browser_context(headless=headless) as ctx:
        page = ctx.new_page()
        page.set_default_timeout(30000)
        page.goto(meeting_url, wait_until='networkidle')
        try:
            page.wait_for_selector('#dagsordenDetaljer tr.punktrow', timeout=15000)
        except PWTimeout:
            page.goto(meeting_url, wait_until='domcontentloaded')
            page.wait_for_selector('#dagsordenDetaljer tr.punktrow', timeout=15000)
        return page.content()


def _clean_item_html(html: str, origin: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')

    for tag in soup.select('span.sagsnummer'):
        tag.decompose()

    for tag in soup.select('.dropdown, .dropdown-menu, button.dropdown-toggle'):
        tag.decompose()

    for audio in soup.find_all('audio'):
        source = audio.get('src')
        if not source:
            source_tag = audio.find('source')
            if source_tag:
                source = source_tag.get('src')
        if source:
            href = urljoin(origin, source)
            anchor = soup.new_tag('a', href=href)
            anchor.string = 'Lydfil'
            audio.replace_with(anchor)
        else:
            audio.decompose()

    for source in soup.find_all('source'):
        src = source.get('src')
        if not src:
            source.decompose()
            continue
        anchor = soup.new_tag('a', href=urljoin(origin, src))
        anchor.string = 'Lydfil'
        source.replace_with(anchor)

    for tag in soup.find_all(['a', 'img']):
        attr = 'href' if tag.name == 'a' else 'src'
        value = tag.get(attr)
        if value:
            tag[attr] = urljoin(origin, value)

    return str(soup)


def _extract_item_id(row: BeautifulSoup) -> str | None:
    row_id = row.get('id', '')
    match = RE_GUID.search(row_id)
    if match:
        return match.group(1)

    link = row.select_one("a[data-type='kopier_punkt']")
    if link:
        data_id = link.get('data-id')
        if data_id and RE_GUID.fullmatch(data_id):
            return data_id

    return None


def parse_meeting_html(html: str, meeting_url: str) -> MeetingData:
    meeting_id, canonical_url = canonical_meeting_url(meeting_url)
    origin = _absolute_origin(canonical_url)
    soup = BeautifulSoup(html, 'html.parser')

    def pick_text(selector: str) -> str | None:
        tag = soup.select_one(selector)
        if not tag:
            return None
        text = tag.get_text(strip=True)
        return text or None

    kind = None
    title_text = pick_text('h1.green.dato')
    if title_text:
        lowered = title_text.lower()
        if 'referat' in lowered:
            kind = 'referat'
        elif 'dagsorden' in lowered:
            kind = 'dagsorden'

    committee = pick_text('.title .udvalg') or pick_text('table.dagsordeninfo .udvalg')
    place = pick_text('.title .sted') or pick_text('table.dagsordeninfo .sted')
    datetime_text = pick_text('table.dagsordeninfo .dato')

    items: list[MeetingItem] = []
    for row in soup.select('#dagsordenDetaljer tr.punktrow'):
        item_id = _extract_item_id(row)
        if not item_id:
            continue

        index = None
        idx_tag = row.select_one('.punkt-tabel .label')
        if idx_tag:
            with contextlib.suppress(Exception):
                index = int(idx_tag.get_text(strip=True))

        title = None
        title_tag = row.select_one('.overskrift')
        if title_tag:
            title = title_tag.get_text(strip=True) or None

        case_number = None
        case_tag = row.select_one('.details .sagsnummer')
        if case_tag:
            case_number = case_tag.get_text(strip=True) or None

        details = row.select_one('.details')
        details_html = str(details) if details else str(row)
        cleaned_html = _clean_item_html(details_html, origin)
        markdown = md(cleaned_html, heading_style='ATX')

        items.append(
            MeetingItem(
                item_id=item_id,
                index=index,
                title=title,
                case_number=case_number,
                html=cleaned_html,
                markdown=markdown.strip(),
            )
        )

    return MeetingData(
        meeting_id=meeting_id,
        meeting_url=canonical_url,
        kind=kind,
        committee=committee,
        place=place,
        datetime_text=datetime_text,
        items=items,
    )


def load_meeting(meeting_url: str, *, headless: bool = True) -> MeetingData:
    html = fetch_meeting_html(meeting_url, headless=headless)
    return parse_meeting_html(html, meeting_url)


def load_meetings(urls: Iterable[str], *, headless: bool = True) -> list[MeetingData]:
    meetings: list[MeetingData] = []
    for url in urls:
        meetings.append(load_meeting(url, headless=headless))
    return meetings
