#!/usr/bin/env python3
"""
Ingestion module: fetch all agreed files for a single meeting (by URL or id)

Project layout (expected):
  /home/waja/code/dagsordener/
    etl/
      extract/ingestion.py          <- this file
    data/
      raw/                          <- output root (default)

What it does (MVP, lean):
- Loads a meeting page (Playwright) and parses:
  • meeting_id, kind (dagsorden|referat), committee, place, datetime_text
  • items: item_id, index, title, case_number
  • attachments per item: (attachment_id, title, url)
  • audio per item: list of mp3 URLs (if present)
- Downloads (httpx, with rate-limit):
  • full.pdf (per meeting): /vis/pdf/dagsorden/<meeting-id>?redirectDirectlyToPdf=true
  • item.pdf (per item):    /vis/pdf/dagsordenpunkt/<item-id>?redirectDirectlyToPdf=true
  • attachments (per item): /vis/pdf/bilag/<attachment-id>?redirectDirectlyToPdf=true
  • audio (per item):       direct mp3 URLs
- Writes meta.json per meeting with sizes + sha256 + paths.

Notes:
- Simple, sequential MVP (no concurrency); easy to extend later.
- Idempotent-ish: if a file already exists, we compute sha256/size and skip re-download.
- Robust selectors based on examples provided by @waja.

Dependencies:
  pip install playwright httpx pydantic
  playwright install

CLI example:
  python -m etl.extract.ingestion \
    --url "https://dagsordener.aarhus.dk/vis?…&id=<GUID>" \
    --out /home/waja/code/dagsordener/data/raw/meetings
  # or
  python -m etl.extract.ingestion \
    --id <GUID> \
    --kind referat \
    --out /home/waja/code/dagsordener/data/raw/meetings
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

BASE = 'https://dagsordener.aarhus.dk'
USER_AGENT = 'dagsordener-ingester/0.1 (+https://aarhus.dk)'
DEFAULT_RPS = 1.5  # ~1-2 req/s

# --------------------------- utils ---------------------------


class RateLimiter:
    def __init__(self, rps: float = DEFAULT_RPS):
        self.min_interval = 1.0 / max(rps, 0.01)
        self._last = 0.0

    def wait(self):
        now = time.time()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.time()


RE_GUID = re.compile(
    r'([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})'
)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            total += len(chunk)
            h.update(chunk)
    return h.hexdigest(), total


def parse_meeting_id_from_url(url: str) -> str | None:
    # Looks for "&id=<GUID>" or generic GUID anywhere in URL
    m = re.search(r'[?&]id=' + RE_GUID.pattern, url)
    if m:
        return m.group(1)
    m2 = RE_GUID.search(url)
    return m2.group(1) if m2 else None


# --------------------------- datamodel ---------------------------


@dataclass
class FileRef:
    url: str
    path: str
    sha256: str | None = None
    size: int | None = None
    mime: str | None = None


@dataclass
class AttachmentMeta:
    attachment_id: str
    title: str
    file: FileRef


@dataclass
class AudioMeta:
    audio_id: str
    title: str | None
    file: FileRef


@dataclass
class ItemMeta:
    item_id: str
    index: int | None
    title: str | None
    case_number: str | None
    item_pdf: FileRef
    attachments: list[AttachmentMeta] = field(default_factory=list)
    audio: list[AudioMeta] = field(default_factory=list)


@dataclass
class MeetingMeta:
    meeting_id: str
    meeting_url: str
    kind: str | None
    committee: str | None
    place: str | None
    datetime_text: str | None
    full_pdf: FileRef
    items: list[ItemMeta]
    items_count: int
    attachments_count_total: int
    audio_count_total: int
    fetched_at: str
    errors: list[dict] = field(default_factory=list)


# --------------------------- browser helpers ---------------------------


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


# --------------------------- DOM parsing ---------------------------


def parse_meeting_dom(page) -> dict:
    """Return dict with meeting fields + list of items (sans downloads)."""
    result: dict = {
        'meeting_id': None,
        'kind': None,
        'committee': None,
        'place': None,
        'datetime_text': None,
        'items': [],
    }
    # H1-kind
    with contextlib.suppress(Exception):
        h1 = page.locator('h1.green.dato').first
        if h1 and h1.count() > 0:
            txt = h1.inner_text().strip()
            if txt:
                if 'Referat' in txt:
                    result['kind'] = 'referat'
                elif 'Dagsorden' in txt:
                    result['kind'] = 'dagsorden'
    # Committee/place
    with contextlib.suppress(Exception):
        com = page.locator('.title .udvalg').first
        if com and com.count() > 0:
            result['committee'] = com.inner_text().strip()
    with contextlib.suppress(Exception):
        pl = page.locator('.title .sted').first
        if pl and pl.count() > 0:
            result['place'] = pl.inner_text().strip()
    # Date/time (raw text)
    with contextlib.suppress(Exception):
        dt_cell = page.locator('table.dagsordeninfo .dato').first
        if dt_cell and dt_cell.count() > 0:
            result['datetime_text'] = dt_cell.inner_text().strip()

    # meeting_id via hidden button or URL in DOM
    with contextlib.suppress(Exception):
        btn = page.locator('#hentHeledagsorden').first
        if btn and btn.count() > 0:
            did = btn.get_attribute('data-id')
            if did and RE_GUID.fullmatch(did):
                result['meeting_id'] = did

    # Items
    rows = page.locator('#dagsordenDetaljer tr.punktrow')
    for i in range(rows.count()):
        row = rows.nth(i)
        item_id = None
        with contextlib.suppress(Exception):
            rid = row.get_attribute('id') or ''
            m = re.search(r'punktrow_' + RE_GUID.pattern, rid)
            if m:
                item_id = m.group(1)
        # number
        idx = None
        with contextlib.suppress(Exception):
            lab = row.locator('.punkt-col .label').first
            if lab and lab.count() > 0:
                try:
                    idx = int(lab.inner_text().strip())
                except Exception:
                    idx = None
        # title
        title = None
        with contextlib.suppress(Exception):
            h2 = row.locator('.overskrift').first
            if h2 and h2.count() > 0:
                title = h2.inner_text().strip()
        # case number
        case_number = None
        with contextlib.suppress(Exception):
            sn = row.locator('.details .sagsnummer').first
            if sn and sn.count() > 0:
                case_number = sn.inner_text().strip()
        # attachments (within row)
        attachments = []
        with contextlib.suppress(Exception):
            for a in row.locator('a.bilag-link[href]').all():
                href = a.get_attribute('href') or ''
                title_a = (a.inner_text() or '').strip()
                # find id via sibling span or enclosure id
                aid = None
                # Try span.pdf-label[data-id] adjacent
                with contextlib.suppress(Exception):
                    span = a.locator(
                        "xpath=preceding-sibling::span[contains(@class,'pdf-label')][1]"
                    ).first
                    if span and span.count() > 0:
                        val = span.get_attribute('data-id')
                        if val and RE_GUID.fullmatch(val):
                            aid = val
                # Try enclosure-<GUID> id on anchor
                if not aid:
                    with contextlib.suppress(Exception):
                        anid = a.get_attribute('id') or ''
                        m = re.search(RE_GUID, anid)
                        if m:
                            aid = m.group(1)
                # Fallback parse from href
                if not aid:
                    m = RE_GUID.search(href or '')
                    if m:
                        aid = m.group(1)
                if not href.startswith('http'):
                    href = BASE.rstrip('/') + href
                attachments.append(
                    {
                        'attachment_id': aid or hashlib.sha1(href.encode()).hexdigest()[:12],
                        'title': title_a,
                        'url': href,
                    }
                )
        # audio (mp3 URLs inside row)
        audio = []
        with contextlib.suppress(Exception):
            # anchors to mp3
            for a in row.locator("a[href$='.mp3']").all():
                href = a.get_attribute('href') or ''
                title_a = (a.inner_text() or '').strip() or None
                if href and not href.startswith('http'):
                    href = BASE.rstrip('/') + href
                audio.append(
                    {
                        'audio_id': RE_GUID.search(href).group(1)
                        if RE_GUID.search(href)
                        else hashlib.sha1(href.encode()).hexdigest()[:12],
                        'title': title_a,
                        'url': href,
                    }
                )
            # <audio src> or <source src>
            for sel in ['audio[src]', 'audio source[src]']:
                for el in row.locator(sel).all():
                    href = el.get_attribute('src') or ''
                    if href.endswith('.mp3'):
                        if not href.startswith('http'):
                            href = BASE.rstrip('/') + href
                        audio.append(
                            {
                                'audio_id': RE_GUID.search(href).group(1)
                                if RE_GUID.search(href)
                                else hashlib.sha1(href.encode()).hexdigest()[:12],
                                'title': None,
                                'url': href,
                            }
                        )
        result['items'].append(
            {
                'item_id': item_id or hashlib.sha1((title or str(i)).encode()).hexdigest()[:12],
                'index': idx,
                'title': title,
                'case_number': case_number,
                'attachments': attachments,
                'audio': audio,
            }
        )
    return result


# --------------------------- download helpers ---------------------------

MIME_EXT = {
    'application/pdf': '.pdf',
    'audio/mpeg': '.mp3',
}


def normalize_attachment_url(url: str) -> str:
    # flip redirectDirectlyToPdf to true when present/possible
    if 'redirectDirectlyToPdf=' in url:
        url = re.sub(r'redirectDirectlyToPdf=(?:false|0)', 'redirectDirectlyToPdf=true', url)
        return url
    # If URL is /vis/pdf/bilag/<id>/ add ?redirectDirectlyToPdf=true
    if '/vis/pdf/bilag/' in url and '?' not in url:
        url = url + '?redirectDirectlyToPdf=true'
    return url


def stream_download(
    client: httpx.Client,
    url: str,
    out_path: Path,
    rate: RateLimiter,
    headers: dict[str, str] | None = None,
) -> FileRef:
    ensure_dir(out_path.parent)
    # If exists, compute sha256 and size and return without re-downloading
    if out_path.exists():
        sha, size = sha256_file(out_path)
        return FileRef(url=url, path=str(out_path), sha256=sha, size=size, mime=None)
    rate.wait()
    with client.stream('GET', url, headers=headers or {}, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        mime = r.headers.get('Content-Type', '') or None
        h = hashlib.sha256()
        size = 0
        with open(out_path, 'wb') as f:
            for chunk in r.iter_bytes():
                size += len(chunk)
                h.update(chunk)
                f.write(chunk)
        sha = h.hexdigest()
    return FileRef(url=url, path=str(out_path), sha256=sha, size=size, mime=mime)


# --------------------------- core ingestion ---------------------------


def ingest_meeting(
    *,
    meeting_url: str | None = None,
    meeting_id: str | None = None,
    out_root: Path | str = 'data/raw/meetings',
    with_audio: bool = True,
    headless: bool = True,
    rps: float = DEFAULT_RPS,
) -> MeetingMeta:
    """
    Ingest a single meeting by URL or meeting_id.
    Writes files under <out_root>/<meeting-id>/ and returns MeetingMeta.
    """
    if not meeting_url and not meeting_id:
        raise ValueError('Provide meeting_url or meeting_id')

    out_root = Path(out_root)
    ensure_dir(out_root)

    errors: list[dict] = []

    with browser_context(headless=headless) as context:
        page = context.new_page()
        page.set_default_timeout(30000)
        resolved_url = meeting_url
        # If only meeting_id is provided, try to generate a view URL
        if not resolved_url and meeting_id:
            # Best effort: there's no stable route aside from search or bookmarks,
            # but in practice caller will pass meeting_url. We'll still proceed with downloads only.
            resolved_url = f'{BASE}/vis?id={meeting_id}'
        # Navigate & wait DOM
        if resolved_url:
            try:
                page.goto(resolved_url, wait_until='networkidle')
                page.wait_for_selector('#dagsordenDetaljer tr.punktrow', timeout=15000)
            except PWTimeout:
                # try domcontentloaded as fallback
                with contextlib.suppress(Exception):
                    page.goto(resolved_url, wait_until='domcontentloaded')
        # parse DOM
        parsed = parse_meeting_dom(page) if resolved_url else {'items': []}

        # ensure meeting_id
        mid = parsed.get('meeting_id') or meeting_id
        if not mid and meeting_url:
            mid = parse_meeting_id_from_url(meeting_url)
        if not mid:
            raise RuntimeError('Could not determine meeting_id')

        # Build output dirs
        meeting_dir = out_root / mid
        ensure_dir(meeting_dir)

        # HTTP client
        rate = RateLimiter(rps)
        headers = {'User-Agent': USER_AGENT, 'Referer': resolved_url or BASE}
        client = httpx.Client(headers=headers)

        try:
            # full.pdf
            full_url = f'{BASE}/vis/pdf/dagsorden/{mid}?redirectDirectlyToPdf=true'
            full_ref = stream_download(
                client, full_url, meeting_dir / 'full.pdf', rate, headers=headers
            )
        except Exception as e:
            errors.append({'stage': 'full.pdf', 'error': str(e)})
            # still proceed with items
            full_ref = FileRef(url=full_url if mid else '', path=str(meeting_dir / 'full.pdf'))

        # Items: build metas and download
        items_meta: list[ItemMeta] = []
        attachments_total = 0
        audio_total = 0
        for it in parsed.get('items', []):
            item_id = it.get('item_id')
            item_dir = (
                meeting_dir
                / 'items'
                / (item_id or hashlib.sha1((it.get('title') or 'x').encode()).hexdigest()[:12])
            )
            ensure_dir(item_dir)
            # item.pdf
            item_pdf_url = (
                f'{BASE}/vis/pdf/dagsordenpunkt/{item_id}?redirectDirectlyToPdf=true'
                if item_id
                else None
            )
            if item_pdf_url is None:
                # skip if we cannot construct URL
                item_ref = FileRef(url='', path=str(item_dir / 'item.pdf'))
                errors.append({'stage': 'item.pdf', 'item_id': item_id, 'error': 'no item_id'})
            else:
                try:
                    item_ref = stream_download(
                        client, item_pdf_url, item_dir / 'item.pdf', rate, headers=headers
                    )
                except Exception as e:
                    item_ref = FileRef(url=item_pdf_url, path=str(item_dir / 'item.pdf'))
                    errors.append({'stage': 'item.pdf', 'item_id': item_id, 'error': str(e)})

            # attachments
            att_metas: list[AttachmentMeta] = []
            for att in it.get('attachments', []):
                att_id = att.get('attachment_id')
                att_title = att.get('title')
                att_url = normalize_attachment_url(att.get('url'))
                # Path & ext
                outp = item_dir / 'attachments' / f'{att_id}.pdf'
                try:
                    file_ref = stream_download(client, att_url, outp, rate, headers=headers)
                except Exception as e:
                    file_ref = FileRef(url=att_url, path=str(outp))
                    errors.append(
                        {
                            'stage': 'attachment',
                            'item_id': item_id,
                            'attachment_id': att_id,
                            'error': str(e),
                        }
                    )
                att_metas.append(
                    AttachmentMeta(attachment_id=att_id, title=att_title, file=file_ref)
                )
            attachments_total += len(att_metas)

            # audio
            aud_metas: list[AudioMeta] = []
            if with_audio:
                for au in it.get('audio', []):
                    aurl = au.get('url')
                    aid = au.get('audio_id')
                    atitle = au.get('title')
                    # guess ext from URL or mime later
                    outp = item_dir / 'audio' / f'{aid}.mp3'
                    try:
                        file_ref = stream_download(client, aurl, outp, rate, headers=headers)
                    except Exception as e:
                        file_ref = FileRef(url=aurl, path=str(outp))
                        errors.append(
                            {'stage': 'audio', 'item_id': item_id, 'audio_id': aid, 'error': str(e)}
                        )
                    aud_metas.append(AudioMeta(audio_id=aid, title=atitle, file=file_ref))
            audio_total += len(aud_metas)

            items_meta.append(
                ItemMeta(
                    item_id=item_id,
                    index=it.get('index'),
                    title=it.get('title'),
                    case_number=it.get('case_number'),
                    item_pdf=item_ref,
                    attachments=att_metas,
                    audio=aud_metas,
                )
            )

        client.close()

    # Meeting meta
    meta = MeetingMeta(
        meeting_id=mid,
        meeting_url=resolved_url or f'{BASE}/vis?id={mid}',
        kind=parsed.get('kind'),
        committee=parsed.get('committee'),
        place=parsed.get('place'),
        datetime_text=parsed.get('datetime_text'),
        full_pdf=full_ref,
        items=items_meta,
        items_count=len(items_meta),
        attachments_count_total=attachments_total,
        audio_count_total=audio_total,
        fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        errors=errors,
    )

    # Write meta.json
    meta_path = meeting_dir / 'meta.json'
    ensure_dir(meta_path.parent)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    return meta


# --------------------------- CLI ---------------------------


def _main():
    ap = argparse.ArgumentParser(description='Ingest one meeting by URL or id')
    ap.add_argument('--url', dest='url', help='Meeting page URL (preferred)')
    ap.add_argument('--id', dest='mid', help='Meeting GUID (if URL not provided)')
    ap.add_argument(
        '--out',
        dest='out',
        default='data/raw/meetings',
        help='Output root (default: data/raw/meetings)',
    )
    ap.add_argument('--no-audio', action='store_true', help='Do not download audio MP3s')
    ap.add_argument('--headful', action='store_true', help='Show browser (not headless)')
    ap.add_argument(
        '--rps', type=float, default=DEFAULT_RPS, help='Max requests per second (default 1.5)'
    )
    args = ap.parse_args()

    meta = ingest_meeting(
        meeting_url=args.url,
        meeting_id=args.mid,
        out_root=args.out,
        with_audio=not args.no_audio,
        headless=not args.headful,
        rps=args.rps,
    )
    print(json.dumps(asdict(meta), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    _main()
