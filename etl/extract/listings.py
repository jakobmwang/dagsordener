from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from urllib.parse import urljoin

from playwright.sync_api import TimeoutError as PWTimeout

from etl.extract.ingestion import browser_context


@dataclass
class ListingEntry:
    url: str
    raw_date: str | None = None


def year_filter_url(frontpage_url: str, year: int) -> str:
    """Return the URL that filters meetings by the provided year."""
    # A relative query is appended to the provided frontpage URL.
    return urljoin(frontpage_url, f'?request.kriterie.udvalgId=&request.kriterie.moedeDato={year}')


def list_available_years(frontpage_url: str, *, headless: bool = True) -> list[int]:
    """Discover available meeting years from the frontpage dropdown."""
    with browser_context(headless=headless) as ctx:
        page = ctx.new_page()
        page.set_default_timeout(15000)
        page.goto(frontpage_url)

        selector = "select[name*='moedeDato'] option"
        values = page.eval_on_selector_all(
            selector,
            'options => options.map(o => ({value: o.value, text: o.textContent || ""}))',
        )

    years: set[int] = set()
    for opt in values:
        raw_value = (opt.get('value') or '').strip()
        raw_text = (opt.get('text') or '').strip()
        candidate = raw_value or raw_text
        m = re.search(r'(20\d{2})', candidate)
        if m:
            years.add(int(m.group(1)))

    if not years:
        years.add(dt.date.today().year)

    return sorted(years)


def collect_year_entries(
    frontpage_url: str,
    year: int,
    *,
    headless: bool = True,
) -> list[ListingEntry]:
    """Collect all meeting entries for a given year."""
    target_url = year_filter_url(frontpage_url, year)

    with browser_context(headless=headless) as ctx:
        page = ctx.new_page()
        page.set_default_timeout(30000)
        page.goto(target_url)

        try:
            page.wait_for_selector('#resultater a.searchresult, a.list-group-item.searchresult', timeout=10000)
        except PWTimeout:
            pass

        def anchors_locator():
            loc = page.locator('#resultater a.searchresult')
            return loc if loc.count() > 0 else page.locator('a.list-group-item.searchresult')

        def anchors_count() -> int:
            return anchors_locator().count()

        def load_more_once() -> bool:
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

        stable_rounds = 0
        prev_count = -1
        prev_height = page.evaluate('() => document.body.scrollHeight')
        max_loops = 400
        for _ in range(max_loops):
            before = anchors_count()

            clicked = False
            try:
                clicked = load_more_once()
            except PWTimeout:
                clicked = False

            if not clicked:
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
                try:
                    page.keyboard.press('End')
                except Exception:
                    pass

            try:
                page.wait_for_load_state('networkidle', timeout=5000)
            except PWTimeout:
                page.wait_for_timeout(1200)

            try:
                page.wait_for_function(
                    'prev => {'
                    " const q = document.querySelectorAll('#resultater a.searchresult, "
                    "a.list-group-item.searchresult');"
                    ' return q.length > prev;'
                    '}',
                    arg=before,
                    timeout=4000,
                )
            except PWTimeout:
                try:
                    page.wait_for_function(
                        'h => document.body.scrollHeight > h',
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

        entries: list[ListingEntry] = []
        loc = anchors_locator()
        for i in range(loc.count()):
            a = loc.nth(i)
            href = (a.get_attribute('href') or '').strip()
            if not href:
                continue
            absolute = urljoin(target_url, href)
            raw_date = _extract_date_from_anchor(a)
            entries.append(ListingEntry(url=absolute, raw_date=raw_date))

    return entries


DATE_PATTERNS: tuple[str, ...] = (
    r'\d{4}-\d{2}-\d{2}',
    r'\d{2}-\d{2}-\d{4}',
    r'\d{2}\.\d{2}\.\d{4}',
    r'\d{2}/\d{2}/\d{4}',
)


def _extract_date_from_anchor(anchor) -> str | None:
    selectors = [
        'time',
        '.dato',
        '.date',
        '.visningDato',
        '.meeting-date',
        'small',
    ]
    for sel in selectors:
        try:
            loc = anchor.locator(sel).first
            if loc and loc.count() > 0:
                text = (loc.inner_text() or '').strip()
                date_token = _extract_date_token(text)
                if date_token:
                    return date_token
        except Exception:
            continue

    try:
        text = (anchor.inner_text() or '').strip()
    except Exception:
        text = ''
    return _extract_date_token(text)


def _extract_date_token(text: str | None) -> str | None:
    if not text:
        return None
    for pattern in DATE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(0)
    return None
