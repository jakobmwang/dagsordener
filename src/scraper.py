"""Playwright-based scraper for dagsordener.aarhus.dk"""

import re
from playwright.sync_api import sync_playwright, Page

BASE_URL = "https://dagsordener.aarhus.dk"


def get_meeting_links(page: Page, url: str = BASE_URL) -> list[str]:
    """Get meeting links from a listing page (frontpage or year filter)."""
    page.goto(url)
    page.locator("a[href*='/vis?']").first.wait_for(timeout=10000)

    links = page.locator("a[href*='/vis?']").all()
    urls = []
    for link in links:
        href = link.get_attribute("href")
        if href:
            # Extract just the id, normalize URL
            match = re.search(r"id=([a-f0-9-]+)", href)
            if match:
                meeting_id = match.group(1)
                urls.append(f"{BASE_URL}/vis?id={meeting_id}")

    return list(dict.fromkeys(urls))  # dedupe, preserve order


def get_meeting_html(page: Page, url: str) -> str:
    """Fetch a meeting page and return the HTML after dynamic content loads."""
    page.goto(url)
    # Wait for the agenda items table to be populated
    page.locator("#dagsordenDetaljer .punktrow").first.wait_for(timeout=15000)
    return page.content()


def get_available_years(page: Page) -> list[int]:
    """Get list of available years from the frontpage dropdown."""
    page.goto(BASE_URL)
    page.locator("select option").first.wait_for(timeout=10000)

    options = page.locator("select option").all()
    years = []
    for opt in options:
        val = opt.get_attribute("value")
        if val and val.isdigit():
            years.append(int(val))

    return sorted(years, reverse=True)


def get_year_meeting_links(page: Page, year: int) -> list[str]:
    """Get all meeting links for a specific year (handles infinite scroll)."""
    url = f"{BASE_URL}/?year={year}"
    page.goto(url)
    page.locator("a[href*='/vis?']").first.wait_for(timeout=10000)

    # Click "Vis flere" until no more
    prev_count = 0
    stable_rounds = 0
    max_rounds = 100

    for _ in range(max_rounds):
        count = page.locator("a[href*='/vis?']").count()

        if count == prev_count:
            stable_rounds += 1
            if stable_rounds >= 3:
                break
        else:
            stable_rounds = 0
            prev_count = count

        # Try clicking "Vis flere" button
        btn = page.locator("button:has-text('Vis flere'), .vis-flere")
        if btn.count() > 0 and btn.first.is_visible():
            btn.first.click()
            page.wait_for_timeout(500)
        else:
            break

    return get_meeting_links(page, url)


def create_browser_context():
    """Create a Playwright browser context."""
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    return playwright, browser, page


def close_browser(playwright, browser):
    """Close browser and playwright."""
    browser.close()
    playwright.stop()
