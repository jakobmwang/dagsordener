"""Playwright-based scraper for dagsordener.aarhus.dk"""

import re
from playwright.sync_api import sync_playwright, Page

BASE_URL = "https://dagsordener.aarhus.dk"


def scrape_links_from_current_page(page: Page) -> list[str]:
    """Scrapes all meeting links from the currently loaded page."""
    page.locator("a[href*='/vis?']").first.wait_for(timeout=10000)

    links = page.locator("a[href*='/vis?']").all()
    urls = []
    for link in links:
        href = link.get_attribute("href")
        if href:
            match = re.search(r"id=([a-f0-9-]+)", href)
            if match:
                meeting_id = match.group(1)
                urls.append(f"{BASE_URL}/vis?id={meeting_id}")
    return list(dict.fromkeys(urls))


def scroll_to_end(page: Page, link_selector: str) -> None:
    """Scroll until no new content appears (few stable rounds in a row)."""
    links = page.locator(link_selector)
    last_count = links.count()
    last_href = links.last.get_attribute("href") if last_count else None
    stable_rounds = 0
    max_rounds = 120  # Safety net to avoid true infinite loops

    for _ in range(max_rounds):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1000)

        count = links.count()
        href = links.last.get_attribute("href") if count else None

        if count != last_count or href != last_href:
            last_count = count
            last_href = href
            stable_rounds = 0
        else:
            stable_rounds += 1
            if stable_rounds >= 5:
                break


def get_meeting_links(page: Page, url: str = BASE_URL) -> list[str]:
    """Get meeting links from a listing page (frontpage or year filter)."""
    page.goto(url)
    return scrape_links_from_current_page(page)


def get_meeting_html(page: Page, url: str) -> str:
    """Fetch a meeting page and return the HTML after dynamic content loads."""
    page.goto(url)
    # Wait for the agenda items table to be populated
    page.locator("#dagsordenDetaljer .punktrow").first.wait_for(timeout=30000)
    return page.content()


def get_available_years(page: Page) -> list[int]:
    """Get list of available years from the frontpage dropdown."""
    page.goto(BASE_URL)
    # The year dropdown is custom-built, not a <select> element.
    # We wait for the container, and then we have to click it to reveal the options.
    year_dropdown_trigger = page.locator("#year-navigation .title")
    year_dropdown_trigger.wait_for(timeout=10000)
    year_dropdown_trigger.click()

    # Now wait for the options to be visible and extract them
    year_option_selector = "#year-dropdown-ul a[data-value]"
    page.locator(year_option_selector).first.wait_for(timeout=5000)

    options = page.locator(year_option_selector).all()
    years = []
    for opt in options:
        val = opt.get_attribute("data-value")
        if val and val.isdigit():
            years.append(int(val))

    # Close the dropdown by clicking the trigger again to not obstruct other actions
    year_dropdown_trigger.click()

    return sorted(list(set(years)), reverse=True)


def get_year_meeting_links(page: Page, year: int) -> list[str]:
    """Get all meeting links for a specific year (handles infinite scroll)."""
    # Use the URL pattern provided by the user.
    url = f"{BASE_URL}/?request.kriterie.udvalgId=&request.kriterie.moedeDato={year}"
    page.goto(url)
    page.locator("a[href*='/vis?']").first.wait_for(timeout=10000)

    scroll_to_end(page, "a[href*='/vis?']")

    # After scrolling, scrape all the links from the fully-loaded page
    return scrape_links_from_current_page(page)


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