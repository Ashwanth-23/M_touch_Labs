from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse
import os

# Set of visited URLs to avoid loops
visited = set()

# Base URL
BASE_URL = "https://mtouchlabs.com/"

# Folder to save scraped data
os.makedirs("M_touch_lab", exist_ok=True)

def scrape_page(page, url):
    print(f"Scraping: {url}")
    page.goto(url, timeout=60000)
    try:
        page.click("text=Accept", timeout=3000)
    except:
        pass  # Cookie already accepted or not found
    page.wait_for_timeout(2000)
    text = page.locator("body").inner_text()
    filename = urlparse(url).path.replace("/", "_") or "home"
    with open(f"M_touch_lab/{filename}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return extract_links(page, url)

def extract_links(page, current_url):
    links = page.eval_on_selector_all("a", "elements => elements.map(e => e.href)")
    valid_links = []
    for link in links:
        if BASE_URL in link and link not in visited:
            visited.add(link)
            valid_links.append(link)
    return valid_links

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # Start crawling from the home page
    to_visit = [BASE_URL]
    visited.add(BASE_URL)

    while to_visit:
        current = to_visit.pop(0)
        try:
            new_links = scrape_page(page, current)
            to_visit.extend(new_links)
        except Exception as e:
            print(f"Failed to scrape {current}: {e}")

    browser.close()
    print("✅ Scraping completed. Check the 'aci_pages/' folder.")
