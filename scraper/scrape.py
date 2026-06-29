import asyncio
import os
import json
import re
import sys
from playwright.async_api import async_playwright

# Reconfigure stdout for UTF-8 compatibility on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "https://genesis-cpo.netlify.app"
INVENTORY_URL = f"{BASE_URL}/en/inventory"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CARS_JSON_PATH = os.path.join(DATA_DIR, "cars.json")

async def parse_detail_page(page, url):
    print(f"Scraping details from: {url}")
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        
        # Extract the page title
        title = await page.title()
        # Clean title (e.g., "GV80 3.5T Royal | Genesis Certified" -> "GV80 3.5T Royal")
        name = title.split("|")[0].strip()
        
        # Get page inner text
        body_text = await page.evaluate("() => document.body.innerText")
        lines = [line.strip() for line in body_text.split("\n") if line.strip()]
        
        car_data = {
            "url": url,
            "name": name,
            "price_sar": None,
            "exterior_color": None,
            "interior_color": None,
            "mileage": None,
            "fuel_type": None,
            "transmission": None,
            "body_type": None,
            "engine": None,
            "features": []
        }
        
        # Parse overview fields using key-value line finding
        for i, line in enumerate(lines):
            line_upper = line.upper()
            if "EXTERIOR COLOR" in line_upper and i + 1 < len(lines):
                car_data["exterior_color"] = lines[i + 1]
            elif "INTERIOR COLOR" in line_upper and i + 1 < len(lines):
                car_data["interior_color"] = lines[i + 1]
            elif "MILEAGE" in line_upper and i + 1 < len(lines):
                car_data["mileage"] = lines[i + 1]
            elif "FUEL TYPE" in line_upper and i + 1 < len(lines):
                car_data["fuel_type"] = lines[i + 1]
            elif "TRANSMISSION" in line_upper and i + 1 < len(lines):
                car_data["transmission"] = lines[i + 1]
            elif "BODY TYPE" in line_upper and i + 1 < len(lines):
                car_data["body_type"] = lines[i + 1]
            elif "ENGINE" in line_upper and i + 1 < len(lines):
                car_data["engine"] = lines[i + 1]
            elif ("TOTAL PURCHASE PRICE" in line_upper or "TOTAL PRICE" in line_upper) and i + 1 < len(lines):
                price_str = lines[i + 1]
                # Extract number, e.g. "335,000" -> 335000
                price_match = re.search(r'[\d,]+', price_str)
                if price_match:
                    car_data["price_sar"] = int(price_match.group(0).replace(",", ""))
        
        # Extract features
        # Look for a line containing "X features" (e.g., "30 features")
        feature_count = None
        feature_start_idx = None
        for i, line in enumerate(lines):
            match = re.match(r'^(\d+)\s+features$', line, re.IGNORECASE)
            if match:
                feature_count = int(match.group(1))
                feature_start_idx = i + 1
                break
                
        if feature_count and feature_start_idx:
            # The next feature_count lines are the key features
            features = []
            for j in range(feature_start_idx, min(feature_start_idx + feature_count, len(lines))):
                features.append(lines[j])
            car_data["features"] = features
            
        return car_data
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

async def main():
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    async with async_playwright() as p:
        print("Launching browser...")
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print(f"Navigating to inventory page: {INVENTORY_URL}")
        await page.goto(INVENTORY_URL, wait_until="networkidle")
        
        car_links = set()
        page_num = 1
        
        while True:
            print(f"Scraping list page {page_num}...")
            
            # Find all links that go to a vehicle detail page
            # These links typically have href starting with "/en/inventory/" but excluding "/en/inventory" itself
            links = await page.eval_on_selector_all(
                "a", 
                "elements => elements.map(el => el.getAttribute('href')).filter(href => href && href.startsWith('/en/inventory/') && href !== '/en/inventory')"
            )
            
            for link in links:
                car_links.add(f"{BASE_URL}{link}")
                
            print(f"Total unique car links found so far: {len(car_links)}")
            
            # Check if there is a "Next page" button and if it is enabled
            next_button = await page.query_selector('button[aria-label="Next page"]')
            if not next_button:
                print("No 'Next page' button found. Stopping list scraping.")
                break
                
            is_disabled = await next_button.is_disabled()
            if is_disabled:
                print("Next page button is disabled. Reached the last page.")
                break
                
            # Click next button and wait for page to update
            print("Clicking 'Next page'...")
            # We can capture the text showing current pagination before click
            # e.g., "Showing 1 - 12 of 56" and wait for it to change
            pagination_text = ""
            pager_info = await page.query_selector("span:has-text('Showing')")
            if pager_info:
                pagination_text = await pager_info.inner_text()
                
            await next_button.click()
            
            if pagination_text:
                # Wait for pagination text to change to indicate page loaded
                try:
                    await page.wait_for_function(
                        f"text => !document.body.innerText.includes(text)",
                        arg=pagination_text,
                        timeout=5000
                    )
                except Exception:
                    # Fallback to simple timeout if wait_for_function fails
                    await page.wait_for_timeout(2000)
            else:
                await page.wait_for_timeout(2000)
                
            page_num += 1
            
        print(f"\nFound {len(car_links)} total car links. Proceeding to scrape details...")
        
        # Scrape detail pages
        all_cars = []
        for idx, url in enumerate(sorted(list(car_links))):
            print(f"[{idx+1}/{len(car_links)}] ", end="")
            car_data = await parse_detail_page(page, url)
            if car_data:
                all_cars.append(car_data)
                # Write intermediate results to avoid data loss
                with open(CARS_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(all_cars, f, indent=2, ensure_ascii=False)
            # Sleep briefly to be a polite scraper
            await asyncio.sleep(0.5)
            
        print(f"\nScraping complete. Successfully scraped {len(all_cars)} cars.")
        print(f"Data saved to {CARS_JSON_PATH}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
