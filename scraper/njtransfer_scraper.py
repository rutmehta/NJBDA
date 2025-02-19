import os
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Constants
BASE_URL = "https://njtransfer.org"
LOGIN_URL = f"{BASE_URL}/auth/validate/"
USERNAME = "njcolleges"
PASSWORD = "njtransfer@2020"
DOWNLOAD_DIR = "downloads"
DEFAULT_TIMEOUT = 60000  # 60 seconds

def setup_browser(playwright):
    """Launch Chrome with user's default profile"""
    user_data_dir = os.path.expanduser("~/Library/Application Support/Google/Chrome")
    browser = playwright.chromium.launch_persistent_context(
        user_data_dir,
        headless=False,
        accept_downloads=True,
        viewport={'width': 1280, 'height': 720}
    )
    page = browser.pages[0] if browser.pages else browser.new_page()
    page.set_default_timeout(DEFAULT_TIMEOUT)
    return browser, page

def wait_for_manual_login(page):
    """Wait for the user to manually log in and reach the resources page."""
    print("\nPlease follow these steps:")
    print("1. The browser will open to njtransfer.org/resources/")
    print("2. Please log in manually through the website")
    print("3. Once you're logged in and can see the resources page, the script will continue automatically\n")
    
    page.goto(f"{BASE_URL}/resources/", wait_until='networkidle')
    
    # Wait for an element that only appears when logged in
    print("Waiting for successful login...")
    page.wait_for_selector('.resources-container', timeout=300000)  # 5 minute timeout
    print("Login detected! Proceeding with download...")

def download_syllabi(page, browser):
    # Create downloads directory if it doesn't exist
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    try:
        # Make sure we're on the resources page
        if not "/resources/" in page.url:
            print("Navigating to resources page...")
            page.goto(f"{BASE_URL}/resources/", wait_until='networkidle')
        
        # Wait for the syllabi section to be visible
        print("Waiting for syllabi section...")
        page.wait_for_selector('h3:has-text("List of Course Syllabi by College")', 
                             state='visible',
                             timeout=DEFAULT_TIMEOUT)
        
        # Get all PDF and document links
        links = page.query_selector_all('a[href*=".pdf"], a[href*=".doc"]')
        
        print(f"Found {len(links)} files to download")
        
        # Download each file
        for link in links:
            try:
                file_url = link.get_attribute('href')
                if file_url:
                    if not file_url.startswith('http'):
                        file_url = urljoin(BASE_URL, file_url)
                    
                    file_name = os.path.basename(file_url)
                    print(f"Downloading: {file_name}")
                    
                    # Use requests to download the file
                    response = requests.get(file_url)
                    if response.status_code == 200:
                        file_path = os.path.join(DOWNLOAD_DIR, file_name)
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Successfully downloaded: {file_name}")
                    time.sleep(1)  # Brief pause between downloads
            except Exception as e:
                print(f"Error downloading file: {str(e)}")
                
    except PlaywrightTimeout as e:
        print(f"Timeout while finding syllabi section: {str(e)}")
        raise
    except Exception as e:
        print(f"Error finding syllabi section: {str(e)}")
        raise

def main():
    print("Using your existing Chrome profile - this should maintain your login session")
    print("Please make sure Chrome is completely closed before running this script")
    input("Press Enter when Chrome is closed...")
    
    with sync_playwright() as playwright:
        browser, page = setup_browser(playwright)
        try:
            print("Successfully launched Chrome with your profile!")
            wait_for_manual_login(page)
            download_syllabi(page, browser)  
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    main()
