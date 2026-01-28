"""
Nodriver Helper - Browser automation utilities for fetching JSON from URLs

USAGE BEST PRACTICES:

This module is designed to process URLs sequentially in a single browser instance,
which is more efficient and stable than opening multiple tabs concurrently.

RECOMMENDED APPROACH:
    1. Create a single list of all URLs you need to fetch
    2. Use NodriverBrowser context manager to create one browser instance
    3. Call fetch_json_from_urls once with all URLs and callbacks
    4. Handle results in callbacks (on_success, on_error) as they arrive
    5. Dynamic pagination: Add new URLs to the list inside callbacks if needed

Example:
    urls = ["https://example.com/api/page1", "https://example.com/api/page2"]

    def on_success(url, data, index):
        save_to_file(data)
        if len(data) == 100:
            urls.append(f"https://example.com/api/page{index + 2}")

    def on_error(url, error, content, index):
        print(f"Failed to fetch {url}: {error}")

    async with NodriverBrowser() as browser:
        await fetch_json_from_urls(browser, urls, on_success=on_success, on_error=on_error)

COMMAND LINE USAGE:
    python3 nodriver_helper.py https://example.com --screenshot --contents
    python3 nodriver_helper.py https://example.com --screenshot --json
    python3 nodriver_helper.py https://example.com --headless --screenshot
    python3 nodriver_helper.py https://example.com/api --json --wait 5
    python3 nodriver_helper.py example.com --screenshot --contents  # auto-adds https://
"""

import nodriver as uc
import asyncio
import json
import random
import os
import re
import logging
import argparse
import urllib.parse
import base64
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable, Any
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class FetchError(Exception):
    """Base exception for fetch errors"""
    pass


class JSONExtractionError(FetchError):
    """Raised when JSON cannot be extracted from content"""
    pass


class BrowserError(FetchError):
    """Raised when browser operations fail"""
    pass


class SelectorTimeoutError(FetchError):
    """Raised when selector wait times out"""
    pass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FetchConfig:
    """Configuration for fetch operations with sensible defaults"""
    wait_time: float = 3.0
    selector: str = 'body'
    selector_timeout: float = 10.0
    delay_range: Tuple[float, float] = (3.0, 15.0)
    debug_dir: Optional[str] = "debug_pages"
    debug_mode: bool = False
    headless: bool = False
    viewport_width: int = 1920
    viewport_height: int = 1080
    screenshot_format: str = "png"

    def __repr__(self) -> str:
        return (f"FetchConfig(wait_time={self.wait_time}s, "
                f"selector_timeout={self.selector_timeout}s, "
                f"debug_mode={self.debug_mode}, headless={self.headless})")


# ============================================================================
# Browser Context Manager
# ============================================================================

class NodriverBrowser:
    """
    Context manager for nodriver browser lifecycle management.
    Ensures proper cleanup even if errors occur.

    Usage:
        async with NodriverBrowser(config) as browser:
            results = await fetch_json_from_urls(browser, urls, config)
    """

    def __init__(self, config: Optional[FetchConfig] = None):
        self.config = config or FetchConfig()
        self.browser = None

    async def __aenter__(self):
        try:
            # Configure for stability - headful by default for debugging
            browser_args = [
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ]

            self.browser = await uc.start(
                no_sandbox=True,
                headless=self.config.headless,
                browser_args=browser_args
            )
            logger.info(f"Browser started (headless={self.config.headless})")
            
            # Give browser connection time to stabilize
            await asyncio.sleep(1)
            return self.browser
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            raise BrowserError(f"Failed to start browser: {e}")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            try:
                # Close all tabs first
                for tab in self.browser.tabs:
                    try:
                        await tab.close()
                    except Exception:
                        pass

                await asyncio.sleep(0.5)
                self.browser.stop()
                await asyncio.sleep(0.5)
                logger.debug("Browser stopped cleanly")
            except Exception as e:
                logger.warning(f"Error stopping browser: {e}")
        return False


# ============================================================================
# Utility Functions
# ============================================================================


    """
    Sanitize text for use in filenames by replacing problematic characters.

    Args:
        text: Text to sanitize (e.g., URL)

    Returns:
        Sanitized string safe for use in filenames
    """
    return text.replace(".", "_").replace("/", "_").replace(":", "_").replace("?", "_").replace("&", "_")


def url_encode_filename(url: str) -> str:
    """
    Create a percent-encoded filename from a URL.
    
    Args:
        url: The URL to encode
        
    Returns:
        Percent-encoded filename with extension
    """
    # Remove protocol
    if "://" in url:
        url = url.split("://", 1)[1]
    
    # Percent-encode the URL
    encoded = urllib.parse.quote(url, safe='')
    return f"{encoded}.html"


def ensure_url_format(url: str) -> str:
    """
    Ensure URL has a protocol. Adds https:// if missing.
    
    Args:
        url: URL string that may or may not have protocol
        
    Returns:
        URL with protocol
    """
    if "://" not in url:
        url = f"https://{url}"
    return url


def extract_json_from_content(content: str) -> dict:
    """
    Extract JSON from page content. Tries pure JSON first, then falls back
    to extracting JSON embedded in HTML content.

    Args:
        content: Raw page content (JSON or HTML with embedded JSON)

    Returns:
        Parsed JSON as dictionary or list

    Raises:
        JSONExtractionError: If JSON cannot be extracted or parsed
    """
    try:
        # First, assume content is pure JSON
        return json.loads(content)
    except json.JSONDecodeError:
        logger.debug("Content is not pure JSON, attempting extraction...")
        
        # Check if content is wrapped in HTML tags
        if content.strip().startswith('<html') or content.strip().startswith('<!DOCTYPE'):
            # Strip HTML tags from start and end using regex
            cleaned = re.sub(r'^(<[^>]+>)+', '', content, flags=re.MULTILINE | re.DOTALL)
            cleaned = re.sub(r'(</[^>]+>)+\s*$', '', cleaned, flags=re.MULTILINE | re.DOTALL)
            cleaned = cleaned.strip()

            try:
                logger.debug("Successfully extracted JSON from HTML-wrapped content")
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass  # Fall through to manual extraction

        # Final fallback: manually find first { or [ to last } or ]
        array_start = content.find('[')
        array_end = content.rfind(']') + 1

        obj_start = content.find('{')
        obj_end = content.rfind('}') + 1

        # Determine which comes first and use that
        if array_start != -1 and (obj_start == -1 or array_start < obj_start):
            if array_end > array_start:
                json_str = content[array_start:array_end]
                logger.debug("Extracted JSON array from content")
                return json.loads(json_str)
        elif obj_start != -1 and obj_end > obj_start:
            json_str = content[obj_start:obj_end]
            logger.debug("Extracted JSON object from content")
            return json.loads(json_str)

        raise JSONExtractionError("Could not extract JSON from page content.")


# ============================================================================
# Main Fetch Functions
# ============================================================================

async def fetch_json_from_urls(
    browser,
    urls: List[str],
    config: Optional[FetchConfig] = None,
    on_success: Optional[Callable[[str, dict, int], Any]] = None,
    on_error: Optional[Callable[[str, str, Optional[str], int], Any]] = None,
    progress_desc: str = "Fetching URLs",
) -> List[Dict]:
    """
    Fetch and parse JSON content from multiple URLs using a single browser instance.
    Each URL is opened in a new tab for isolation, then closed after processing.
    Results are saved immediately via callbacks as they are fetched.

    Args:
        browser: Active nodriver browser instance
        urls: List of URLs to fetch
        config: FetchConfig object with fetch settings (default: FetchConfig())
        on_success: Callback function(url, data, index) called when URL is successfully fetched
        on_error: Callback function(url, error, content, index) called when URL fetch fails
        progress_desc: Description for the progress bar

    Returns:
        List of result dictionaries, one per URL:
        - Success: {"url": str, "status": "success", "data": dict}
        - Error: {"url": str, "status": "error", "error": str, "content": str (optional)}

    Raises:
        BrowserError: If browser operations fail
    """
    config = config or FetchConfig()
    results = []
    logger.info(f"Starting fetch of {len(urls)} URLs with config: {config}")

    for i, url in tqdm(enumerate(urls), total=len(urls), desc=progress_desc, unit="url"):
        page = None
        content = None

        try:
            # Ensure URL has protocol
            url = ensure_url_format(url)
            
            # Open URL in new tab for isolation
            logger.debug(f"Opening URL {i+1}/{len(urls)}: {url}")
            page = await browser.get(url, new_tab=True)

            # Wait for page to load
            await page.sleep(config.wait_time)

            # Try to wait for selector (continue even if it times out)
            try:
                await page.select(config.selector, timeout=config.selector_timeout)
                logger.debug(f"Selector '{config.selector}' found")
            except Exception as e:
                logger.warning(f"Selector '{config.selector}' not found or timeout: {e}")

            # Get page content
            content = await page.get_content()

            # Parse JSON from content
            data = extract_json_from_content(content)

            # Save debug content in debug mode (even on success)
            if config.debug_mode and config.debug_dir:
                os.makedirs(config.debug_dir, exist_ok=True)
                safe_name = sanitize_filename(url)
                debug_path = os.path.join(config.debug_dir, f"{safe_name}_success.html")
                with open(debug_path, "w", encoding='utf-8') as f:
                    f.write(content)
                logger.debug(f"Saved debug content to {debug_path}")

            # Call success callback immediately if provided
            if on_success:
                on_success(url, data, i)

            results.append({
                "url": url,
                "status": "success",
                "data": data
            })
            logger.info(f"✓ Successfully fetched: {url}")

        except JSONExtractionError as e:
            logger.error(f"JSON extraction failed for {url}: {e}")
            error_msg = f"JSON extraction error: {e}"
            
            if config.debug_dir and content:
                os.makedirs(config.debug_dir, exist_ok=True)
                safe_name = sanitize_filename(url)
                debug_path = os.path.join(config.debug_dir, f"{safe_name}_json_error.html")
                with open(debug_path, "w", encoding='utf-8') as f:
                    f.write(content)
                logger.debug(f"Saved error debug content to {debug_path}")

            if on_error:
                on_error(url, error_msg, content, i)

            results.append({
                "url": url,
                "status": "error",
                "error": error_msg,
                "content": content if content else None
            })

        except Exception as e:
            logger.error(f"Fetch failed for {url}: {e}")
            
            # Save debug content if enabled and content was captured
            if config.debug_dir and content:
                os.makedirs(config.debug_dir, exist_ok=True)
                safe_name = sanitize_filename(url)
                debug_path = os.path.join(config.debug_dir, f"{safe_name}_error.html")
                with open(debug_path, "w", encoding='utf-8') as f:
                    f.write(content)
                logger.debug(f"Saved error debug content to {debug_path}")

            # Call error callback immediately if provided
            if on_error:
                on_error(url, str(e), content, i)

            results.append({
                "url": url,
                "status": "error",
                "error": str(e),
                "content": content if content else None
            })

        finally:
            # Close the tab
            if page:
                try:
                    await page.close()
                except Exception as e:
                    logger.warning(f"Error closing tab: {e}")

            # Apply random delay between requests (but not after the last one)
            if i < len(urls) - 1:
                delay = random.uniform(config.delay_range[0], config.delay_range[1])
                await asyncio.sleep(delay)

    logger.info(f"Fetch complete: {len([r for r in results if r['status'] == 'success'])}/{len(urls)} successful")
    return results


async def simple_fetch_json(url: str, config: Optional[FetchConfig] = None) -> dict:
    """
    Fetch and parse JSON from a single URL. Opens its own browser instance.
    
    Usage:
        data = await simple_fetch_json("https://api.example.com/data")
        
    Args:
        url: URL to fetch
        config: FetchConfig object with fetch settings
        
    Returns:
        Parsed JSON data
        
    Raises:
        FetchError: If fetch or JSON parsing fails
    """
    config = config or FetchConfig()
    logger.info(f"Simple fetch: {url}")
    
    async with NodriverBrowser(config) as browser:
        results = await fetch_json_from_urls(browser, [url], config)
        
        if results[0]["status"] == "error":
            raise FetchError(f"Failed to fetch {url}: {results[0]['error']}")
        
        return results[0]["data"]


async def fetch_json_batch(urls: List[str], config: Optional[FetchConfig] = None) -> List[Dict]:
    """
    Fetch JSON from multiple URLs. Handles browser lifecycle automatically.
    
    Usage:
        results = await fetch_json_batch([url1, url2, url3])
        for result in results:
            if result["status"] == "success":
                print(result["data"])
                
    Args:
        urls: List of URLs to fetch
        config: FetchConfig object with fetch settings
        
    Returns:
        List of result dictionaries
    """
    config = config or FetchConfig()
    logger.info(f"Batch fetch: {len(urls)} URLs")
    
    async with NodriverBrowser(config) as browser:
        return await fetch_json_from_urls(browser, urls, config)


async def fetch_page_content(
    url: str, 
    config: Optional[FetchConfig] = None,
    save_screenshot: bool = False,
    save_content: bool = False,
    output_dir: str = "."
) -> Dict[str, Any]:
    """
    Fetch page content with optional screenshots and content saving.
    
    Usage:
        result = await fetch_page_content(
            "https://example.com",
            save_screenshot=True,
            save_content=True
        )
        
    Args:
        url: URL to fetch
        config: FetchConfig object with fetch settings
        save_screenshot: Whether to save screenshot (1920x1080)
        save_content: Whether to save page content
        output_dir: Directory to save files
        
    Returns:
        Dictionary with keys: url, status, content, screenshot_path, content_path, error
    """
    config = config or FetchConfig()
    url = ensure_url_format(url)
    logger.info(f"Fetching page content: {url}")
    
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "url": url,
        "status": "error",
        "content": None,
        "screenshot_path": None,
        "content_path": None,
        "error": None
    }
    
    try:
        async with NodriverBrowser(config) as browser:
            page = await browser.get(url, new_tab=True)
            
            # Wait for page to load (set_viewport may not be available in all versions)
            await page.sleep(config.wait_time)
            
            # Set viewport to configured size using Emulation for precision
            try:
                await page.send(uc.cdp.emulation.set_device_metrics_override(
                    width=config.viewport_width,
                    height=config.viewport_height,
                    device_scale_factor=1,
                    mobile=False
                ))
            except Exception as e:
                logger.warning(f"Failed to set device metrics override: {e}")
                # Fallback to window size
                try:
                    await page.set_window_size(config.viewport_width, config.viewport_height)
                except Exception:
                    pass
            
            # Try to wait for selector
            try:
                await page.select(config.selector, timeout=config.selector_timeout)
            except Exception:
                pass
            
            # Save screenshot if requested
            if save_screenshot:
                # Force format to JPEG and extension to .jpg as per user request
                forced_format = "jpeg"
                screenshot_path = os.path.join(output_dir, f"{url_encode_filename(url).replace('.html', '')}_screenshot.{forced_format}")
                try:
                    # Try different screenshot methods depending on nodriver version
                    screenshot_data = None
                    
                    # Try method 1: page.screenshot() with file path
                    if hasattr(page, 'screenshot'):
                        try:
                            # Note: nodriver's screenshot() method might not directly support format argument
                            # We'll rely on the CDP command for format control if this fails to respect it
                            await page.screenshot(screenshot_path)
                            result["screenshot_path"] = screenshot_path
                            logger.info(f"Screenshot saved: {screenshot_path}")
                        except Exception:
                            pass
                    
                    # Try method 2: page.save_screenshot()
                    if not result["screenshot_path"] and hasattr(page, 'save_screenshot'):
                        try:
                            await page.save_screenshot(screenshot_path)
                            result["screenshot_path"] = screenshot_path
                            logger.info(f"Screenshot saved: {screenshot_path}")
                        except Exception:
                            pass
                    
                    # Try method 3: page.get_screenshot() returns bytes
                    if not result["screenshot_path"] and hasattr(page, 'get_screenshot'):
                        try:
                            # Again, nodriver's get_screenshot() might not take format argument directly
                            screenshot_data = await page.get_screenshot()
                            if screenshot_data:
                                with open(screenshot_path, 'wb') as f:
                                    f.write(screenshot_data)
                                result["screenshot_path"] = screenshot_path
                                logger.info(f"Screenshot saved: {screenshot_path}")
                        except Exception:
                            pass
                    
                    # Try method 4: Use CDP command directly for explicit format control
                    if not result["screenshot_path"]:
                        try:
                            # Use Chrome DevTools Protocol directly via nodriver.cdp
                            # Always request JPEG format from browser for screenshots
                            screenshot_data_str = await page.send(uc.cdp.page.capture_screenshot(format_=forced_format))
                            
                            if screenshot_data_str:
                                screenshot_bytes = base64.b64decode(screenshot_data_str)
                                with open(screenshot_path, 'wb') as f:
                                    f.write(screenshot_bytes)
                                result["screenshot_path"] = screenshot_path
                                logger.info(f"Screenshot saved: {screenshot_path}")
                        except Exception as e:
                            logger.debug(f"CDP screenshot failed: {e}")
                    
                    if not result["screenshot_path"]:
                        logger.warning(f"Screenshot not supported in this nodriver version or failed to save.")
                        
                except Exception as e:
                    logger.warning(f"Failed to save screenshot: {e}")
            
            # Get content
            content = await page.get_content()
            result["content"] = content
            result["status"] = "success"
            
            # Save content if requested
            if save_content:
                content_path = os.path.join(output_dir, url_encode_filename(url))
                with open(content_path, "w", encoding='utf-8') as f:
                    f.write(content)
                result["content_path"] = content_path
                logger.info(f"Content saved: {content_path}")
            
            await page.close()
            
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Failed to fetch page: {e}")
    
    return result


# ============================================================================
# Command Line Interface
# ============================================================================

async def main_async(args):
    """Async main for CLI"""
    config = FetchConfig(
        wait_time=args.wait,
        selector_timeout=args.selector_timeout,
        headless=args.headless,
        debug_mode=args.debug,
        viewport_width=args.width,
        viewport_height=args.height,
        screenshot_format=args.screenshot_format
    )
    
    url = ensure_url_format(args.url)
    logger.info(f"Processing: {url}")
    logger.info(f"Config: {config}")
    
    # Determine output directory
    output_dir = args.output or "."
    os.makedirs(output_dir, exist_ok=True)
    
    # Fetch page content
    result = await fetch_page_content(
        url,
        config=config,
        save_screenshot=args.screenshot,
        save_content=args.contents or args.json,
        output_dir=output_dir
    )
    
    # Parse JSON if requested
    if (args.json or (not args.contents and not args.screenshot)) and result["content"]:
        try:
            data = extract_json_from_content(result["content"])
            json_path = os.path.join(output_dir, f"{url_encode_filename(url).replace('.html', '')}_data.json")
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"JSON saved: {json_path}")
            result["json_path"] = json_path
        except JSONExtractionError as e:
            logger.warning(f"Could not extract JSON: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("FETCH COMPLETE")
    print("="*60)
    print(f"URL: {url}")
    print(f"Status: {result['status']}")
    
    if result["screenshot_path"]:
        print(f"Screenshot: {result['screenshot_path']}")
    if result["content_path"]:
        print(f"Content: {result['content_path']}")
    if result.get("json_path"):
        print(f"JSON: {result['json_path']}")
    if result["error"]:
        print(f"Error: {result['error']}")
    
    print("="*60)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Browser automation tool for fetching content from URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 nodriver_helper.py https://example.com --screenshot --contents
  python3 nodriver_helper.py example.com --screenshot --json
  python3 nodriver_helper.py https://api.example.com/data --json
  python3 nodriver_helper.py https://example.com --headless --screenshot --output results/
        """
    )
    
    parser.add_argument(
        "url",
        help="URL to fetch (https:// added automatically if missing)"
    )
    
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="Save screenshot at 1920x1080 resolution"
    )

    parser.add_argument(
        "--screenshot-format",
        default="png",
        choices=["png", "jpeg"],
        help="Format for screenshots: 'png' or 'jpeg' (default: png)"
    )
    
    parser.add_argument(
        "--contents",
        action="store_true",
        help="Save page HTML contents to percent-encoded filename"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Extract and save JSON from page content"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for saved files (default: current directory)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (no GUI)"
    )
    
    parser.add_argument(
        "--wait",
        type=float,
        default=3.0,
        help="Seconds to wait after page load (default: 3.0)"
    )
    
    parser.add_argument(
        "--selector-timeout",
        type=float,
        default=10.0,
        help="Timeout for CSS selector wait in seconds (default: 10.0)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Viewport width in pixels (default: 1920)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Viewport height in pixels (default: 1080)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (saves all page content)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run async main
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()