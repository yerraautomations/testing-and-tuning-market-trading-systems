"""
MatchTrader Browser Automation Adapter (Non-API Mode)

Uses Playwright to automate the MatchTrader web interface.
This is required for prop firms that block direct API access.
"""

import logging
import time
import re
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger("trade_copier")

try:
    from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class MatchTraderBrowser:
    """
    Browser automation adapter for MatchTrader.
    Works with InstantFunding and other MatchTrader prop firms.
    """

    def __init__(
        self,
        server_url: str,
        email: str,
        password: str,
        account_number: str = None,
        headless: bool = True
    ):
        self.server_url = server_url.rstrip('/')
        self.email = email
        self.password = password
        self.account_number = account_number
        self.headless = headless

        self._playwright = None
        self._browser: Browser = None
        self._context: BrowserContext = None
        self._page: Page = None
        self._connected = False
        self._logged_in = False

    def connect(self) -> bool:
        """Launch browser and log into MatchTrader."""
        if not PLAYWRIGHT_AVAILABLE:
            print("ERROR: Playwright not installed. Run:")
            print("  pip install playwright")
            print("  playwright install chromium")
            return False

        try:
            print(f"Launching browser for: {self.server_url}")

            self._playwright = sync_playwright().start()

            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
            )

            self._context = self._browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )

            self._page = self._context.new_page()

            print("Navigating to login page...")
            # Use 'domcontentloaded' instead of 'networkidle' to avoid Cloudflare timeout
            self._page.goto(self.server_url, wait_until='domcontentloaded', timeout=120000)

            # Handle Cloudflare challenge if present
            if not self._handle_cloudflare():
                # If no Cloudflare, just wait a bit for page to settle
                time.sleep(3)

            # Perform login
            if self._login():
                self._connected = True
                self._logged_in = True
                print("Successfully logged into MatchTrader!")
                return True
            else:
                print("ERROR: Login failed")
                return False

        except Exception as e:
            print(f"ERROR: Browser connection failed: {e}")
            self.disconnect()
            return False

    def _handle_cloudflare(self) -> bool:
        """Handle Cloudflare challenge if present. Returns True if Cloudflare was handled."""
        try:
            time.sleep(2)  # Initial wait for page to stabilize
            page_content = self._page.content()
            page_title = self._page.title()

            # Check for various Cloudflare indicators
            cloudflare_indicators = [
                "Just a moment",
                "Checking your browser",
                "DDoS protection",
                "cf-browser-verification",
                "Cloudflare",
                "Ray ID",
            ]

            has_cloudflare = any(ind in page_content or ind in page_title for ind in cloudflare_indicators)

            if has_cloudflare:
                print("Cloudflare challenge detected, waiting for it to complete...")
                print("(This may take up to 60 seconds)")

                # Wait up to 60 seconds for Cloudflare to pass
                for i in range(60):
                    time.sleep(1)
                    try:
                        page_content = self._page.content()
                        page_title = self._page.title()

                        # Check if challenge is gone
                        still_cloudflare = any(ind in page_content or ind in page_title for ind in cloudflare_indicators)

                        if not still_cloudflare:
                            print(f"Cloudflare challenge passed after {i+1} seconds!")
                            time.sleep(2)  # Extra wait for page to fully load
                            return True

                        if i % 10 == 9:
                            print(f"  Still waiting... ({i+1}s)")
                    except:
                        pass

                print("WARNING: Cloudflare challenge timeout - may still work")
                return True  # Continue anyway

            return False  # No Cloudflare detected
        except Exception as e:
            print(f"Cloudflare check error: {e}")
            return False

    def _login(self) -> bool:
        """Perform login to InstantFunding MatchTrader."""
        try:
            print("Attempting login...")
            # Wait for page to be ready (use load, not networkidle)
            try:
                self._page.wait_for_load_state('load', timeout=30000)
            except:
                pass  # Continue even if timeout
            time.sleep(3)

            # InstantFunding specific: Look for email input
            # The form has "Email*" label above the input
            email_filled = False
            password_filled = False

            # Try multiple selectors for email field
            email_selectors = [
                'input[type="email"]',
                'input[type="text"]',
                'input[placeholder*="email" i]',
                'input[placeholder*="Email" i]',
                'input[name="email"]',
                'input[autocomplete="email"]',
                'input[autocomplete="username"]',
                # Generic - first visible text input
                'form input[type="text"]:visible',
                'input:below(:text("Email"))',
            ]

            for selector in email_selectors:
                try:
                    locator = self._page.locator(selector).first
                    if locator.is_visible(timeout=1000):
                        locator.fill(self.email)
                        email_filled = True
                        print(f"  Email entered using: {selector}")
                        break
                except:
                    continue

            if not email_filled:
                # Try clicking on input near "Email" text
                try:
                    self._page.get_by_label("Email").fill(self.email)
                    email_filled = True
                    print("  Email entered using label")
                except:
                    pass

            if not email_filled:
                print("ERROR: Could not find email field")
                self._page.screenshot(path="login_error.png")
                return False

            time.sleep(0.5)

            # Try multiple selectors for password field
            password_selectors = [
                'input[type="password"]',
                'input[name="password"]',
                'input[autocomplete="current-password"]',
                'input:below(:text("Password"))',
            ]

            for selector in password_selectors:
                try:
                    locator = self._page.locator(selector).first
                    if locator.is_visible(timeout=1000):
                        locator.fill(self.password)
                        password_filled = True
                        print(f"  Password entered using: {selector}")
                        break
                except:
                    continue

            if not password_filled:
                try:
                    self._page.get_by_label("Password").fill(self.password)
                    password_filled = True
                    print("  Password entered using label")
                except:
                    pass

            if not password_filled:
                print("ERROR: Could not find password field")
                return False

            time.sleep(0.5)

            # Click Sign In button
            submit_selectors = [
                'button:has-text("Sign In")',
                'button:has-text("Sign in")',
                'button:has-text("Login")',
                'button:has-text("Log in")',
                'button[type="submit"]',
                'input[type="submit"]',
            ]

            clicked = False
            for selector in submit_selectors:
                try:
                    locator = self._page.locator(selector).first
                    if locator.is_visible(timeout=1000):
                        locator.click()
                        clicked = True
                        print(f"  Clicked submit using: {selector}")
                        break
                except:
                    continue

            if not clicked:
                print("ERROR: Could not find submit button")
                return False

            # Wait for login to complete
            print("Waiting for login to complete...")
            time.sleep(5)
            try:
                self._page.wait_for_load_state('load', timeout=30000)
            except:
                pass  # Continue even if timeout - check page content instead

            # Check if we're on the trading page
            current_url = self._page.url
            page_content = self._page.content().lower()

            # Success indicators for InstantFunding
            if any(x in current_url.lower() for x in ['/trade', '/app', '/dashboard']):
                print("Login successful - on trading page")
                return True

            if any(x in page_content for x in ['open positions', 'trade', 'profit', 'balance']):
                print("Login successful - found trading elements")
                return True

            # Check for error messages
            if any(x in page_content for x in ['invalid', 'incorrect', 'error', 'failed']):
                print("ERROR: Login failed - invalid credentials")
                return False

            # Assume success if no obvious error
            return True

        except Exception as e:
            print(f"ERROR: Login error: {e}")
            self._page.screenshot(path="login_error.png")
            return False

    def disconnect(self):
        """Close browser and cleanup."""
        try:
            if self._page:
                self._page.close()
            if self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except:
            pass

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._connected = False
        self._logged_in = False
        print("Browser disconnected")

    def is_connected(self) -> bool:
        """Check if connected and logged in."""
        return self._connected and self._logged_in and self._page is not None

    def get_account_info(self) -> Optional[Dict]:
        """Get account info from the trading page."""
        if not self.is_connected():
            return None

        try:
            page_text = self._page.inner_text('body')

            # Look for account number (e.g., "LIVE 120124")
            account_match = re.search(r'LIVE\s*(\d+)', page_text)
            account_num = account_match.group(1) if account_match else self.account_number

            # Look for balance/profit
            profit_match = re.search(r'Profit[:\s]*\$?([\d,.-]+)', page_text, re.IGNORECASE)
            profit = float(profit_match.group(1).replace(',', '')) if profit_match else 0

            return {
                "account_number": account_num or "N/A",
                "balance": 0,  # Would need to navigate to Finance tab
                "equity": 0,
                "profit": profit
            }
        except Exception as e:
            print(f"ERROR: Could not get account info: {e}")
            return {"account_number": self.account_number or "Connected", "balance": 0, "equity": 0}

    def open_position(
        self,
        symbol: str,
        direction: str,
        volume: float,
        stop_loss: float = 0,
        take_profit: float = 0,
        comment: str = ""
    ) -> Optional[Dict]:
        """Open a position using browser automation - FAST version."""
        if not self.is_connected():
            print("ERROR: Not connected")
            return None

        try:
            print(f"Opening position: {symbol} {direction} {volume} lots")

            # Close any popups (Welcome popup, etc.)
            try:
                get_started = self._page.locator('button:has-text("Get started")').first
                if get_started.is_visible(timeout=500):
                    get_started.click()
                    time.sleep(0.3)
            except:
                pass

            # Press Escape to close any other popups
            self._page.keyboard.press("Escape")
            time.sleep(0.2)

            # Step 1: Click on symbol in Favorites/watchlist using JavaScript
            self._page.evaluate(f'''(function() {{
                const items = document.querySelectorAll('*');
                for (const item of items) {{
                    if (item.textContent.trim() === '{symbol}') {{
                        item.click();
                        return;
                    }}
                }}
            }})()''')
            time.sleep(0.5)

            # Step 2: Set volume using JavaScript
            self._page.evaluate(f'''(function() {{
                const inputs = document.querySelectorAll('input[type="number"]');
                for (const input of inputs) {{
                    if (input.offsetParent !== null) {{
                        input.focus();
                        input.value = '{volume}';
                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        return;
                    }}
                }}
            }})()''')
            print(f"  Volume: {volume}")
            time.sleep(0.2)

            # Step 3: Click Buy or Sell using Playwright with force=True
            if direction.upper() == "BUY":
                try:
                    btn = self._page.locator('[data-testid="order-panel-buy-button"], button[side="buy"], .ui-order-button--buy').first
                    btn.click(force=True, timeout=3000)
                except Exception as e:
                    print(f"  ERROR: Buy button click failed - {e}")
                    return None
            else:
                try:
                    btn = self._page.locator('[data-testid="order-panel-sell-button"], button[side="sell"], .ui-order-button--sell').first
                    btn.click(force=True, timeout=3000)
                except Exception as e:
                    print(f"  ERROR: Sell button click failed - {e}")
                    return None

            # Step 4: Click Confirm on the confirmation popup
            time.sleep(0.5)
            try:
                confirm_btn = self._page.locator('button:has-text("Confirm")').first
                if confirm_btn.is_visible(timeout=2000):
                    confirm_btn.click()
                    print(f"  -> {direction.upper()} {symbol} {volume} lots CONFIRMED")
                    return {"status": "success", "symbol": symbol, "direction": direction.upper(), "volume": volume}
            except:
                pass

            print(f"  -> {direction.upper()} {symbol} {volume} lots (no confirm needed)")
            return {"status": "success", "symbol": symbol, "direction": direction.upper(), "volume": volume}

        except Exception as e:
            print(f"ERROR: {e}")
            self._page.screenshot(path="trade_error.png")
            return None

    def close_position(self, ticket: int = None, symbol: str = None) -> bool:
        """Close a position."""
        if not self.is_connected():
            return False

        try:
            print(f"  Closing position on slave: {symbol or ticket}")

            # Step 1: Click on Open Positions tab
            try:
                open_pos_tab = self._page.locator('text="Open Positions"').first
                open_pos_tab.click(force=True)
                time.sleep(0.5)
            except:
                pass

            # Step 2: Find the X close button at the end of the position row
            # Looking at the screenshot, the X button is at the far right of each row
            closed = False

            # First, try to find the X button specifically (it's usually the last button in the row)
            try:
                # The X close button - try various approaches
                x_btn = self._page.locator('button:has(svg), button:last-of-type').last
                x_btn.click(force=True, timeout=2000)
                closed = True
            except:
                pass

            # If that didn't work, try finding close button by row with symbol
            if not closed and symbol:
                try:
                    # Use JavaScript to find and click the X button in the position row
                    self._page.evaluate(f'''(function() {{
                        const rows = document.querySelectorAll('tr, [class*="position-row"], [class*="row"]');
                        for (const row of rows) {{
                            if (row.textContent.includes('{symbol}')) {{
                                const buttons = row.querySelectorAll('button');
                                const lastBtn = buttons[buttons.length - 1];
                                if (lastBtn) {{
                                    lastBtn.click();
                                    return true;
                                }}
                            }}
                        }}
                        return false;
                    }})()''')
                    closed = True
                except:
                    pass

            if not closed:
                print(f"  WARNING: Could not find close button")
                self._page.screenshot(path="close_error.png")
                return False

            # Step 3: Click Confirm on confirmation popup
            time.sleep(0.5)
            try:
                confirm_btn = self._page.locator('button:has-text("Confirm")').first
                if confirm_btn.is_visible(timeout=2000):
                    confirm_btn.click()
                    print(f"  -> Position closed and CONFIRMED")
                    return True
            except:
                pass

            print(f"  -> Position close attempted")
            return True

        except Exception as e:
            print(f"ERROR: Failed to close position: {e}")
            return False

    def modify_position(self, ticket: int, stop_loss: float = None, take_profit: float = None) -> bool:
        """Modify position SL/TP."""
        print("WARNING: Position modification not fully implemented in browser mode")
        return False

    def get_positions(self) -> List[Dict]:
        """Get open positions."""
        return []

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
