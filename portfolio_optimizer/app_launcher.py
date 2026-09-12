"""
Standalone launcher for Portfolio Optimizer.
This script launches the Streamlit app in a way that can be packaged with PyInstaller.
"""

import sys
import os
import webbrowser
import threading
import time
from pathlib import Path

def get_app_path():
    """Get the path to the streamlit app, works for both dev and frozen (exe) modes."""
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        app_dir = Path(sys._MEIPASS)
    else:
        # Running as script
        app_dir = Path(__file__).parent
    return app_dir

def open_browser():
    """Open browser after a short delay to let server start."""
    time.sleep(3)
    webbrowser.open('http://localhost:8501')

def main():
    """Launch the Streamlit application."""
    import streamlit.web.cli as stcli

    app_path = get_app_path()
    app_file = app_path / "streamlit_app.py"

    # Set up the arguments for streamlit
    # Disable development mode for packaged exe
    sys.argv = [
        "streamlit",
        "run",
        str(app_file),
        "--global.developmentMode=false",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.port=8501",
        "--theme.base=light"
    ]

    print("=" * 50)
    print("  Portfolio Optimizer")
    print("=" * 50)
    print()
    print("Starting application...")
    print("Opening in your default browser shortly...")
    print()
    print("Keep this window open while using the app.")
    print("Close this window to stop the application.")
    print()

    # Start browser opener in background thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Run streamlit
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
