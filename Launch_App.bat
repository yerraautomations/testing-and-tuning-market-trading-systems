@echo off
title Trading Workbench
echo ========================================
echo    Trading Workbench - Starting...
echo ========================================
echo.
echo Opening in your browser shortly...
echo (Keep this window open while using the app)
echo.
cd /d "%~dp0"
timeout /t 2 /nobreak >nul
start http://localhost:8501
python -m streamlit run "%~dp0app\Home.py" --server.headless=true --browser.gatherUsageStats=false
pause
