@echo off
title Portfolio Optimizer
echo ========================================
echo    Portfolio Optimizer - Starting...
echo ========================================
echo.
echo Opening in your browser shortly...
echo (Keep this window open while using the app)
echo.
cd /d "%~dp0"
timeout /t 2 /nobreak >nul
start http://localhost:8501
streamlit run "%~dp0streamlit_app.py" --server.headless=true
pause
