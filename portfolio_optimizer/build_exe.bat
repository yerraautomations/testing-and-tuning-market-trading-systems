@echo off
echo ========================================
echo  Portfolio Optimizer - Build EXE
echo ========================================
echo.

REM Check if PyInstaller is installed
pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
    echo.
)

echo Starting build process...
echo This may take 5-10 minutes...
echo.

cd /d "%~dp0"

REM Clean previous builds
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

REM Run PyInstaller
pyinstaller portfolio_optimizer.spec --clean

echo.
if exist "dist\PortfolioOptimizer.exe" (
    echo ========================================
    echo  BUILD SUCCESSFUL!
    echo ========================================
    echo.
    echo Your executable is at:
    echo   dist\PortfolioOptimizer.exe
    echo.
    echo File size:
    for %%A in ("dist\PortfolioOptimizer.exe") do echo   %%~zA bytes
    echo.
) else (
    echo ========================================
    echo  BUILD FAILED
    echo ========================================
    echo Check the error messages above.
)

pause
