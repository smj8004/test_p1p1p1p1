@echo off
chcp 65001 > nul
echo.
echo ================================================================
echo   7-Day Long-Term Analysis System
echo ================================================================
echo.
echo   Start: %date% %time%
echo   Deadline: 2026-02-21 18:00
echo.
echo   Results: data\analysis_results\
echo   Logs: logs\
echo.
echo   Press Ctrl+C to stop (results will be saved)
echo.
echo ================================================================
echo.

cd /d "%~dp0"
python long_analysis.py

echo.
echo ================================================================
echo   Analysis finished at: %date% %time%
echo   Check results in: data\analysis_results\
echo ================================================================
pause
