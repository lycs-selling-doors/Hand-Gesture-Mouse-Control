@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m app.gui
) else (
  python -m app.gui
)
if errorlevel 1 pause
