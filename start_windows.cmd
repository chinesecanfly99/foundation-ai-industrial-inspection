@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup_windows.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m app.app --host 127.0.0.1 --port 7860
pause
