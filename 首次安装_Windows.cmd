@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
if errorlevel 1 (
  echo Installation failed. Keep this window and check the error above.
  pause
  exit /b 1
)
echo Installation finished. Next run start_windows.cmd.
pause
