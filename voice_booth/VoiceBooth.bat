@echo off
rem Voice Booth: record your own narration against a finished clay_film piece.
rem Starts the local server (media venv: faster-whisper, numpy, ffmpeg) and opens the page.
rem Close this window to stop it. Takes and builds land in pieces\<piece>\voice\.
cd /d "%~dp0"
if "%CLAY_FILM_PODCAST_PY%"=="" (set PY=python) else (set PY=%CLAY_FILM_PODCAST_PY%)
powershell -NoProfile -Command "try { (New-Object Net.Sockets.TcpClient('127.0.0.1',7090)).Close(); exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
  echo Voice Booth is already running; opening it.
  start "" "http://127.0.0.1:7090"
  exit /b 0
)
start "" /b cmd /c "timeout /t 2 >nul & start "" http://127.0.0.1:7090"
"%PY%" -u server.py
pause
