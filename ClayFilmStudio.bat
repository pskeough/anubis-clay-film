@echo off
rem Clay Film Studio: starts the local web UI on http://127.0.0.1:8761/ and opens the browser.
rem Add --dry to fake every GPU job. Close this window to stop the server.
cd /d "%~dp0"
if "%CLAY_FILM_PY%"=="" (set PY=python) else (set PY=%CLAY_FILM_PY%)
if not exist "%PY%" set PY=python
"%PY%" webui\server.py --open %*
if errorlevel 1 pause
