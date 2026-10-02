@echo off
rem Frees both GPUs: ends every clay_film run/shot job (and the TTS, H3 and assembly children
rem they started), then stops the ComfyUI farm. Finished stills and clips are kept.
rem Resume later with ClayFilmResume.bat NAME.
cd /d "%~dp0"
if "%CLAY_FILM_PY%"=="" (set PY=python) else (set PY=%CLAY_FILM_PY%)
if not exist "%PY%" set PY=python
"%PY%" -m clay_film stop
pause
