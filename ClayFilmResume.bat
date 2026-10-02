@echo off
rem Carries a piece on from where it stopped: stills, then motion, then assembly.
rem Each stage skips shots whose output already exists, so this is safe to run again.
rem   ClayFilmResume.bat anubis
rem   ClayFilmResume.bat anubis motion,assemble
rem Stop at any time with ClayFilmStop.bat.
cd /d "%~dp0"
if "%~1"=="" (echo usage: ClayFilmResume.bat NAME [stages] & pause & exit /b 1)
set STAGES=%~2
if "%STAGES%"=="" set STAGES=stills,motion,assemble
if "%CLAY_FILM_PY%"=="" (set PY=python) else (set PY=%CLAY_FILM_PY%)
if not exist "%PY%" set PY=python
set PYTHONIOENCODING=utf-8
"%PY%" -u -m clay_film run %1 --stages %STAGES%
pause
