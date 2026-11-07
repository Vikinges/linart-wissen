@echo off
title LinArt Wissen
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
rem Uses .\venv by default; set LINART_PYTHON to use another interpreter.
set "PY=%~dp0venv\Scripts\python.exe"
if defined LINART_PYTHON set "PY=%LINART_PYTHON%"
echo  LinArt Wissen - http://127.0.0.1:7890
"%PY%" "%~dp0server.py"
pause
