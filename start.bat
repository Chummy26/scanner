@echo off
title Team OP Scanner
echo.
echo   Team OP Scanner
echo   ==============
echo.

cd /d "%~dp0backend"
if errorlevel 1 (
    echo   ERRO: Nao foi possivel acessar a pasta backend
    echo   Caminho: %~dp0backend
    pause
    exit /b 1
)

echo   Diretorio: %cd%
echo.

if not exist ".venv\Scripts\python.exe" (
    echo   ERRO: Virtual environment nao encontrado em .venv
    echo   Execute: python -m venv .venv
    echo   Depois:  .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('.venv\Scripts\python.exe --version 2^>^&1') do set PYVER=%%i
echo   Python: %PYVER% (venv)

set "PYTHONPATH=%cd%"
echo   PYTHONPATH configurado
echo.
echo   Starting server on http://127.0.0.1:8000
echo   Press Ctrl+C to stop
echo.

.venv\Scripts\python.exe -u src\server.py %*

echo.
echo   Servidor encerrou (codigo: %errorlevel%)
pause
