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

where python >nul 2>&1
if errorlevel 1 (
    echo   ERRO: Python nao encontrado no PATH
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo   Python: %PYVER%

set "PYTHONPATH=%cd%\.venv\lib\python3.12\site-packages;%cd%"
echo   PYTHONPATH configurado
echo.
echo   Starting server on http://127.0.0.1:8000
echo   Press Ctrl+C to stop
echo.

python -u src\server.py %*

echo.
echo   Servidor encerrou (codigo: %errorlevel%)
pause
