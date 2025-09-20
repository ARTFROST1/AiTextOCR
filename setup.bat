@echo off
REM AiTextOCR Setup (Windows CMD)

SETLOCAL ENABLEDELAYEDEXPANSION

ECHO ============================================================
ECHO  AiTextOCR - Setup (Windows CMD)
ECHO ============================================================

REM Change to the script directory
cd /d %~dp0

REM Try python, then py
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    where py >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        ECHO Python not found. Please install Python 3.10+ from https://www.python.org/downloads/
        EXIT /B 1
    ) ELSE (
        set PYEXE=py
    )
) ELSE (
    set PYEXE=python
)

ECHO Running environment setup...
%PYEXE% scripts\setup_environment.py
IF %ERRORLEVEL% NEQ 0 (
    ECHO Setup failed.
    EXIT /B 1
)

ECHO.
ECHO Next steps:
ECHO 1) Activate venv:  .\venv_cuda\Scripts\activate
ECHO 2) Run GUI:        python run_gui.py
ECHO 3) Or run CLI:     python scripts\run_full_evaluation.py

ENDLOCAL
