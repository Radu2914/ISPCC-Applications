@echo off
title Phosphorus v3 -- Nuclear Material Classifier
cd /d "%~dp0"
setlocal enabledelayedexpansion

:: ── find Python ───────────────────────────────────────────────────────────────
set "PY="
where python >nul 2>&1 && set "PY=python"
if "!PY!"=="" where py >nul 2>&1 && set "PY=py"
if "!PY!"=="" (
    echo.
    echo  ERROR: Python not found.
    echo  Install Python 3.7+ from python.org and add to PATH.
    echo.
    pause
    exit /b 1
)

:: ── check script present ──────────────────────────────────────────────────────
if not exist "%~dp0phosphorus_v3.py" (
    echo.
    echo  ERROR: phosphorus_v3.py not found in this folder.
    echo  Place phosphorus.bat in the same folder as phosphorus_v3.py.
    echo.
    pause
    exit /b 1
)

:: ── header ────────────────────────────────────────────────────────────────────
:header
cls
echo.
echo  ============================================================
echo   Phosphorus v3  --  Nuclear Material Classifier
echo   38-element nuclear subset  /  Pi x E search
echo  ============================================================
echo.
echo   --query Fe                   nearest 5 to iron
echo   --query U --k 8              nearest 8 to uranium
echo   --list                       all 38 nuclear elements
echo.
echo   --props mass=M z=Z E1=V1 E2=V2       property search
echo       requires exactly: mass + z + 2 E properties
echo.
echo   --props group=G period=P     periodic table grid search
echo.
echo   E properties (pick any 2):
echo   electronegativity  ie  radius  mp  bp  density  period  group
echo.
echo   help    show this again
echo   cls     clear screen
echo   exit    quit
echo.
echo  ============================================================
echo.

:: ── main loop ─────────────────────────────────────────────────────────────────
:loop
set "INPUT="
set /p "INPUT=  phosphorus> "
echo.

if "!INPUT!"==""      goto loop
if /i "!INPUT!"=="exit"   goto end
if /i "!INPUT!"=="quit"   goto end
if /i "!INPUT!"=="q"      goto end
if /i "!INPUT!"=="cls"    goto header
if /i "!INPUT!"=="clear"  goto header
if /i "!INPUT!"=="help"   goto header

!PY! phosphorus_v3.py !INPUT!
goto loop

:: ── exit ──────────────────────────────────────────────────────────────────────
:end
echo.
echo  Goodbye.
echo.
timeout /t 2 /nobreak >nul
exit /b 0
