@echo off
setlocal enabledelayedexpansion

echo.
echo  CSP — Cascade State Predictor
echo  Installing command...
echo.

:: Directory where this batch file lives (same as csp.py)
set "SCRIPT_DIR=%~dp0"

:: Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: Check csp.py is present
if not exist "%SCRIPT_DIR%\csp.py" (
    echo  [error] csp.py not found in %SCRIPT_DIR%
    echo  Run install.bat from the folder that contains csp.py
    pause
    exit /b 1
)

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo  [error] Python not found on PATH.
    echo  Install Python from python.org and ensure it is on your PATH.
    pause
    exit /b 1
)

:: Create csp.bat in the same directory
echo @echo off > "%SCRIPT_DIR%\csp.bat"
echo python "%SCRIPT_DIR%\csp.py" %%* >> "%SCRIPT_DIR%\csp.bat"

:: Add SCRIPT_DIR to current user PATH (no admin required)
:: Read current user PATH from registry
for /f "tokens=2*" %%A in (
    'reg query "HKCU\Environment" /v PATH 2^>nul'
) do set "CURRENT_PATH=%%B"

:: Check if already on PATH
echo !CURRENT_PATH! | find /i "%SCRIPT_DIR%" >nul 2>&1
if not errorlevel 1 (
    echo  [ok] %SCRIPT_DIR% is already on your PATH.
) else (
    :: Prepend script dir to PATH
    reg add "HKCU\Environment" /v PATH /t REG_EXPAND_SZ ^
        /d "%SCRIPT_DIR%;!CURRENT_PATH!" /f >nul
    echo  [ok] Added %SCRIPT_DIR% to your user PATH.
)

echo.
echo  ══════════════════════════════════════════════════════════
echo   CSP installed.
echo.
echo   Open a NEW Command Prompt window and type:
echo.
echo     csp              ^(interactive menu^)
echo     csp train        ^(train surrogate^)
echo     csp deploy       ^(live cascade detection^)
echo     csp probe        ^(structural typing probe^)
echo     csp status       ^(model info^)
echo     csp help         ^(command reference^)
echo.
echo   All four .py files must stay in:
echo   %SCRIPT_DIR%
echo  ══════════════════════════════════════════════════════════
echo.

pause
endlocal
