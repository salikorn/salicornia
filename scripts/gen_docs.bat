@echo off
REM Generate salikorn API documentation with Doxygen.
REM Usage: scripts\gen_docs.bat
REM Output: build\docs\html  (warnings log: build\docs\doxygen_warnings.log)

setlocal
set ROOT=%~dp0..
set DOXYFILE=%ROOT%\Doxyfile

where doxygen >nul 2>nul
if errorlevel 1 (
    echo [gen_docs] doxygen not found on PATH.
    echo [gen_docs] Install it with:  winget install --id DimitriVanHeesch.doxygen -e
    exit /b 1
)

REM Doxygen writes OUTPUT_DIRECTORY relative to the current directory, so run from the repo root.
pushd "%ROOT%"
doxygen "%DOXYFILE%"
set RC=%ERRORLEVEL%
popd

if %RC% neq 0 (
    echo [gen_docs] doxygen failed with exit code %RC%.
    exit /b %RC%
)

echo [gen_docs] HTML docs written to build\docs\html
if exist "%ROOT%\build\docs\doxygen_warnings.log" (
    REM Count via PowerShell: cmd's `find` resolves to GNU find when run from Git Bash.
    powershell -NoProfile -Command "$n=(Get-Content -LiteralPath '%ROOT%\build\docs\doxygen_warnings.log' | Measure-Object -Line).Lines; Write-Host ('[gen_docs] ' + $n + ' warning lines in build\docs\doxygen_warnings.log')"
)
exit /b 0
