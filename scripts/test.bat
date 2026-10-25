@echo off
setlocal
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VCVARS%" call "%VCVARS%" >nul
if "%~1"=="" (
  ctest --test-dir "%~dp0..\build" -L unit --output-on-failure
) else (
  ctest --test-dir "%~dp0..\build" -L %~1 --output-on-failure
)
