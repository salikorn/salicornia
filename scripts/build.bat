@echo off
setlocal
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  echo [build] vcvars64.bat not found under VS 2022 Community & exit /b 1
)
call "%VCVARS%" >nul
cmake -S "%~dp0.." -B "%~dp0..\build" -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
cmake --build "%~dp0..\build" %* || exit /b 1
