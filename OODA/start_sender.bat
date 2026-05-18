@echo off
chcp 65001 >nul
title AFSIM File Sender
cd /d "%~dp0"

set IP=%1
set PORT=%2
if "%IP%"=="" set IP=192.168.1.100
if "%PORT%"=="" set PORT=9600

echo ==========================================
echo   AFSIM File Sender
echo   Target: %IP%:%PORT%
echo ==========================================
echo.

python sender.py %IP% %PORT%
pause
