@echo off
chcp 65001 >nul
set "APP_NAME=文献综述与论文复现智能体"
title %APP_NAME%
cd /d "%~dp0\.."
echo Starting %APP_NAME%...
powershell -ExecutionPolicy Bypass -File "literature_showcase\run_web.ps1"
pause
