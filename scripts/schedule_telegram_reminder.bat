@echo off
REM Schedule telegram_reminder.py to run every 3 hours
REM Run this with Admin privileges

setlocal enabledelayedexpansion

REM Get Python path
for /f "tokens=*" %%i in ('where python') do set PYTHON_PATH=%%i

if "!PYTHON_PATH!"=="" (
    echo Error: Python not found in PATH
    exit /b 1
)

REM Set script path
set SCRIPT_PATH=D:\pipeline_SVM\scripts\telegram_reminder.py

echo Registering task: ART2_TelegramReminder
echo Script: !SCRIPT_PATH!
echo Python: !PYTHON_PATH!
echo.

REM Delete existing task if present
taskkill /tn "ART2_TelegramReminder" /f >nul 2>&1
schtasks /delete /tn "ART2_TelegramReminder" /f >nul 2>&1

REM Create task: runs every 3 hours, starting now
REM /SC HOURLY /MO 3 = every 3 hours
REM /ST 09:00 = start at 9 AM
REM /RL HIGHEST = run with highest privileges (needed for network)
schtasks /create /tn "ART2_TelegramReminder" ^
  /tr "\"!PYTHON_PATH!\" \"!SCRIPT_PATH!\"" ^
  /sc HOURLY /mo 3 ^
  /st 09:00 ^
  /rl HIGHEST ^
  /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [✓] Task scheduled successfully
    echo [✓] Will run every 3 hours starting at 09:00
    echo.
    REM Show task details
    schtasks /query /tn "ART2_TelegramReminder" /v
) else (
    echo [✗] Failed to create task (ErrorLevel: %ERRORLEVEL%)
    echo Check that you have Admin privileges
    exit /b %ERRORLEVEL%
)

endlocal
