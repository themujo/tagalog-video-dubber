@echo off
title AI Video Dubber GUI
cd /d "C:\Users\AdminPC\Desktop\dub\kazakh-video-dubbing"

:: Inayos mula sa .venv patungong venv batay sa iyong folder setup
call venv\Scripts\activate

echo Nagsisimula na ang Python Web GUI...
echo Pakihintay ng ilang segundo bago magbukas ang browser...

:: Maghintay ng 3 segundo para makapag-load ang Python bago buksan ang browser
timeout /t 3 >nul
start http://127.0.0.1:7860

python web_gui.py

pause