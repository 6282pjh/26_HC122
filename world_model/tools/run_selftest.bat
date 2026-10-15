@echo off
chcp 65001 >nul
rem world_model 폴더 기준으로 실행 (이 파일은 tools\ 안에 있음)
cd /d "%~dp0.."
echo [world_model data tools] selftest
python -c "import cv2, numpy" 2>nul || (
  echo [install] tools\requirements.txt
  python -m pip install -r tools\requirements.txt
)
python tools\selftest.py
echo.
pause
