@echo off
REM E-Fatura AI Demo - Windows Batch Script
echo ============================================
echo    E-FATURA AI DEMO - WINDOWS
echo ============================================
echo.

REM Python kontrolu
python --version >nul 2>&1
if errorlevel 1 (
    echo HATA: Python bulunamadi!
    echo Lutfen Python 3.8+ yukleyin: https://python.org
    pause
    exit /b 1
)

echo Python bulundu: 
python --version

echo.
echo Gereksinimler kontrol ediliyor...

REM Pip kontrolu
pip --version >nul 2>&1
if errorlevel 1 (
    echo HATA: pip bulunamadi!
    pause
    exit /b 1
)

echo.
echo Demo baslatiliyor...
echo.

REM Demo'yu calistir
python demo.py

echo.
echo Demo tamamlandi!
pause