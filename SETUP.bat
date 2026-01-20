@echo off
REM ═════════════════════════════════════════════════════════════════════════════════
REM Breathing Sound Analysis for Asthma Detection - COMPLETE SETUP
REM Windows Batch Script
REM ═════════════════════════════════════════════════════════════════════════════════

color 3f
cls

echo ╔═════════════════════════════════════════════════════════════════════════════╗
echo ║                                                                             ║
echo ║         🫁 ASTHMA AI DETECTOR - COMPLETE SETUP (WINDOWS) 🫁              ║
echo ║                                                                             ║
echo ╚═════════════════════════════════════════════════════════════════════════════╝

echo.
echo 📋 System Check...
echo.

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found! Please install from https://nodejs.org
    pause
    exit /b 1
) else (
    echo ✅ Node.js installed: 
    node --version
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install from https://python.org
    pause
    exit /b 1
) else (
    echo ✅ Python installed:
    python --version
)

REM Check npm
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm not found!
    pause
    exit /b 1
) else (
    echo ✅ npm installed:
    npm --version
)

echo.
echo ═════════════════════════════════════════════════════════════════════════════════

REM Frontend Setup
echo.
echo 🔧 FRONTEND SETUP (React + Tailwind CSS)
echo.

if exist "node_modules" (
    echo ✅ Dependencies already installed
) else (
    echo 📦 Installing npm dependencies...
    call npm install
    if errorlevel 1 (
        echo ❌ npm install failed
        pause
        exit /b 1
    )
)

if not exist "tailwind.config.js" (
    echo 🎨 Installing Tailwind CSS...
    call npm install -D tailwindcss postcss autoprefixer
    call npx tailwindcss init -p
    if errorlevel 1 (
        echo ⚠️  Tailwind setup had issues, continuing...
    )
)

echo ✅ Frontend setup complete!

REM Backend Setup
echo.
echo 🔧 BACKEND SETUP (Flask + ML Libraries)
echo.

if exist "Backend\venv" (
    echo ✅ Virtual environment exists
    call Backend\venv\Scripts\activate.bat
) else (
    echo 🐍 Creating Python virtual environment...
    cd Backend
    python -m venv venv
    call venv\Scripts\activate.bat
    cd ..
)

echo 📦 Installing Python dependencies...
pip install -r Backend\requirements_full.txt
if errorlevel 1 (
    echo ⚠️  Some packages may not have installed. Check manually.
)

echo ✅ Backend setup complete!

echo.
echo ═════════════════════════════════════════════════════════════════════════════════
echo 🎉 SETUP COMPLETE!
echo ═════════════════════════════════════════════════════════════════════════════════

echo.
echo 📋 NEXT STEPS:
echo.
echo 1️⃣  START BACKEND SERVER (in terminal 1):
echo     cd Backend
echo     python app.py
echo.
echo 2️⃣  START REACT APP (in terminal 2):
echo     npm start
echo.
echo 3️⃣  OPEN BROWSER:
echo     http://localhost:3000
echo.
echo 4️⃣  API WILL RUN AT:
echo     http://localhost:5000
echo.
echo ═════════════════════════════════════════════════════════════════════════════════

echo.
echo ✨ For detailed documentation, see:
echo    - Frontend: DOCUMENTATION_INDEX.md
echo    - Backend: Backend/README_MODELS.md
echo.

pause
