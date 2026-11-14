@echo off
setlocal EnableDelayedExpansion

title Movie Review Intelligence Studio
cd /d "%~dp0"

echo =====================================================================
echo           MOVIE REVIEW SENTIMENT ANALYZER AI - LAUNCHER
echo              Project 04 -- Stanford IMDB Benchmark
echo =====================================================================
echo.

:: 1. Check Python Availability
set "PYTHON_EXE="

:: Check Python in PATH
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=python"
    goto :PYTHON_FOUND
)

:: Check py launcher
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=py"
    goto :PYTHON_FOUND
)

echo [ERROR] Python is not installed or not in system PATH.
echo Please install Python 3.11 or 3.12 from https://www.python.org/
pause
exit /b 1

:PYTHON_FOUND
for /f "tokens=*" %%v in ('%PYTHON_EXE% --version 2^>^&1') do set "PY_VER=%%v"
echo [INFO] Detected: %PY_VER%

:: 2. Create or verify virtual environment (.venv)
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Creating local virtual environment in .venv...
    %PYTHON_EXE% -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [INFO] Existing virtual environment found.
)

:: 3. Activate virtual environment
call .venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: 4. Verify / Install Dependencies
python -c "import nicegui, sklearn, joblib, numpy" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing required dependencies from requirements.txt...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed successfully.
) else (
    echo [OK] Dependencies verified.
)

:: 5. Check whether a valid trained model already exists
if exist "models\sentiment_pipeline.joblib" (
    echo [OK] Trained model artifact found: models\sentiment_pipeline.joblib
    goto :LAUNCH_APP
)

echo.
echo [NOTICE] Trained model artifact not found. Initial setup required.

:: 6. Check for official dataset or download
if not exist "data\raw\aclImdb\train\pos" (
    echo [INFO] Stanford IMDB dataset not detected. Downloading official benchmark...
    python download_data.py
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Dataset download or extraction failed.
        pause
        exit /b 1
    )
) else (
    echo [OK] Stanford IMDB dataset files verified.
)

:: 7. Train the model
echo.
echo [INFO] Training sentiment pipeline on official dataset...
echo [INFO] (This trains TF-IDF + Logistic Regression on 25,000 reviews and evaluates on 25,000 test reviews)
python train.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Model training failed.
    pause
    exit /b 1
)
echo [OK] Model training and evaluation completed successfully!

:: 8. Launch the NiceGUI Application
:LAUNCH_APP
echo.
echo =====================================================================
echo   Launching Movie Review Intelligence Studio...
echo   Open your browser at: http://127.0.0.1:8080
echo   Press Ctrl+C in this terminal window to stop the studio.
echo =====================================================================
echo.

python app.py --host 127.0.0.1 --port 8080

if %ERRORLEVEL% neq 0 (
    echo [NOTICE] Application terminated with exit code %ERRORLEVEL%.
)

pause

