@echo off
setlocal

echo ============================================================
echo  Setting up Python Virtual Environment for FYRP
echo  Adversarial Face Detection (ResNet-18)
echo ============================================================

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Download from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo [1/4] Creating virtual environment...
python -m venv venv

if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Upgrading core build tools...
python -m pip install --upgrade pip setuptools wheel --quiet
python -m pip cache purge

echo [4/4] Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo [Bonus] Installing torchattacks (bypassing conflicts)...
pip install "torchattacks>=3.5.1" --no-deps

if errorlevel 1 (
    echo [ERROR] Some packages failed to install. Check requirements.txt.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  To activate the environment next time:
echo      venv\Scripts\activate
echo.
echo  To run training (Phase 1 - clean data):
echo      python train.py --mode clean --data_dir data/clean
echo.
echo  To generate adversarial dataset:
echo      python train.py --mode generate --data_dir data/clean --checkpoint checkpoints/best_model.pth
echo.
echo  To run training (Phase 2 - adversarial fine-tuning):
echo      python train.py --mode adversarial --csv data/labels.csv
echo.
echo  To evaluate a single image:
echo      python evaluate.py --image face.jpg --checkpoint checkpoints/best_model.pth
echo ============================================================
pause
