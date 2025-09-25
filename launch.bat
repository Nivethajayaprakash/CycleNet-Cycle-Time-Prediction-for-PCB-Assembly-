 # Windows Launch Script
# launch.bat - For Windows users

@echo off
echo 🔧 PCB Cycle Time Predictor - Windows Setup & Launch
echo ===================================================

REM Check if virtual environment exists
if not exist "pcb_env" (
    echo 📦 Creating virtual environment...
    python -m venv pcb_env
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call pcb_env\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Check if model exists
if not exist "pcb_model.pkl" (
    echo 🤖 Training machine learning model...
    python model_utils.py
) else (
    echo ✅ Model already trained!
)

REM Launch dashboard
echo 🚀 Launching dashboard...
streamlit run dashboard.py

pause