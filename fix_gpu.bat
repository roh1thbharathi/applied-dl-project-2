@echo off
echo.
echo ============================================================
echo   Fix GPU: Install CUDA PyTorch for RTX 3060 Ti
echo ============================================================
echo.

echo Step 1: Python + pip info...
python --version
pip --version
echo.

echo Step 2: Check NVIDIA driver (trying common paths)...
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    nvidia-smi
) else (
    if exist "C:\Windows\System32\nvidia-smi.exe" (
        C:\Windows\System32\nvidia-smi.exe
    ) else if exist "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" (
        "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    ) else (
        echo nvidia-smi not found in PATH.
        echo Checking via Python...
        python -c "import subprocess; r=subprocess.run(['where','nvidia-smi'],capture_output=True,text=True); print(r.stdout or 'nvidia-smi not found anywhere')"
    )
)
echo.

echo Step 3: Installing correct CUDA torch via conda (most reliable)...
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
if errorlevel 1 (
    echo Conda install failed, trying pip with cu124 index...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo Trying cu121 with explicit latest stable version...
        pip install "torch==2.5.1+cu121" "torchaudio==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
    )
)
echo.

echo Step 4: Verifying GPU...
python -c "import torch; print('PyTorch :', torch.__version__); print('CUDA    :', torch.cuda.is_available()); print('GPU     :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"
echo.
pause
