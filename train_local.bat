@echo off
REM ============================================================
REM  Local Training Script  -  Codec-Robust Deepfake Detector
REM  RTX 3060 Ti  (CUDA 12.1 build of PyTorch)
REM ============================================================

SET ROOT=%~dp0
REM Strip trailing backslash from ROOT for clean path usage
SET ASV_ROOT=%ROOT:~0,-1%
SET OUT_DIR=%ROOT%results

echo.
echo ============================================================
echo   Codec-Robust Deepfake Detector -- Local Training
echo   ASV root : %ASV_ROOT%
echo   Output   : %OUT_DIR%
echo ============================================================
echo.

REM ── Step 1: Install CUDA-enabled PyTorch ──────────────────────
REM    This MUST come before the generic requirements.txt install.
REM    We target CUDA 12.1 (works on 3060 Ti with any recent driver).
REM    If you have CUDA 11.8 instead, swap cu121 -> cu118 below.
echo [1/4] Installing PyTorch with CUDA 12.1 support...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
if errorlevel 1 (
    echo WARN: CUDA 12.1 wheel failed, trying CUDA 11.8...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
)
echo   Done.

REM ── Step 2: Install remaining dependencies ──────────────────
echo.
echo [2/4] Installing other requirements...
pip install scikit-learn pandas matplotlib tensorboard tqdm soundfile librosa --quiet
if errorlevel 1 (echo ERROR: pip install failed & pause & exit /b 1)
echo   Done.

REM ── Step 3: Verify GPU is visible ──────────────────────────
echo.
echo [3/4] Checking GPU...
python -c "import torch; gpu=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if gpu else 'NOT FOUND'; print(f'  CUDA available : {gpu}'); print(f'  GPU            : {name}'); exit(0 if gpu else 1)"
if errorlevel 1 (
    echo.
    echo ============================================================
    echo   ERROR: CUDA GPU not detected by PyTorch.
    echo   Possible causes:
    echo     1. NVIDIA driver too old  -  update from nvidia.com
    echo     2. CUDA version mismatch  -  check: nvidia-smi
    echo        If it shows CUDA 11.x, edit this file: cu121 -> cu118
    echo     3. Virtual env conflict   -  try running as Administrator
    echo ============================================================
    pause
    exit /b 1
)

REM ── Step 4: Smoke test ─────────────────────────────────────
echo.
echo [4/4] Running smoke test...
cd /d "%ROOT%"
python src/smoke_test.py
if errorlevel 1 (echo ERROR: Smoke test failed & pause & exit /b 1)

REM ── Step 5: Training ───────────────────────────────────────
echo.
echo ============================================================
echo   Starting training -- 30 epochs on your 3060 Ti
echo   Expected time : ~3-5 hours (ASVspoof train set)
echo   Best model    : %OUT_DIR%\best_model.pt
echo   TensorBoard   : tensorboard --logdir "%OUT_DIR%\tb_logs"
echo   Press Ctrl+C to stop early (best_model.pt is saved live)
echo ============================================================
echo.

python src/train.py ^
    --asv_root      "%ASV_ROOT%" ^
    --epochs        30 ^
    --batch_size    32 ^
    --lr            3e-4 ^
    --alpha         0.5 ^
    --beta          0.1 ^
    --embed_dim     256 ^
    --num_workers   4 ^
    --output_dir    "%OUT_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: Training exited with an error. Check output above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   DONE!
echo   Results    : %OUT_DIR%
echo   Best model : %OUT_DIR%\best_model.pt
echo   TensorBoard: tensorboard --logdir "%OUT_DIR%\tb_logs"
echo ============================================================
pause
