@echo off
echo ========================================
echo  Skin Cancer Model - Setup
echo ========================================
echo.
echo Step 1: Installing PyTorch with CUDA...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if %ERRORLEVEL% neq 0 (
    echo Failed to install PyTorch.
    pause
    exit /b 1
)
echo.
echo Step 2: Installing remaining dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)
echo.
echo Step 3: Check GPU...
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo.
echo ========================================
echo  Setup complete!
echo.
echo  Commands:
echo    Train model : python train_seg.py --batch_size 32 --target_size 512 --epochs 50
echo    Evaluate    : python inference.py --model_path skin_cancer_model.pth
echo    Evaluate EMA: python inference.py --model_path skin_cancer_model_ema.pth
echo ========================================
pause
