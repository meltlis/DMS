@echo off
chcp 65001 >nul
echo 正在激活 CUDA 虚拟环境...
call "%~dp0.venv-cuda\Scripts\activate.bat"
if errorlevel 1 (
    echo ❌ 激活失败
    exit /b 1
)
echo ✅ CUDA 虚拟环境已激活
echo.
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo.
echo 现在可以运行项目了，例如:
echo   python src/pipeline.py
cmd /k
