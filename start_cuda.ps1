# 启动 CUDA 环境并验证
# 如果执行策略阻止运行，请使用: powershell -ExecutionPolicy Bypass -File start_cuda.ps1

$venvPath = Join-Path $PSScriptRoot ".venv-cuda\Scripts\Activate.ps1"

if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "✅ CUDA 虚拟环境已激活" -ForegroundColor Green
    
    # 验证 CUDA
    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
    
    Write-Host ""
    Write-Host "现在可以运行项目了，例如:" -ForegroundColor Cyan
    Write-Host "  python src/pipeline.py" -ForegroundColor Yellow
} else {
    Write-Host "❌ 找不到虚拟环境: $venvPath" -ForegroundColor Red
}
