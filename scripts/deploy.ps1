$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Test-Path .venv)) {
    uv venv
}

.\.venv\Scripts\Activate.ps1
uv pip install -e .
python -m src.pipeline --source synthetic --max-frames 90
