<#
Setup script for Windows PowerShell
- Creates venv in ./venv_cuda
- Installs PyTorch (CUDA if available, otherwise CPU)
- Installs pinned Python dependencies
#>

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " AiTextOCR - Setup (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Ensure we run from the script's directory
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Check python availability
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
}

if (-not $python) {
    Write-Host "✗ Python not found. Please install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Run setup script
try {
    Write-Host "Running environment setup..." -ForegroundColor Yellow
    & $python.Source scripts/setup_environment.py
}
catch {
    Write-Host "✗ Setup failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "\nNext steps:" -ForegroundColor Green
Write-Host "1) Activate venv: .\\venv_cuda\\Scripts\\activate" -ForegroundColor Green
Write-Host "2) Run GUI:       python run_gui.py" -ForegroundColor Green
Write-Host "3) Or run CLI:    python scripts\\run_full_evaluation.py" -ForegroundColor Green
