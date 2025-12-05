Write-Host "=== Windows Setup Started ==="

# Config
$VenvDir = "venv"
$PythonExe = "python"

# Parse arguments
param(
    [string]$venv = "venv",
    [string]$python = "python"
)

if ($venv) { $VenvDir = $venv }
if ($python) { $PythonExe = $python }

Write-Host "Venv: $VenvDir"
Write-Host "Python: $PythonExe"

# Create venv
if (!(Test-Path $VenvDir)) {
    Write-Host "[INFO] Creating virtual environment..."
    & $PythonExe -m venv $VenvDir
} else {
    Write-Host "[INFO] Virtual environment already exists."
}

# Install packages
Write-Host "[INFO] Upgrading pip..."
& "$VenvDir\Scripts\python.exe" -m pip install --upgrade pip wheel setuptools

Write-Host "[INFO] Installing requirements..."
& "$VenvDir\Scripts\pip.exe" install -r requirements.txt

# Create folders
New-Item -ItemType Directory -Path "outputs\models" -Force | Out-Null
New-Item -ItemType Directory -Path "outputs\logs" -Force | Out-Null
New-Item -ItemType Directory -Path "data" -Force | Out-Null

Write-Host "=== Windows Setup Completed ==="
Write-Host "Activate venv with:"
Write-Host "  .\$VenvDir\Scripts\Activate.ps1"
