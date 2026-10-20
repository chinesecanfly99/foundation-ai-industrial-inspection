$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Please install Python 3.11 from python.org and retry.' }
}
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $projectPython -m pip install -r requirements/base.txt
if ($LASTEXITCODE -ne 0) { throw 'Base dependency installation failed.' }
& $projectPython -m pip install -r requirements/torch-cpu-windows.txt
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $projectPython -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
& $projectPython -m industrial_inspection prepare-demo
if ($LASTEXITCODE -ne 0) { throw 'Demo preparation failed.' }
& $projectPython -m industrial_inspection check-env
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
Write-Host 'Ready. Run start_windows.cmd, then open http://127.0.0.1:7860'
