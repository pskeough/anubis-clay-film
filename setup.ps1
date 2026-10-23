# Scriptorium setup: one command for everything that does not need a model download.
#   powershell -ExecutionPolicy Bypass -File setup.ps1 [-NoTorch] [-CudaIndex cu128]
# Creates .venv, installs torch (CUDA build) and requirements.txt into it, runs npm install in filmprep/,
# copies .env.example to .env (if there is no .env yet) pointing CLAY_FILM_PY at the venv, then runs the
# CPU self-test. ComfyUI and the model weights are separate: see AGENT_SETUP.md.
param(
  [switch]$NoTorch,
  [string]$CudaIndex = "cu128"   # RTX 50-series needs cu128 or later
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }

Step "Python"
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { throw "Python 3.12 or later is not on PATH. Install it from python.org, then rerun." }
$ver = & $py -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$ver -lt [version]"3.12") { throw "Python $ver found; 3.12 or later is needed." }
Write-Host "python $ver at $py"

if (-not (Test-Path .venv\Scripts\python.exe)) { & $py -m venv .venv }
$vpy = (Resolve-Path .venv\Scripts\python.exe).Path
& $vpy -m pip install --upgrade pip --quiet

if (-not $NoTorch) {
  Step "torch ($CudaIndex)"
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    & $vpy -m pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$CudaIndex"
  } else {
    Write-Host "nvidia-smi not found: skipping torch. Narration and the voice booth need it." -ForegroundColor Yellow
  }
}

Step "requirements.txt"
& $vpy -m pip install -r requirements.txt

Step "filmprep (Node, for clay staging)"
if (Get-Command npm -ErrorAction SilentlyContinue) {
  Push-Location filmprep; npm install --no-audit --no-fund; Pop-Location
} else {
  Write-Host "npm not found: clay staging will not run. Install Node 20 or later, then run npm install in filmprep." -ForegroundColor Yellow
}

Step ".env"
if (Test-Path .env) {
  Write-Host ".env already exists; left as it is."
} else {
  (Get-Content .env.example) -replace '^CLAY_FILM_PY=.*', "CLAY_FILM_PY=$vpy" `
                             -replace '^CLAY_FILM_PODCAST_PY=.*', "CLAY_FILM_PODCAST_PY=$vpy" |
    Set-Content .env -Encoding utf8
  Write-Host "wrote .env (CLAY_FILM_PY and CLAY_FILM_PODCAST_PY point at .venv). Set COMFY_ROOT if ComfyUI is not at C:\ComfyUI."
}

Step "self-test (CPU)"
& $vpy -m clay_film selftest
if ($LASTEXITCODE -ne 0) { throw "selftest failed; see the output above." }

Write-Host "`nDone. Next: ComfyUI and the model weights (AGENT_SETUP.md), then:" -ForegroundColor Green
Write-Host "  .venv\Scripts\python farm\preflight.py"
Write-Host "  .venv\Scripts\python -m clay_film new my_film examples\example_plan.json"
