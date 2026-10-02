# farm.ps1 - run two ComfyUI workers, one pinned per GPU.
#
#   .\farm.ps1 start          # preflight, then bring both up and wait for health
#   .\farm.ps1 start -Skip    # skip preflight (you already know what you broke)
#   .\farm.ps1 status
#   .\farm.ps1 stop
#   .\farm.ps1 restart
#   .\farm.ps1 logs gpu0      # tail one worker
#   .\farm.ps1 start -Big     # ONE worker seeing BOTH cards, for the 21 GB
#                             # int8 H3 split with DisTorch2. Replaces the pair.
#   .\farm.ps1 stop           # stops the pair AND the big worker, always
#
# Windows PowerShell 5.1: no && or ||, no ternary, no ?. - do not add them.
#
# Each worker is launched with --cuda-device N so the process sees exactly one
# GPU. That is not a preference. ComfyUI 0.34.0+ sets CUDA_VISIBLE_DEVICES=0 on
# Windows whenever it sees two NVIDIA cards, because of an open CUDA bug
# (ComfyUI #15255) where the second card merely being visible poisons the
# first card's context. One GPU per process means the guard never fires.

param(
  [Parameter(Position = 0)]
  [ValidateSet('start', 'stop', 'restart', 'status', 'logs')]
  [string]$Action = 'status',

  [Parameter(Position = 1)]
  [string]$Worker = '',

  [switch]$Skip,

  # Bring up the dual-card worker INSTEAD of the pinned pair. Not alongside:
  # three ComfyUI processes would contend for the same 33.5 GB of host RAM, and
  # H3's 32B text encoder alone needs ~16 GB of it to load.
  [switch]$Big
)

$ErrorActionPreference = 'Stop'

$FarmDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$Project    = Split-Path -Parent $FarmDir
$StateDir   = Join-Path $FarmDir '_state'
$ComfyRoot  = if ($env:COMFY_ROOT) { $env:COMFY_ROOT } else { 'C:\ComfyUI' }
$ComfyPy    = Join-Path $ComfyRoot 'venv\Scripts\python.exe'
$ComfyMain  = Join-Path $ComfyRoot 'main.py'
$SysPy      = if ($env:CLAY_FILM_PY) { $env:CLAY_FILM_PY } else { 'python' }

# Keep in step with farm/config.py WORKERS.
#
# 'big' is a THIRD, normally-stopped worker that sees BOTH cards, for models
# too large for one 16 GB card - the 21 GB minimax_h3_fl2va_pruned_int8_convrot
# above all, which ComfyUI-MultiGPU's DisTorch2 loader splits across the pair.
# It cannot be one of the paired workers: those launch --cuda-device N and so
# see exactly one GPU, and a loader told to donate to cuda:1 in a process where
# cuda:1 does not exist cannot work.
#
# --cuda-device all is what skips ComfyUI's Windows guard (main.py line 97:
# the guard at line 48 only fires when cuda_device is None). The guard exists
# for ComfyUI issue #15255 - on Windows the second card merely being VISIBLE
# poisons the first card's CUDA context on the first host transfer - and
# --disable-pinned-memory is the documented price of turning it off. That costs
# async weight offload, so this worker is slower per job and is started only
# when a job actually needs both cards.
#
# Start it explicitly: farm.ps1 start -Big
$Workers = @(
  [pscustomobject]@{ Name = 'gpu0'; Device = 0; Port = 8190 },
  [pscustomobject]@{ Name = 'gpu1'; Device = 1; Port = 8191 }
)

$BigWorker = [pscustomobject]@{
  Name = 'big'; Device = 'all'; Port = 8192
  Extra = @('--disable-pinned-memory')
}

$BaseArgs = @(
  '--disable-auto-launch',
  '--listen', '127.0.0.1',
  '--async-offload', '2',
  '--vram-headroom', '0.5',
  '--fast-disk'
)

$BootTimeoutSec = 240

function Write-Line { param([string]$m) Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) }

function Get-WorkerPaths {
  param([string]$Name)
  $base = Join-Path $StateDir $Name
  return [pscustomobject]@{
    Base   = $base
    Output = Join-Path $base 'output'
    Temp   = Join-Path $base 'temp'
    Input  = Join-Path $base 'input'
    Log    = Join-Path $base 'comfy.log'
    Err    = Join-Path $base 'comfy.err.log'
    Pid    = Join-Path $base 'pid'
  }
}

function Test-Port {
  param([int]$Port)
  $c = New-Object System.Net.Sockets.TcpClient
  try {
    $ar = $c.BeginConnect('127.0.0.1', $Port, $null, $null)
    $ok = $ar.AsyncWaitHandle.WaitOne(400, $false)
    if ($ok) { $c.EndConnect($ar); return $true }
    return $false
  } catch {
    return $false
  } finally {
    $c.Close()
  }
}

function Test-Health {
  param([int]$Port)
  try {
    $r = Invoke-RestMethod "http://127.0.0.1:$Port/system_stats" -TimeoutSec 4
    return $r
  } catch {
    return $null
  }
}

function Get-WorkerPid {
  param($W)
  $p = (Get-WorkerPaths $W.Name).Pid
  if (-not (Test-Path $p)) { return 0 }
  $raw = (Get-Content $p -Raw).Trim()
  if ($raw -match '^\d+$') { return [int]$raw }
  return 0
}

# ---------------------------------------------------------------- start ---

function Start-Worker {
  param($W)

  $paths = Get-WorkerPaths $W.Name
  foreach ($d in @($paths.Base, $paths.Output, $paths.Temp, $paths.Input)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
  }

  if (Test-Port $W.Port) {
    Write-Line ("{0}: port {1} already listening, not starting a second one" -f $W.Name, $W.Port)
    return $true
  }

  # Start-Process -ArgumentList joins the array with spaces, so every element
  # holding a space has to carry its own quotes. This project lives under
  # "C:\AI Coding Projects\...", which splits at "AI" otherwise and produces
  # `main.py: error: unrecognized arguments: Coding Projects\...`.
  function Q { param([string]$s) return '"' + $s + '"' }

  $argList = @((Q $ComfyMain)) + $BaseArgs + @(
    '--port', $W.Port,
    '--cuda-device', $W.Device,
    '--output-directory', (Q $paths.Output),
    '--temp-directory', (Q $paths.Temp),
    '--input-directory', (Q $paths.Input)
  )
  # the big worker adds --disable-pinned-memory; see $BigWorker above
  if ($W.PSObject.Properties.Name -contains 'Extra' -and $W.Extra) {
    $argList += $W.Extra
  }

  # Blackwell allocator fragmentation produced spurious CUDA OOMs on this box;
  # expandable_segments was the fix. Set per-process, not globally.
  $env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

  Write-Line ("{0}: launching on GPU {1}, port {2}" -f $W.Name, $W.Device, $W.Port)
  $proc = Start-Process -FilePath $ComfyPy `
    -ArgumentList $argList `
    -WorkingDirectory $ComfyRoot `
    -RedirectStandardOutput $paths.Log `
    -RedirectStandardError $paths.Err `
    -WindowStyle Hidden `
    -PassThru

  Set-Content -Path $paths.Pid -Value $proc.Id -Encoding utf8
  return $true
}

function Wait-Worker {
  param($W, [int]$TimeoutSec)
  $paths = Get-WorkerPaths $W.Name
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    $stats = Test-Health $W.Port
    if ($null -ne $stats) {
      $dev = $stats.devices[0]
      $freeGb = [math]::Round($dev.vram_free / 1GB, 2)
      $totGb  = [math]::Round($dev.vram_total / 1GB, 2)
      Write-Line ("{0}: up. device='{1}' vram {2}/{3} GB free" -f $W.Name, $dev.name, $freeGb, $totGb)
      # A worker that reports more than one device means --cuda-device did not
      # take, which is the exact state the Windows CUDA bug punishes.
      # The big worker is the deliberate exception: it is SUPPOSED to see both.
      if ($stats.devices.Count -gt 1 -and $W.Device -ne 'all') {
        Write-Line ("{0}: WARNING - {1} devices visible, expected 1. Check --cuda-device." -f $W.Name, $stats.devices.Count)
      }
      return $true
    }
    $procId = Get-WorkerPid $W
    if ($procId -gt 0) {
      $alive = Get-Process -Id $procId -ErrorAction SilentlyContinue
      if ($null -eq $alive) {
        Write-Line ("{0}: process exited during boot. Last stderr:" -f $W.Name)
        if (Test-Path $paths.Err) { Get-Content $paths.Err -Tail 25 | ForEach-Object { Write-Host "    $_" } }
        return $false
      }
    }
    Start-Sleep -Milliseconds 1500
  }
  Write-Line ("{0}: did not answer within {1}s" -f $W.Name, $TimeoutSec)
  if (Test-Path $paths.Err) { Get-Content $paths.Err -Tail 25 | ForEach-Object { Write-Host "    $_" } }
  return $false
}

function Invoke-Start {
  if (-not $Skip) {
    Write-Line 'running preflight'
    $py = $SysPy
    if (-not (Test-Path $py)) { $py = 'python' }
    # Out-Host: this function's return value is captured by `exit (Invoke-Start)`,
    # so preflight's own output would otherwise be folded into the exit code.
    & $py (Join-Path $FarmDir 'preflight.py') | Out-Host
    if ($LASTEXITCODE -ne 0) {
      Write-Line 'preflight FAILED - not launching. Fix the items above, or pass -Skip.'
      return 1
    }
  }

  $targets = $Workers
  if ($Big) {
    Write-Line 'starting the DUAL-CARD worker (both GPUs visible, pinned memory off)'
    $targets = @($BigWorker)
  }

  $allOk = $true
  foreach ($w in $targets) {
    $started = Start-Worker $w
    if (-not $started) { $allOk = $false }
  }
  # Boot serially. Two cold ComfyUI processes reading weights off one NVMe at
  # once is slower than doing it one at a time, and a boot failure is easier to
  # attribute.
  foreach ($w in $targets) {
    $ok = Wait-Worker $w $BootTimeoutSec
    if (-not $ok) { $allOk = $false }
  }
  if ($allOk) { Write-Line 'farm up' } else { Write-Line 'farm came up DEGRADED - see above' }
  Invoke-Status | Out-Null
  if ($allOk) { return 0 }
  return 1
}

# ----------------------------------------------------------------- stop ---

function Invoke-Stop {
  # always sweep the big worker too: `stop` has to mean stop, whether or
  # not -Big was passed, or a forgotten dual-card process keeps both cards'
  # VRAM and the pinned pair will not start.
  $targets = $Workers + @($BigWorker)
  foreach ($w in $targets) {
    $paths = Get-WorkerPaths $w.Name
    $procId = Get-WorkerPid $w
    if ($procId -le 0) {
      Write-Line ("{0}: no recorded pid" -f $w.Name)
    } else {
      $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
      if ($null -eq $proc) {
        Write-Line ("{0}: pid {1} already gone" -f $w.Name, $procId)
      } else {
        Write-Line ("{0}: stopping pid {1}" -f $w.Name, $procId)
        try { Stop-Process -Id $procId -Force -Confirm:$false -ErrorAction Stop } catch { Write-Line ("  {0}" -f $_.Exception.Message) }
      }
      Remove-Item $paths.Pid -Force -ErrorAction SilentlyContinue
    }
  }
  Start-Sleep -Seconds 2
  foreach ($w in $targets) {
    if (Test-Port $w.Port) {
      Write-Line ("{0}: port {1} STILL listening. Some GPU processes on this box survive taskkill; if VRAM stays held, a reboot is the only fix." -f $w.Name, $w.Port)
    }
  }
  Invoke-Status | Out-Null
  return 0
}

# --------------------------------------------------------------- status ---

function Invoke-Status {
  Write-Host ''
  $rows = @()
  foreach ($w in $Workers) {
    $listening = Test-Port $w.Port
    $stats = $null
    if ($listening) { $stats = Test-Health $w.Port }
    $state = 'down'
    $devName = ''
    $vram = ''
    if ($listening -and $null -ne $stats) {
      $state = 'up'
      $devName = $stats.devices[0].name
      $vram = '{0:N1}/{1:N1} GB free' -f ($stats.devices[0].vram_free / 1GB), ($stats.devices[0].vram_total / 1GB)
      if ($stats.devices.Count -gt 1) { $state = 'up (MULTI-DEV!)' }
    } elseif ($listening) {
      $state = 'listening, not answering'
    }
    $rows += [pscustomobject]@{
      Worker = $w.Name; GPU = $w.Device; Port = $w.Port
      State = $state; ProcId = (Get-WorkerPid $w); Device = $devName; VRAM = $vram
    }
  }
  # Out-Host, not a bare pipeline: this function's return value is captured by
  # `exit (Invoke-Status)`, and Format-Table would otherwise be folded into it.
  $rows | Format-Table -AutoSize | Out-Host

  $smi = nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  Write-Host 'nvidia-smi (index, used MiB, total MiB, util %):'
  foreach ($line in $smi) { Write-Host ("  {0}" -f $line) }
  if (Test-Port 8080) { Write-Host '  note: 8080 is listening (llama-server / the 27B holds both cards in tensor-split mode)' }
  Write-Host ''
  return 0
}

# ----------------------------------------------------------------- logs ---

function Invoke-Logs {
  $name = $Worker
  if ([string]::IsNullOrWhiteSpace($name)) { $name = 'gpu0' }
  $paths = Get-WorkerPaths $name
  Write-Host ("--- {0} stdout ---" -f $name)
  if (Test-Path $paths.Log) { Get-Content $paths.Log -Tail 40 }
  Write-Host ("--- {0} stderr ---" -f $name)
  if (Test-Path $paths.Err) { Get-Content $paths.Err -Tail 40 }
  return 0
}

# ----------------------------------------------------------------- main ---

switch ($Action) {
  'start'   { exit (Invoke-Start) }
  'stop'    { exit (Invoke-Stop) }
  'restart' { Invoke-Stop | Out-Null; Start-Sleep -Seconds 3; exit (Invoke-Start) }
  'status'  { exit (Invoke-Status) }
  'logs'    { exit (Invoke-Logs) }
}
