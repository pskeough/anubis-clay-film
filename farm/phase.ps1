# phase.ps1 - hand both GPUs between the 27B and the ComfyUI farm.
#
#   .\farm\phase.ps1 status
#   .\farm\phase.ps1 gpu            # release the LLM, bring the farm up
#   .\farm\phase.ps1 gpu -Force     # also stop Voice_Chat if it holds VRAM
#   .\farm\phase.ps1 llm            # stop the farm, bring the 27B back up
#   .\farm\phase.ps1 free           # release everything, start nothing
#
# The 27B in tensor-split mode occupies BOTH cards (~13 GB each), so the LLM
# phase and the diffusion phase cannot overlap in time. This is the same
# conclusion VN_Generation_Base reached for the same reason.
#
# Waiting is done by POLLING nvidia-smi, never by sleeping. A fixed sleep is
# what made llama-server die with 0xC0000142 while ComfyUI still held ~10 GB.
# gpu-lease.ts in the Roko pipeline learned this the same way.
#
# Windows PowerShell 5.1: no && or ||, no ternary.

param(
  [Parameter(Position = 0)]
  [ValidateSet('status', 'gpu', 'llm', 'free')]
  [string]$Action = 'status',

  [switch]$Force,                 # also stop Voice_Chat
  # Bring up the DUAL-CARD worker instead of the pinned pair. MiniMax H3's
  # int8 build is 21 GB and will not load on a single 16 GB card, so the video
  # phase needs the one process that sees both. The pair is still the right
  # choice for image-only work, which can run two jobs at once.
  [switch]$Big,
  [int]$Ctx = 262144,             # context for the 27B when returning to llm phase
  [int]$WaitSeconds = 120
)

$ErrorActionPreference = 'Stop'

$FarmDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$FarmPs1   = Join-Path $FarmDir 'farm.ps1'
. (Join-Path $FarmDir 'envfile.ps1')
Import-RepoEnv (Join-Path (Split-Path -Parent $FarmDir) '.env')   # the repo-root .env; the shell wins
$LlmScript = $env:CLAY_FILM_LLM_SCRIPT  # optional: a script that starts/stops a local LLM sharing the GPUs

# A card is considered free below this. The 27B sits at ~13 GB per card, so
# anything under 3 GB means the heavyweight tenants are genuinely gone; the
# residue is display compositing on GPU0.
$FreeThresholdMiB = 3000

function Write-Line { param([string]$m) Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) }

function Get-GpuUsed {
  $vals = nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
  $out = @()
  foreach ($line in $vals) {
    $parts = $line -split ','
    $out += [pscustomobject]@{ Index = [int]$parts[0].Trim(); UsedMiB = [int]$parts[1].Trim() }
  }
  return $out
}

function Get-GpuHolders {
  # Real compute clients, not every process that merely touched the display.
  $names = @('llama-server', 'python', 'pythonw', 'ComfyUI')
  $procIds = nvidia-smi --query-compute-apps=pid --format=csv,noheader
  $out = @()
  foreach ($raw in $procIds) {
    $t = "$raw".Trim()
    if ($t -notmatch '^\d+$') { continue }
    $p = Get-Process -Id ([int]$t) -ErrorAction SilentlyContinue
    if ($null -eq $p) { continue }
    if ($names -contains $p.ProcessName) {
      $cmd = ''
      try { $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId = $($p.Id)" -ErrorAction SilentlyContinue).CommandLine } catch { $cmd = '' }
      $out += [pscustomobject]@{ Id = $p.Id; Name = $p.ProcessName; WorkingSetGB = [math]::Round($p.WS / 1GB, 2); CommandLine = $cmd }
    }
  }
  return $out | Sort-Object -Property Id -Unique
}

function Wait-Vram {
  param([int]$ThresholdMiB, [int]$TimeoutSec)
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    $cards = Get-GpuUsed
    $worst = ($cards | Measure-Object -Property UsedMiB -Maximum).Maximum
    if ($worst -le $ThresholdMiB) {
      Write-Line ("vram released: max {0} MiB used across cards" -f $worst)
      return $true
    }
    Start-Sleep -Milliseconds 1500
  }
  $cards = Get-GpuUsed
  foreach ($c in $cards) { Write-Line ("  gpu{0}: {1} MiB still held" -f $c.Index, $c.UsedMiB) }
  Write-Line ("vram did NOT drop below {0} MiB within {1}s" -f $ThresholdMiB, $TimeoutSec)
  Write-Line 'Some GPU processes on this box survive taskkill and keep answering HTTP. If this persists, a reboot is the only fix.'
  return $false
}

function Stop-Llm {
  if (-not (Test-Path $LlmScript)) {
    Write-Line "llm launcher not found at $LlmScript"
    return
  }
  Write-Line 'stopping the 27B through its own launcher (-Stop)'
  & powershell -NoProfile -ExecutionPolicy Bypass -File $LlmScript -Stop | Out-Host
}

function Stop-VoiceChat {
  $holders = Get-GpuHolders | Where-Object { $_.CommandLine -like '*Voice_Chat*' }
  if ($holders.Count -eq 0) { return }
  foreach ($h in $holders) {
    if ($Force) {
      Write-Line ("stopping Voice_Chat pid {0}" -f $h.Id)
      try { Stop-Process -Id $h.Id -Force -Confirm:$false -ErrorAction Stop } catch { Write-Line ("  {0}" -f $_.Exception.Message) }
    } else {
      Write-Line ("Voice_Chat is holding a card (pid {0}, {1} GB). Leaving it alone - pass -Force to stop it." -f $h.Id, $h.WorkingSetGB)
    }
  }
}

# --------------------------------------------------------------- status ---

function Invoke-Status {
  Write-Host ''
  Write-Host 'GPU memory:'
  foreach ($c in Get-GpuUsed) {
    Write-Host ("  gpu{0}: {1,6} MiB used of 16311" -f $c.Index, $c.UsedMiB)
  }
  Write-Host ''
  $holders = Get-GpuHolders
  if ($holders.Count -eq 0) {
    Write-Host 'No heavyweight GPU tenants.'
  } else {
    Write-Host 'GPU tenants:'
    foreach ($h in $holders) {
      $cmd = "$($h.CommandLine)"
      if ($cmd.Length -gt 90) { $cmd = $cmd.Substring(0, 90) + '...' }
      Write-Host ("  pid {0,-6} {1,-14} {2,6} GB  {3}" -f $h.Id, $h.Name, $h.WorkingSetGB, $cmd)
    }
  }
  Write-Host ''
  & $FarmPs1 status | Out-Host
  return 0
}

# ------------------------------------------------------------ gpu phase ---

function Invoke-GpuPhase {
  Write-Line 'entering GPU phase: the farm gets both cards'
  Stop-Llm
  Stop-VoiceChat
  $ok = Wait-Vram $FreeThresholdMiB $WaitSeconds
  if (-not $ok) {
    Write-Line 'not launching the farm into a card that is still occupied'
    Invoke-Status | Out-Null
    return 1
  }
  if ($Big) {
    Write-Line 'starting the DUAL-CARD worker (H3 int8 is 21 GB, needs both)'
    & $FarmPs1 start -Big -Skip | Out-Host
  } else {
    & $FarmPs1 start | Out-Host
  }
  return $LASTEXITCODE
}

# ------------------------------------------------------------ llm phase ---

function Invoke-LlmPhase {
  Write-Line 'entering LLM phase: the 27B gets both cards'
  & $FarmPs1 stop | Out-Host
  $ok = Wait-Vram $FreeThresholdMiB $WaitSeconds
  if (-not $ok) {
    Write-Line 'not launching llama-server into a card that is still occupied'
    return 1
  }
  if (-not (Test-Path $LlmScript)) {
    Write-Line "llm launcher not found at $LlmScript"
    return 1
  }
  Write-Line ("starting the 27B at ctx {0}" -f $Ctx)
  # -Effort is left at the launcher's own default on purpose. This GGUF's chat
  # template accepts only low / medium / xhigh; anything else raises inside
  # Jinja and returns HTTP 500 on every request while /health still reports OK.
  & powershell -NoProfile -ExecutionPolicy Bypass -File $LlmScript -Ctx $Ctx | Out-Host
  return 0
}

# ----------------------------------------------------------------- free ---

function Invoke-Free {
  Write-Line 'releasing everything, starting nothing'
  & $FarmPs1 stop | Out-Host
  Stop-Llm
  Stop-VoiceChat
  Wait-Vram $FreeThresholdMiB $WaitSeconds | Out-Null
  Invoke-Status | Out-Null
  return 0
}

switch ($Action) {
  'status' { exit (Invoke-Status) }
  'gpu'    { exit (Invoke-GpuPhase) }
  'llm'    { exit (Invoke-LlmPhase) }
  'free'   { exit (Invoke-Free) }
}
