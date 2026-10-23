# envfile.ps1 - dot-sourced by farm.ps1 and phase.ps1: read the repo-root .env into $env:.
#
# The same rules as clay_film/envfile.py, so a .env means the same thing to the scripts and to
# Python: blank lines and # comments skipped, an optional `export ` prefix, single quotes literal,
# double quotes with \n \t \" \\ unescaped, an unquoted value cut at " #". A variable already set
# in the environment is never overridden, an empty value leaves it unset, and a value starting
# with ./ or ../ is read relative to the repo root.
#
# Windows PowerShell 5.1: no && or ||, no ternary, no ?. - do not add them.

function Import-RepoEnv {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }
  $root = Split-Path -Parent $Path
  foreach ($line in [System.IO.File]::ReadAllLines($Path)) {
    $t = $line.Trim()
    if ($t -eq '' -or $t.StartsWith('#')) { continue }
    if ($t.StartsWith('export ')) { $t = $t.Substring(7).TrimStart() }
    $eq = $t.IndexOf('=')
    if ($eq -lt 1) { continue }
    $key = $t.Substring(0, $eq).Trim()
    if ($key -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') { continue }
    $raw = $t.Substring($eq + 1).Trim()
    if ($raw.StartsWith('"') -or $raw.StartsWith("'")) {
      $q = $raw[0]
      $sb = New-Object System.Text.StringBuilder
      $i = 1
      while ($i -lt $raw.Length -and $raw[$i] -ne $q) {
        if ($q -eq '"' -and $raw[$i] -eq '\' -and $i + 1 -lt $raw.Length) {
          $c = $raw[$i + 1]
          if ($c -eq 'n') { [void]$sb.Append("`n") }
          elseif ($c -eq 't') { [void]$sb.Append("`t") }
          elseif ($c -eq '"' -or $c -eq '\') { [void]$sb.Append($c) }
          else { [void]$sb.Append('\').Append($c) }
          $i += 2
          continue
        }
        [void]$sb.Append($raw[$i])
        $i += 1
      }
      $val = $sb.ToString()
    } else {
      $m = [regex]::Match($raw, '\s#')
      if ($m.Success) { $val = $raw.Substring(0, $m.Index).TrimEnd() } else { $val = $raw }
    }
    if ($val -eq '') { continue }
    if (Test-Path -LiteralPath ("env:" + $key)) { continue }
    if ($val -match '^\.\.?[\\/]') { $val = [System.IO.Path]::GetFullPath((Join-Path $root $val)) }
    Set-Item -LiteralPath ("env:" + $key) -Value $val
  }
}
