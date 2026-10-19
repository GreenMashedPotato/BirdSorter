# PowerShell wrapper for scripts/release.sh (Windows convenience).
#
# Usage:
#   .\scripts\release.ps1
#   .\scripts\release.ps1 --models
#
# Requires Git for Windows (which provides bash.exe). The .sh scripts are the
# source of truth so macOS/Linux use the same logic.

$ErrorActionPreference = "Stop"

$bash = @(
  "C:\Program Files\Git\bin\bash.exe",
  "C:\Program Files\Git\usr\bin\bash.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $bash) {
  $cmd = Get-Command bash -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source -match "Git") { $bash = $cmd.Source }
}

if (-not $bash) {
  throw "Git Bash is required. Install Git for Windows, or run scripts/release.sh from a shell."
}

& $bash (Join-Path $PSScriptRoot "release.sh") @args
exit $LASTEXITCODE
