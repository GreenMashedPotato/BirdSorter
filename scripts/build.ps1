# PowerShell wrapper for scripts/build.sh (Windows convenience).
#
# Usage:
#   .\scripts\build.ps1
#   .\scripts\build.ps1 --models --bundle
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
  throw "Git Bash is required. Install Git for Windows, or run scripts/build.sh from a shell."
}

& $bash (Join-Path $PSScriptRoot "build.sh") @args
exit $LASTEXITCODE
