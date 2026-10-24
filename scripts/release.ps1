# PowerShell wrapper for scripts/release.sh (Windows convenience).
#
# Usage:
#   .\scripts\release.ps1
#   .\scripts\release.ps1 --models
#
# Requires Git for Windows (which provides bash.exe). The .sh scripts are the
# source of truth so macOS/Linux use the same logic.
#
# NOTE: no `$ErrorActionPreference = "Stop"` here. Windows PowerShell 5.1 turns
# the first stderr line of a native command (cargo prints its progress there)
# into a terminating NativeCommandError, which killed this wrapper right after
# the first gate. We check $LASTEXITCODE instead, like the .sh does.

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

# Format the child's stderr as plain text: PowerShell wraps each stderr line in
# an ErrorRecord, and a blank one would print as "System.Management.Automation.
# RemoteException" instead of nothing.
& $bash (Join-Path $PSScriptRoot "release.sh") @args 2>&1 | ForEach-Object {
  if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { "$_" }
}
exit $LASTEXITCODE
