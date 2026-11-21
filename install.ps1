$ErrorActionPreference = "Stop"
$Dest = Join-Path $env:LOCALAPPDATA "NALA-ModelForge"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Copy-Item -Force (Join-Path $PSScriptRoot "NALA-ModelForge.exe") (Join-Path $Dest "NALA-ModelForge.exe")
Write-Host "Installed NALA-ModelForge-RS to $Dest\NALA-ModelForge.exe"
Write-Host "Launch studio: & '$Dest\NALA-ModelForge.exe' --open"
Write-Host "MCP stdio:     & '$Dest\NALA-ModelForge.exe' --mcp-stdio"
