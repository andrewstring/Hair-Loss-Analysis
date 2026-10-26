<#
.SYNOPSIS
  One-step Windows setup for Scalp Analyzer (Hair-Loss-Analysis).

.DESCRIPTION
  - Finds Python 3.10-3.13 (installs 3.12 via winget if missing)
  - Creates .venv and installs requirements-base.txt
  - Installs the right ONNX Runtime build for this PC
  - Creates data/models/samples/outputs folders and default calibration
  - Runs check_env.py to verify everything

.PARAMETER Mode
  auto      (default) DirectML if a GPU is found, otherwise CPU
  cpu       onnxruntime (CPU only)
  directml  onnxruntime-directml (any DX12 GPU: AMD, NVIDIA, Intel)
  cuda      onnxruntime-gpu (NVIDIA only; needs CUDA + cuDNN installed)

.PARAMETER Recreate
  Delete and rebuild .venv from scratch.

.PARAMETER NoPythonInstall
  Do not try to install Python with winget if it is missing.

.EXAMPLE
  .\windows\setup_windows.ps1
  .\windows\setup_windows.ps1 -Mode cpu
  .\windows\setup_windows.ps1 -Recreate
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', 'cpu', 'directml', 'cuda')]
    [string]$Mode = 'auto',
    [switch]$Recreate,
    [switch]$NoPythonInstall
)

$ErrorActionPreference = 'Stop'
# This script lives in <project>\windows\ ; work from the project root one level up.
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

# ---------------------------------------------------------------- helpers ---
function Write-Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2([string]$msg){ Write-Host "    $msg" -ForegroundColor Yellow }
function Fail([string]$msg)       { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

# Run a native command; never throw on stderr output, return the exit code.
function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Exe @Arguments; return $LASTEXITCODE }
    finally { $ErrorActionPreference = $prev }
}

# Re-read PATH so a just-installed Python is visible without reopening the shell.
function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ';'
}

# Returns @{ Exe=..; Args=@(..); Version=".." } for a usable Python 3.10-3.13, or $null.
function Find-Python {
    $candidates = @(
        @{ Exe = 'py';     Args = @('-3.12') },
        @{ Exe = 'py';     Args = @('-3.11') },
        @{ Exe = 'py';     Args = @('-3.13') },
        @{ Exe = 'py';     Args = @('-3.10') },
        @{ Exe = 'python'; Args = @() },
        @{ Exe = 'python3'; Args = @() }
    )
    foreach ($c in $candidates) {
        if (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
        try {
            $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
            $out = & $c.Exe @($c.Args + @('-c', 'import sys;print("%d.%d" % sys.version_info[:2])')) 2>$null
            $ErrorActionPreference = $prev
        } catch { $ErrorActionPreference = $prev; continue }
        # The Microsoft Store "python" stub prints nothing useful; require a clean version string.
        if ($out -match '^(\d+)\.(\d+)$') {
            $maj = [int]$Matches[1]; $min = [int]$Matches[2]
            if ($maj -eq 3 -and $min -ge 10 -and $min -le 13) {
                return @{ Exe = $c.Exe; Args = $c.Args; Version = "$maj.$min" }
            }
        }
    }
    return $null
}

# ---------------------------------------------------------- 1. Python check ---
Write-Step 'Checking for Python 3.10 - 3.13'
$py = Find-Python

if (-not $py -and -not $NoPythonInstall) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Warn2 'Python not found. Installing Python 3.12 with winget (per-user)...'
        $rc = Invoke-Native 'winget' @('install', '-e', '--id', 'Python.Python.3.12',
            '--scope', 'user', '--accept-package-agreements', '--accept-source-agreements')
        if ($rc -ne 0) { Write-Warn2 "winget exited with code $rc" }
        Update-SessionPath
        $py = Find-Python
    } else {
        Write-Warn2 'winget is not available on this PC.'
    }
}

if (-not $py) {
    Fail ("No compatible Python found.`n" +
          "    Install Python 3.12 from https://www.python.org/downloads/windows/`n" +
          "    (tick 'Add python.exe to PATH'), then run this script again.")
}
Write-Ok "Using Python $($py.Version)  [$($py.Exe) $($py.Args -join ' ')]"

# -------------------------------------------------------------- 2. venv ------
$venvDir = Join-Path $Root '.venv'
$vpy     = Join-Path $venvDir 'Scripts\python.exe'

if ($Recreate -and (Test-Path $venvDir)) {
    Write-Step 'Removing existing .venv (-Recreate)'
    Remove-Item -Recurse -Force $venvDir
}

Write-Step 'Creating virtual environment (.venv)'
if (-not (Test-Path $vpy)) {
    $rc = Invoke-Native $py.Exe ($py.Args + @('-m', 'venv', $venvDir))
    if ($rc -ne 0 -or -not (Test-Path $vpy)) { Fail 'Could not create the virtual environment.' }
    Write-Ok 'Created .venv'
} else {
    Write-Ok '.venv already exists (use -Recreate to rebuild it)'
}

# -------------------------------------------------------- 3. base packages ---
Write-Step 'Upgrading pip'
[void](Invoke-Native $vpy @('-m', 'pip', 'install', '--upgrade', 'pip', '--quiet', '--disable-pip-version-check'))

Write-Step 'Installing base dependencies (numpy, opencv-python)'
if (-not (Test-Path 'requirements-base.txt')) { Fail 'requirements-base.txt not found. Keep the windows\ folder inside the project folder.' }
$rc = Invoke-Native $vpy @('-m', 'pip', 'install', '-r', 'requirements-base.txt', '--disable-pip-version-check')
if ($rc -ne 0) { Fail 'Failed to install base dependencies. See the pip output above.' }

# ------------------------------------------------------ 4. ONNX Runtime ------
Write-Step 'Selecting ONNX Runtime build'

$gpuNames = @()
try {
    $gpuNames = @(Get-CimInstance Win32_VideoController -ErrorAction Stop |
        Where-Object { $_.Name -and $_.Name -notmatch 'Basic|Remote|Virtual|Parsec|Citrix' } |
        ForEach-Object { $_.Name })
} catch { }

if ($gpuNames.Count -gt 0) { Write-Ok ("Detected GPU(s): " + ($gpuNames -join '; ')) }
else { Write-Ok 'No discrete/integrated GPU detected.' }

$chosen = $Mode
if ($Mode -eq 'auto') { $chosen = if ($gpuNames.Count -gt 0) { 'directml' } else { 'cpu' } }
if ($Mode -eq 'cuda' -and -not ($gpuNames -match 'NVIDIA')) {
    Write-Warn2 'No NVIDIA GPU detected; CUDA mode will not work. Falling back to DirectML.'
    $chosen = 'directml'
}

$package = switch ($chosen) {
    'cpu'      { 'onnxruntime' }
    'directml' { 'onnxruntime-directml' }
    'cuda'     { 'onnxruntime-gpu' }
}
Write-Ok "Mode: $chosen  ->  $package"

# Only one onnxruntime variant may be installed at a time.
[void](Invoke-Native $vpy @('-m', 'pip', 'uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu',
        'onnxruntime-directml', '--quiet', '--disable-pip-version-check'))

$rc = Invoke-Native $vpy @('-m', 'pip', 'install', $package, '--disable-pip-version-check')
if ($rc -ne 0 -and $chosen -ne 'cpu') {
    Write-Warn2 "$package failed to install; falling back to CPU build."
    $chosen = 'cpu'; $package = 'onnxruntime'
    $rc = Invoke-Native $vpy @('-m', 'pip', 'install', $package, '--disable-pip-version-check')
}
if ($rc -ne 0) { Fail 'Failed to install ONNX Runtime. See the pip output above.' }

# DirectML exposes "DmlExecutionProvider", which the project's config.py does not
# list yet. Add it (idempotent, with a .bak backup) so the GPU is actually used.
if ($chosen -eq 'directml') {
    $cfg = Join-Path $Root 'config.py'
    if (Test-Path $cfg) {
        $text = [System.IO.File]::ReadAllText($cfg)
        if ($text -notmatch 'DmlExecutionProvider') {
            $patched = $text -replace '(?m)^([ \t]*)"CUDAExecutionProvider",', "`$1`"CUDAExecutionProvider`",`n`$1`"DmlExecutionProvider`","
            if ($patched -ne $text) {
                Copy-Item $cfg "$cfg.bak" -Force
                [System.IO.File]::WriteAllText($cfg, $patched, (New-Object System.Text.UTF8Encoding($false)))
                Write-Ok 'Added DmlExecutionProvider to config.py (backup: config.py.bak)'
            } else {
                Write-Warn2 'Could not patch config.py automatically; add "DmlExecutionProvider" to PROVIDER_PREFERENCE manually.'
            }
        }
    }
}

# ------------------------------------------------------ 5. scaffolding -------
Write-Step 'Creating project folders'
foreach ($d in 'data', 'models', 'samples', 'outputs') {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

$calib = Join-Path $Root 'data\calibration_reference.json'
if (-not (Test-Path $calib)) {
    $json = '{"x20": {"px_per_mm": 1180.0}, "x50": {"px_per_mm": 2950.0}, "x100": {"px_per_mm": 5900.0}}'
    [System.IO.File]::WriteAllText($calib, $json, (New-Object System.Text.UTF8Encoding($false)))
    Write-Ok 'Wrote default data\calibration_reference.json'
}
Write-Ok 'Folders ready'

# ------------------------------------------------------- 6. verification -----
Write-Step 'Verifying installation'
$env:PYTHONUTF8 = '1'
if (Test-Path 'check_env.py') {
    [void](Invoke-Native $vpy @('check_env.py'))
} else {
    [void](Invoke-Native $vpy @('-c', 'import numpy, cv2, onnxruntime as o; print(o.get_available_providers())'))
}

Write-Host ''
Write-Host '======================================================' -ForegroundColor Green
Write-Host ' Installation complete.' -ForegroundColor Green
Write-Host '======================================================' -ForegroundColor Green
Write-Host ' Analyze an image:'
Write-Host '   - Drag a photo onto  windows\run_analysis.bat   or'
Write-Host '   - .\windows\run_analysis.bat C:\path\to\scalp.jpg x20'
Write-Host ' Manual use:'
Write-Host '   .\.venv\Scripts\Activate.ps1'
Write-Host '   python main.py C:\path\to\scalp.jpg --mag x20 --json outputs\report.json'
Write-Host ''
Write-Host ' Note: the GPU is only used once .onnx models exist in models\ ;'
Write-Host '       without them the built-in OpenCV heuristics run on the CPU.'
exit 0
