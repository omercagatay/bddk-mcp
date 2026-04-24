# scripts/pull_benchmark_models.ps1
# Pulls every model referenced by benchmark/config.py so the benchmark harness
# can run without hitting "model not found" halfway through Phase 1.
#
# Usage (from repo root, PowerShell):
#     .\scripts\pull_benchmark_models.ps1
#
# Prereqs: Ollama installed and running (`ollama serve` in the background, or
# the Windows tray service started). The script talks to the local daemon via
# `ollama pull`, so no extra auth is needed.
#
# Total download size: roughly 80 GB across all 7 models. Each `ollama pull`
# is resumable — interrupting and rerunning picks up where it left off.

$ErrorActionPreference = "Stop"

$Models = @(
    "gemma4:e2b-it-q8_0",
    "gemma4:e4b-it-q8_0",
    "gemma4:26b-a4b-it-q4_K_M",
    "qwen3.5:9b-q8_0",
    "qwen3.6:27b-q4_K_M",
    "qwen3.6:35b-a3b-q4_K_M",
    "huihui_ai/qwen3.5-abliterated:9b-q8_0"
)

# Preflight: Ollama reachable?
try {
    $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5
} catch {
    Write-Host "ERROR: Ollama daemon not reachable at http://localhost:11434" -ForegroundColor Red
    Write-Host "  Start it with 'ollama serve' or ensure the Windows service is running, then retry."
    exit 1
}

$Total = $Models.Count
$Index = 0
foreach ($Model in $Models) {
    $Index++
    Write-Host ""
    Write-Host "[$Index/$Total] Pulling $Model" -ForegroundColor Cyan
    & ollama pull $Model
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pull failed for $Model (exit $LASTEXITCODE). Retry the script — previous pulls are cached." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "All $Total models pulled. Current library:" -ForegroundColor Green
& ollama list
