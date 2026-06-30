param(
    [string]$Date = "",
    [string]$OutDir = "data\intraday_live",
    [int]$RefreshSeconds = 300,
    [int]$HistoryDays = 120,
    [int]$PrimaryLookbackDays = 30,
    [int]$CreditQuoteLimit = 80,
    [string]$CreditWatchlist = "",
    [int]$Timeout = 30,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$CredentialScript = Join-Path $PSScriptRoot "set_dm_credentials.local.ps1"
if (Test-Path $CredentialScript) {
    . $CredentialScript *> $null
} else {
    Write-Warning "Missing scripts\set_dm_credentials.local.ps1. Please set INNO_APP_KEY and INNO_APP_SECRET first."
}

while ($true) {
    $RunDate = if ($Date) { $Date } else { Get-Date -Format "yyyy-MM-dd" }
    $BaseDir = Join-Path $OutDir $RunDate
    $SnapshotDir = Join-Path $BaseDir "snapshot"
    $ContextDir = Join-Path $BaseDir "context"
    $ReportPath = Join-Path $BaseDir "intraday_market_dashboard.html"

    New-Item -ItemType Directory -Force -Path $BaseDir | Out-Null

    python ".\scripts\run_dm_intraday_snapshot.py" `
        "--date" $RunDate `
        "--out-dir" $SnapshotDir `
        "--timeout" $Timeout

    $ContextArgs = @(
        ".\scripts\run_dm_market_context.py",
        "--date", $RunDate,
        "--out-dir", $ContextDir,
        "--history-days", "$HistoryDays",
        "--primary-lookback-days", "$PrimaryLookbackDays",
        "--credit-quote-limit", "$CreditQuoteLimit",
        "--timeout", "$Timeout"
    )
    if ($CreditWatchlist) {
        $ContextArgs += @("--credit-watchlist", $CreditWatchlist)
    }
    python @ContextArgs

    python ".\scripts\run_intraday_market_dashboard.py" `
        "--snapshot-dir" $SnapshotDir `
        "--context-dir" $ContextDir `
        "--out" $ReportPath `
        "--refresh-seconds" $RefreshSeconds

    Write-Host "日内市场网页已更新：$ReportPath"

    if ($Once) {
        break
    }
    Start-Sleep -Seconds $RefreshSeconds
}
