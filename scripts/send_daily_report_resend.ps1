param(
    [string]$To = "yuweiqian@cib.com.cn",
    [string]$Report = "data\dm_daily_master_curve_2020\reports\daily_dm_update_report.html",
    [string]$JsonReport = "data\dm_daily_master_curve_2020\reports\daily_dm_update_report.json",
    [string]$Pdf = "output\daily_reports\creditbond_ai_daily_latest.pdf",
    [string]$Subject = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$CredentialScript = Join-Path $PSScriptRoot "set_resend_credentials.local.ps1"
$ExampleScript = Join-Path $PSScriptRoot "set_resend_credentials.example.ps1"

if (-not (Test-Path $CredentialScript)) {
    Copy-Item -LiteralPath $ExampleScript -Destination $CredentialScript
    Write-Host "Created $CredentialScript"
    Write-Host "Please edit RESEND_API_KEY and RESEND_FROM, save, then run this script again."
    notepad $CredentialScript
    exit 1
}

. $CredentialScript

if (-not $env:RESEND_API_KEY -or $env:RESEND_API_KEY -like "*xxxxxxxx*") {
    throw "RESEND_API_KEY is still a placeholder. Edit scripts\set_resend_credentials.local.ps1."
}
if (-not $env:RESEND_FROM -or $env:RESEND_FROM -like "*your-verified-domain*") {
    throw "RESEND_FROM is still a placeholder. Edit scripts\set_resend_credentials.local.ps1."
}

if (-not (Test-Path $Pdf)) {
    python "scripts\build_daily_report_pdf.py" "--json-report" $JsonReport "--html-report" $Report "--out" $Pdf
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$ArgsList = @(
    "scripts\send_daily_report_resend.py",
    "--to", $To,
    "--from-email", $env:RESEND_FROM,
    "--json-report", $JsonReport,
    "--report", $Report,
    "--pdf", $Pdf
)

if ($Subject) {
    $ArgsList += @("--subject", $Subject)
}

if ($DryRun) {
    $ArgsList += "--dry-run"
}

python @ArgsList
exit $LASTEXITCODE
