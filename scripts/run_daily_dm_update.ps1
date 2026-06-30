param(
    [string]$StartDate = "",
    [string]$EndDate = "",
    [switch]$FullRefresh,
    [switch]$SkipFetch,
    [string]$OutDir = 'data\dm_daily',
    [string]$Device = 'cuda',
    [string]$ModelReadyMaxMissingRatio = '0.2',
    [int]$Timeout = 60
)

$ErrorActionPreference = 'Stop'

$Root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $Root

$CredentialScript = Join-Path $PSScriptRoot 'set_dm_credentials.local.ps1'
if (Test-Path $CredentialScript) {
    . $CredentialScript
} else {
    Write-Warning 'Missing scripts\set_dm_credentials.local.ps1. Please set INNO_APP_KEY and INNO_APP_SECRET first.'
}

$ArgsList = @(
    '-m', 'creditbond_ai.cli', 'daily-dm-update',
    '--config', 'configs\dm_master_indicators.csv',
    '--out-dir', $OutDir,
    '--model-ready-max-missing-ratio', $ModelReadyMaxMissingRatio,
    '--model-dir', 'models\curve_2020_AAA3Y_h5\01_full_features\gru',
    '--model-dir', 'models\curve_2020_AAA3Y_h5\01_full_features\tcn',
    '--model-dir', 'models\curve_2020_AAA3Y_h5\01_full_features\transformer',
    '--model-dir', 'models\curve_2020_AAA5Y_h5\01_full_features\gru',
    '--model-dir', 'models\curve_2020_AAA5Y_h5\01_full_features\tcn',
    '--model-dir', 'models\curve_2020_AAA5Y_h5\01_full_features\transformer',
    '--model-dir', 'models\curve_2020_AAA10Y_h5\01_full_features\gru',
    '--model-dir', 'models\curve_2020_AAA10Y_h5\01_full_features\tcn',
    '--model-dir', 'models\curve_2020_AAA10Y_h5\01_full_features\transformer',
    '--model-dir', 'models\curve_2020_AAAp20Y_h5\01_full_features\gru',
    '--model-dir', 'models\curve_2020_AAAp20Y_h5\01_full_features\tcn',
    '--model-dir', 'models\curve_2020_AAAp20Y_h5\01_full_features\transformer',
    '--device', $Device,
    '--timeout', $Timeout
)

if ($StartDate) {
    $ArgsList += @('--start-date', $StartDate)
}
if ($EndDate) {
    $ArgsList += @('--end-date', $EndDate)
}
if ($FullRefresh) {
    $ArgsList += '--full-refresh'
}
if ($SkipFetch) {
    $ArgsList += '--skip-fetch'
}

python @ArgsList
