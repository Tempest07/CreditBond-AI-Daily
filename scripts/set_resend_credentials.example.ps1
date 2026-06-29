$env:RESEND_API_KEY = "re_xxxxxxxxxxxxxxxxxxxxxxxxx"
$env:RESEND_FROM = "CreditBond AI <reports@your-verified-domain.com>"

Write-Host "RESEND_API_KEY set:" ($env:RESEND_API_KEY -like "re_*")
Write-Host "RESEND_FROM:" $env:RESEND_FROM
