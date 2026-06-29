# 复制本文件为 set_dm_credentials.local.ps1 后填写真实值。
# 不要把真实 AppKey/AppSecret 发给别人，也不要提交到仓库。

$env:INNO_APP_KEY = "在这里填写你的AppKey"
$env:INNO_APP_SECRET = "在这里填写你的AppSecret"

python -m creditbond_ai.cli dm-check
