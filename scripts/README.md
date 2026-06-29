# 本地脚本

## DM 凭证

把 `set_dm_credentials.example.ps1` 复制成 `set_dm_credentials.local.ps1`，在 local 文件里填写真实 AppKey/AppSecret，然后在 PowerShell 里运行：

```powershell
.\scripts\set_dm_credentials.local.ps1
```

真实凭证只进入当前 PowerShell 会话的环境变量，不写入项目代码。
