# 信用债 AI 日报云端化方案

## 为什么不直接放进 Cloudflare Worker

隔壁 `ALL` 里的网页项目主要是这套结构：

- Cloudflare Pages：展示网页。
- Cloudflare Worker / Pages Functions：做轻量 API、定时触发、发邮件。
- Resend：负责邮件发送，API Key 放在 Secret 中。

这套结构适合“展示”和“发送”，但不适合直接运行我们当前的信用债 AI 主流程。原因是主流程依赖 Python、PyTorch、DM 的 wheel 包、模型权重和本地文件流水线；这些更适合跑在 GitHub Actions、云服务器或你本机，而不是塞进 Worker。

## 推荐架构

1. GitHub Actions 每个交易日北京时间 8 点触发。
2. 脚本先判断今天是不是中国交易日，不是就跳过。
3. 如果是交易日，用前一个交易日作为数据截止日，调用 DM API 拉数。
4. 跑 3年、5年、10年、20年各期限模型。
5. 生成 HTML 日报、PDF 日报、网页发布目录。
6. 通过 Resend 发邮件：正文可直接阅读，同时附带 PDF。
7. 如果配置了 Cloudflare Secret，再把 `output/site` 直接部署到 Cloudflare Pages。

## 已新增的脚本

- `scripts/run_cloud_daily_report.py`：云端友好的总入口。
- `scripts/build_daily_report_pdf.py`：从日报 JSON 生成 PDF。
- `scripts/publish_daily_site.py`：生成 Cloudflare Pages 可发布目录。
- `scripts/daily_report_utils.py`：日报展示、格式化和汇总工具。
- `.github/workflows/daily-creditbond-ai.yml`：GitHub Actions 定时任务。

## 需要在 GitHub Secrets 里配置

必需：

- `INNO_APP_KEY`
- `INNO_APP_SECRET`
- `RESEND_API_KEY`
- `RESEND_FROM`

可选：

- `RESEND_TO`：不填则默认发到 `yuweiqian@cib.com.cn`。
- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

如果要自动上 Cloudflare Pages，需要先创建一个 Pages 项目，建议项目名：

```text
creditbond-ai-daily
```

## 和 ALL 网页入口的衔接

如果希望以后通过 `tempest07.com/creditbond-ai/` 访问，可以在：

```text
C:\Users\weiqi\Documents\ALL\tempest07-home\gateway-worker.js
```

的 `ROUTES` 里追加：

```js
{
  prefix: "/creditbond-ai",
  origin: "https://creditbond-ai-daily.pages.dev",
}
```

这一步属于已有线上入口的生产路由变更，建议等 Pages 项目先部署成功后再改。

## 本地验证命令

```powershell
python scripts\build_daily_report_pdf.py --json-report data\dm_daily_master_curve_2020\reports\daily_dm_update_report.json
python scripts\publish_daily_site.py --json-report data\dm_daily_master_curve_2020\reports\daily_dm_update_report.json --html-report data\dm_daily_master_curve_2020\reports\daily_dm_update_report.html --pdf output\daily_reports\creditbond_ai_daily_2026-06-29.pdf
python scripts\send_daily_report_resend.py --dry-run --json-report data\dm_daily_master_curve_2020\reports\daily_dm_update_report.json --report data\dm_daily_master_curve_2020\reports\daily_dm_update_report.html --pdf output\daily_reports\creditbond_ai_daily_2026-06-29.pdf
```

## 运行口径

自动任务默认使用“前一个交易日”的数据生成日报。比如北京时间 2026-06-30 早上 8 点运行，会以 2026-06-29 为数据截止日。周末和法定节假日会跳过。
