# DM API 全接管后的下一步

后续不会再有“合并 Wind 手工导出信用债数据”这一步。

新的日常逻辑是：

1. 从 DM API 拉取全部指标，包括信用债收益率、信用利差、利率债、资金面和宏观数据。
2. 把 DM 原始数据保存到 `data/dm_raw`，保留可复现的原始记录。
3. 用 `publish_time` 或保守滞后规则生成防穿越宽表。
4. 生成模型就绪表和增强特征。
5. 调用主力模型和参照模型，输出最新预测。
6. 如果数据覆盖有变化，再运行 `auto-research` 重新训练和自动选指标。

## 每日流水线

基础版已经实现为 `daily-dm-update`。它会：

1. 按 `configs/dm_master_indicators.csv` 从 DM API 拉取指标。
2. 增量更新本地原始数据。
3. 生成防穿越宽表、模型就绪表和增强特征。
4. 调用主力模型和参照模型。
5. 输出 HTML、Markdown 和 JSON 报告。

直接运行：

```powershell
.\scripts\run_daily_dm_update.ps1
```

或手动运行：

```powershell
python -m creditbond_ai.cli daily-dm-update `
  --config configs\dm_master_indicators.csv `
  --out-dir data\dm_daily `
  --model-dir models\auto_research_AAA3Y_h5_formal\04_selected_features\transformer `
  --model-dir models\auto_research_AAA3Y_h5_formal\07_positive_features\tcn `
  --model-dir models\dm_only_AAA5Y_h5_auto\01_full_features\transformer `
  --model-dir models\dm_only_AAA10Y_h5_auto\07_positive_features\transformer `
  --model-dir models\dm_only_AAAp20Y_h5_auto\07_positive_features\transformer `
  --device cuda
```

第一次只想拉最近一段用于验证，可以加：

```powershell
.\scripts\run_daily_dm_update.ps1 -StartDate "2025-01-01"
```

如果已经有本地原始数据，只想重建宽表和预测，可以加：

```powershell
.\scripts\run_daily_dm_update.ps1 -SkipFetch
```

## 继续完善项

1. 继续扩充 `configs/dm_master_indicators.csv`，加入资金面、存单、国开、信用利差和更多宏观指标。
2. 用完整 DM 历史数据重新训练模型，逐步替换早期混合数据训练出的模型。
3. 加滚动验证，让模型在多个历史区间反复检验。
4. 继续美化 HTML 报告，加入历史信号曲线、近 20 日收益率变化和模型分歧提醒。
5. 做本地仪表盘，把每日信号、概率、重要性和历史表现放到同一屏。

## 给普通观众看的结果

每日流程会生成：

- `daily_dm_update_report.html`：面向普通读者的一页式报告，优先看这个。
- `daily_dm_update_report.md`：适合复制到笔记或聊天里的文字报告。
- `daily_dm_update_report.json`：给程序继续读取的结构化结果。

HTML 报告现在包含：

- 一句话结论和综合方向。
- 预测日期、数据范围和特征数量。
- 看空、看多、震荡概率条。
- 3年、5年、10年、20年信用债期限走势板块。
- 每个期限的近20日走势图、最新收益率、1日/5日/20日变化。
- 信用利差和近20日利差变化。
- 模型分歧提醒。
- 模型历史测试表现。

注意：当前 DM 中 `中短期票据(AAA)` 到期收益率只找到 15年以内，20年板块暂用 `中短期票据(AAA+)20年` 口径，并在报告中标注。

## 期限专属模型状态

截至 2026-06-23，3年、5年、10年、20年信用债期限模型已经接入预测和回测流程：

- 3年：`models\auto_research_AAA3Y_h5_formal\04_selected_features\transformer`，目标为 `中债中短期票据到期收益率(AAA):3年`。
- 5年：`models\dm_only_AAA5Y_h5_auto\01_full_features\transformer`，目标为 `中债中短期票据到期收益率(AAA):5年`。
- 10年：`models\dm_only_AAA10Y_h5_auto\07_positive_features\transformer`，目标为 `中债中短期票据到期收益率(AAA):10年`。
- 20年：`models\dm_only_AAAp20Y_h5_auto\07_positive_features\transformer`，目标为 `中债中短期票据到期收益率(AAA+):20年`。

其中 5年、10年、20年是 DM-only 短样本第一版。训练特征表覆盖 2025-01-16 至 2026-06-23，回测样本仍偏少，适合作为研究和决策辅助，不适合直接当作自动交易信号。后续如果拿到更长历史或更精确的 AAA 20年口径，应重新训练并替换当前模型。

## 简单仓位规则回测

已经实现 `position-backtest`，用于把模型历史信号翻译成期限仓位回测：

- 3年信号只调整 3年仓位，5年信号只调整 5年仓位，10年信号只调整 10年仓位，20年信号只调整 20年仓位。
- 看多：买入 5% 该期限仓位。
- 看空：卖出 5% 该期限仓位。
- 震荡：不动。
- 单个期限仓位限制在 0% 至 100%。

当前第一版回测命令：

```powershell
python -m creditbond_ai.cli position-backtest `
  --features data\dm_daily_master_training\processed\dm_features_latest.csv `
  --model-dir models\auto_research_AAA3Y_h5_formal\04_selected_features\transformer `
  --model-dir models\dm_only_AAA5Y_h5_auto\01_full_features\transformer `
  --model-dir models\dm_only_AAA10Y_h5_auto\07_positive_features\transformer `
  --model-dir models\dm_only_AAAp20Y_h5_auto\07_positive_features\transformer `
  --out-dir data\backtests\dm_tenor_position_first `
  --initial-position 0.5 `
  --step 0.05
```

输出网页在 `data\backtests\dm_tenor_position_first\position_backtest_report.html`。收益使用“票息 carry + 久期近似价格变化”估算，不是中债财富指数的真实全价收益。

如需查看包含训练期和验证期的全历史回放，使用：

```powershell
python -m creditbond_ai.cli position-backtest `
  --features data\dm_daily_master_training\processed\dm_features_latest.csv `
  --model-dir models\auto_research_AAA3Y_h5_formal\04_selected_features\transformer `
  --model-dir models\dm_only_AAA5Y_h5_auto\01_full_features\transformer `
  --model-dir models\dm_only_AAA10Y_h5_auto\07_positive_features\transformer `
  --model-dir models\dm_only_AAAp20Y_h5_auto\07_positive_features\transformer `
  --out-dir data\backtests\dm_tenor_position_full_history `
  --initial-position 0.5 `
  --step 0.05 `
  --signal-scope all
```

全历史回放输出网页在 `data\backtests\dm_tenor_position_full_history\position_backtest_report.html`。这个版本用于观察策略形态，不作为严格样本外检验。

## CSV 工具的新定位

`prepare-wide`、`merge-exports`、`clean-data`、`clean-batch` 仍保留，但只用于：

- 历史文件迁移。
- 临时补数。
- 外部数据调试。
- DM 暂时没有覆盖的非核心数据。

日常正式数据入口优先使用 DM API。
