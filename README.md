# CreditBond AI

本项目是你旧版 `AIMODEL.zip` 构想的本地神经网络 v2：用 DM API 自动拉取宏观、利率、信用债和资金面数据，训练本地时序模型，预测未来 X 天信用债收益率方向。

## 它是不是真 AI？

是。当前版本的预测器是本地神经网络：

- 输入是过去 `window` 天的多因子时间序列。
- 模型通过反向传播学习权重，不使用手写的择时规则。
- 输出是 `看空 / 看多 / 震荡` 三类概率。
- 保存的 `model.pt` 是真实神经网络权重。

项目里仍然会有“量化研究”组件，例如标签定义、时间切分、标准化、回测 proxy。这些不是预测规则，而是让神经网络知道训练目标、并验证它有没有用的实验框架。

标签定义沿用旧版想法：

- `看空`: 未来 X 天目标信用债收益率上行超过阈值，债券价格承压。
- `看多`: 未来 X 天目标信用债收益率下行超过阈值，债券价格受益。
- `震荡`: 未来变化落在阈值以内。

阈值默认用历史 `abs(未来收益率变化)` 的 60% 分位数，可用 `--theta-quantile` 调整。

## 快速自检

```powershell
python -m creditbond_ai.cli make-demo --out data/demo_creditbond.csv
python -m creditbond_ai.cli compare --data data/demo_creditbond.csv --target-col credit_aaa_1y_yield --out-dir models/demo_compare --horizon 5 --window 60 --epochs 5 --models lstm,gru,tcn,transformer
python -m creditbond_ai.cli train --data data/demo_creditbond.csv --target-col credit_aaa_1y_yield --model tcn --out-dir models/demo --horizon 5 --window 60 --epochs 5
python -m creditbond_ai.cli predict --model-dir models/demo --data data/demo_creditbond.csv --out reports/demo_prediction.json
```

比较输出在 `models/demo_compare/comparison.csv`。单个模型训练输出在 `models/demo`：

- `model.pt`: PyTorch 模型
- `scaler.joblib`: 训练集拟合的标准化器
- `metrics.json`: 分类指标和方向回测 proxy
- `test_predictions.csv`: 测试集逐日预测
- `backtest_proxy.png`: 简化方向回测曲线

## 可选神经网络

当前可直接比较四类模型：

- `lstm`: 长短记忆网络，稳健基线，适合小样本。
- `gru`: 门控循环网络，参数更少，常常比长短记忆网络更快。
- `tcn`: 时序卷积网络，并行度高，更适合 4090 这类显卡。
- `transformer`: 注意力模型，并行度高，但更吃数据，样本少时更容易过拟合。

建议先运行 `compare`，再选综合表现最稳的模型进入正式研究。

## 主线数据流程：DM API 全接管

后续日常使用不再依赖 Wind 手工导出 CSV。标准流程是：

1. 用 DM API 按配置表批量拉取指标原始数据。
2. 用 `publish_time` 或保守滞后规则生成防穿越宽表。
3. 生成模型就绪表和增强特征。
4. 运行自动研究流水线或最新预测。

已接入的 DM EDB 示例：

```powershell
python -m creditbond_ai.cli dm-fetch-data-batch --config configs/dm_edb_minimal_real.csv --out-dir data/dm_raw/edb_minimal_real
python -m creditbond_ai.cli dm-edb-to-wide --input data/dm_raw/edb_minimal_real --out data/dm_processed/edb_minimal_wide.csv --dictionary-out data/dm_processed/edb_minimal_dictionary.csv --model-ready-out data/dm_processed/edb_minimal_model_ready.csv
```

自动研究流水线示例：

```powershell
python -m creditbond_ai.cli auto-research --data data\dm_processed\creditbond_plus_edb_minimal_features.csv --target-col "中债中短期票据到期收益率(AAA):3年" --out-dir models\auto_research_AAA3Y_h5_formal --horizon 5 --window 60 --epochs 40 --models gru,tcn,transformer --importance-repeats 5 --device cuda
```

每日 DM 更新和预测：

```powershell
.\scripts\run_daily_dm_update.ps1
```

运行后优先打开 `data\dm_daily\reports\daily_dm_update_report.html`，这是面向普通读者的一页式可视化报告。

首次验证可只抓最近一段：

```powershell
.\scripts\run_daily_dm_update.ps1 -StartDate "2025-01-01"
```

说明：`prepare-wide`、`merge-exports` 等 CSV 命令仍保留，用于历史文件导入、临时补数和调试，不作为后续主线。

## 历史 CSV 兼容入口

如果只有历史 CSV，仍可先用清洗工具转成模型表；但新增和日常数据应优先进入 DM API 配置表。

## RTX 4090

代码会自动使用显卡：`--device auto` 会在 `torch.cuda.is_available()` 为真时走显卡。当前环境已切换到 `torch 2.9.0+cu128`，可以识别 RTX 4090；正式训练建议显式传入 `--device cuda`。

## 风控提醒

这不是交易建议生成器，而是一个研究和决策辅助框架。`backtest_proxy` 是基于收益率方向和久期近似的 proxy，不等于真实组合收益；正式使用前需要加入交易成本、久期、评级/期限分层、流动性和人工风控规则。
