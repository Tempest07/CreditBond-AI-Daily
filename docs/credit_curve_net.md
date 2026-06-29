# CreditCurveNet 自研神经网络线

这个目录是第二条模型线，保留原来的 GRU / TCN / Transformer 日报模型不变。

## 目标

CreditCurveNet 不是单纯把旧模型加大，而是把信用债期限曲线的几个业务特征写进网络结构：

- 多时间尺度：同时看短期、中期、长期窗口里的收益率和因子变化。
- 期限专属：3 年、5 年、10 年、20 年用同一结构，但模型内部有期限嵌入，允许不同期限学出不同规律。
- 市场状态门控：模型会在多个“状态专家”之间自动加权，不强行假设市场永远牛市或永远熊市。
- 特征门控：每次预测都会给每个指标一个权重，后续可以检查模型到底依赖哪些指标。
- 辅助任务：除了预测看空、看多、震荡，也同时预测未来收益率变化幅度，逼模型理解债券价格与收益率方向。

## 训练单个期限

```powershell
python .\scripts\train_credit_curve_net.py `
  --data data\dm_daily_master_curve_2020\processed\dm_features_latest.csv `
  --target-col "中债中短期票据到期收益率(AAA):10年" `
  --out-dir models\credit_curve_net\curve_2020_AAA10Y_h5\v1_custom_credit_curve_net `
  --device cuda `
  --epochs 80 `
  --batch-size 256 `
  --exclude-target-feature
```

## 训练四个期限

```powershell
python .\scripts\train_credit_curve_net_curve_2020.py --device cuda
```

也可以只训练某个期限：

```powershell
python .\scripts\train_credit_curve_net_curve_2020.py --device cuda --only AAA10Y
```

## 读取最新信号

```powershell
python .\scripts\predict_credit_curve_net.py `
  --model-dir models\credit_curve_net\curve_2020_AAA10Y_h5\v1_custom_credit_curve_net `
  --out output\credit_curve_net_latest_AAA10Y.json `
  --device cuda
```

## 重点输出

- `metrics.json`：测试集准确率、F1、方向代理收益、状态权重。
- `test_predictions.csv`：每个测试日期的真实标签、预测标签、概率、预测收益率变化。
- `feature_gate_importance.csv`：模型平均最关注的特征。
- `training_history.png`：训练过程是否稳定。
- `backtest_proxy.png`：方向代理收益曲线。

## 当前定位

这条线先作为研究实验线，不直接替换日报。等它在滚动预测和样本外回测中明显优于旧模型，再接入正式日报。
