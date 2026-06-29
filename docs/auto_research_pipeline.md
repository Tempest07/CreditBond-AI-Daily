# 自动研究流水线说明

这个流程把“训练模型、评估指标重要性、自动筛选特征、用精选特征重训、生成报告”合成一个命令。

它的目的不是一次就给出最终投资结论，而是让每一批新数据都能被同一套流程审计和测试，逐步判断哪些指标真的对信用债走势有帮助。

## 一条命令做什么

`auto-research` 会依次完成：

1. 用当前全部可用特征训练多种神经网络模型。
2. 选出第一轮表现最好的模型。
3. 对这个模型做特征重要性测试：逐个打乱指标，看模型表现会不会变差。
4. 根据重要性筛出第一批核心特征。
5. 用第一批核心特征重新训练模型。
6. 再对精选特征模型做一次重要性测试。
7. 只保留第二轮仍然有正贡献的特征，再训练一版更精简模型。
8. 输出对比表和 Markdown 报告。

## 当前短跑示例

这条命令用于验证流程能跑通，训练轮数较少，不代表正式模型：

```powershell
python -m creditbond_ai.cli auto-research `
  --data data\dm_processed\creditbond_plus_edb_minimal_features.csv `
  --target-col "中债中短期票据到期收益率(AAA):3年" `
  --out-dir models\auto_research_AAA3Y_h5_smoke `
  --horizon 5 `
  --window 60 `
  --epochs 3 `
  --batch-size 128 `
  --hidden-size 64 `
  --layers 2 `
  --models gru,tcn,transformer `
  --importance-repeats 1 `
  --first-top-n 30 `
  --second-top-n 15
```

## 建议的正式版参数

等数据更多后，可以把训练轮数和重要性重复次数调高：

```powershell
python -m creditbond_ai.cli auto-research `
  --data data\dm_processed\creditbond_plus_edb_minimal_features.csv `
  --target-col "中债中短期票据到期收益率(AAA):3年" `
  --out-dir models\auto_research_AAA3Y_h5_formal `
  --horizon 5 `
  --window 60 `
  --epochs 40 `
  --batch-size 128 `
  --hidden-size 128 `
  --layers 2 `
  --models gru,tcn,transformer `
  --importance-repeats 5 `
  --first-top-n 40 `
  --second-top-n 20
```

## 主要输出

在 `--out-dir` 指定的目录里，会生成：

- `01_full_features`：全特征模型训练结果。
- `02_full_model_feature_importance.csv`：第一轮特征重要性。
- `03_selected_top*.txt`：第一轮自动筛出的特征清单。
- `04_selected_features`：第一轮筛选特征重训结果。
- `05_selected_model_feature_importance.csv`：第二轮特征重要性。
- `06_positive_top*.txt`：第二轮正贡献特征清单。
- `07_positive_features`：第二轮精简特征重训结果。
- `auto_research_report.md`：自动汇总报告。

## 怎么读结果

不要只看准确率。信用债决策辅助至少要同时看：

- 准确率：模型整体猜对方向的比例。
- 宏平均 F1：模型对上涨、下跌、震荡三类是否比较均衡。
- 收益代理：如果按模型方向做久期方向，粗略收益如何。
- 活跃信号占比：模型是否经常给方向信号，还是过于保守。

分类指标高，不一定代表决策效果好。后面正式版会继续加入滚动验证、最大回撤、换手率和多期限目标。
