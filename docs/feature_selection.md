# 指标选择与重要性

指标选择分三层。

## 一、资格审查

先判断指标能不能进入模型：

- 是否有真实可获得时间。
- 是否存在未来信息泄露。
- 缺失率是否过高。
- 历史长度是否足够。
- 是否和目标变量存在定义重叠。

这一步不交给模型自由发挥。

## 二、模型重要性

当前实现的是置换重要性：

```powershell
python -m creditbond_ai.cli feature-importance --model-dir models/creditbond_plus_edb_minimal_AAA3Y_h5_smoke/tcn --data data/dm_processed/creditbond_plus_edb_minimal_features.csv --out reports/feature_importance_tcn.csv --repeats 3
```

逻辑：

1. 先记录原模型表现。
2. 每次打乱一个指标。
3. 重新评估模型。
4. 表现下降越多，该指标越重要。

## 三、滚动稳定性

后续要在滚动回测中重复计算重要性。只有长期稳定重要的指标，才应该成为核心因子。
