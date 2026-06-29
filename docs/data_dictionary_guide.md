# 数据字典指南

数据字典的作用是防止模型在数据层犯错。尤其是信用债和宏观数据，最大风险不是模型不够复杂，而是某些指标在历史时点其实还不可获得。

## 字段说明

- `列名`: 数据 CSV 中实际列名，必须完全一致。
- `指标名称`: 给人看的名称，可以和列名相同。
- `指标代码`: 优先记录 DM `indicator_id`；历史 Wind 数据可记录 Wind 代码。当前部分历史字典列名仍显示为 `Wind代码`，后续会统一迁移为更中性的 `指标代码`。
- `频率`: 日、周、月、季、年。
- `单位`: 例如 `%`、亿元、点。
- `数据来源`: 后续主线应优先填写 DM；历史数据可保留 Wind、中债估值中心、交易所、手工整理等。
- `可获得规则`: 这个指标在什么时间点可用于决策。
- `发布时间`: DM 返回 `publish_time` 时优先记录。
- `发布滞后天数`: 没有可靠 `publish_time` 时，低频宏观指标必须填保守滞后天数，防止提前使用未来数据。
- `填充方法`: 通常是向前填充；月度、季度数据必须先按发布日期滞后。
- `是否入模`: 是否作为模型输入。
- `是否目标候选`: 是否可作为预测目标。
- `指标类别`: 收益率、信用利差、资金面、宏观、风险偏好等。
- `开始日期` / `结束日期`: 当前数据覆盖范围。
- `缺失数` / `缺失率`: 数据质量检查。
- `备注`: 人工说明。

## 命令

从整理后的数据生成数据字典：

```powershell
python -m creditbond_ai.cli init-dictionary --data data/测试数据集_整理后.csv --raw 测试数据集.csv --out data/测试数据集_数据字典.csv
```

DM 数据转宽表时会同步生成字典：

```powershell
python -m creditbond_ai.cli dm-edb-to-wide --input data/dm_raw/edb_minimal_real --out data/dm_processed/edb_minimal_wide.csv --dictionary-out data/dm_processed/edb_minimal_dictionary.csv --model-ready-out data/dm_processed/edb_minimal_model_ready.csv
```

校验数据字典：

```powershell
python -m creditbond_ai.cli validate-dictionary --data data/测试数据集_整理后.csv --dictionary data/测试数据集_数据字典.csv --report reports/测试数据集_数据字典校验.json
```

## 最重要的规则

日频市场数据可以向前填充，不能向后填充。

月度、季度、年度宏观数据不能按“所属期”直接放进模型，必须按真实发布日期进入模型。例如 5 月社融通常不是 5 月每天都知道，而是在 6 月某天发布。模型只能从发布日之后使用它。
