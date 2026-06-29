# 数据清洗流水线

以后任何新数据先走这条流水线，再进入神经网络。

后续主线数据源统一为 DM API。CSV 清洗仍保留，但定位是历史迁移、临时补数和调试，不再作为日常数据入口。

## 流程

1. 读取 DM API 原始数据或历史 CSV，自动识别编码。
2. 识别日期列。DM 数据优先使用 `publish_time` 作为可获得日，历史 CSV 则识别普通日期列。
3. 删除频率、单位、指标 ID、来源等非日期行；DM 元数据进入数据字典。
4. 数值列转换，记录无法转换的异常值。
5. 按日期排序并去重，同日多条保留最后一条。
6. 对齐到目标频率，默认交易日频率。
7. 只允许向前填充，不允许向后填充。
8. 删除开头仍有缺失的区间，防止未来数据倒灌。
9. 生成数据字典。
10. 生成质量报告。
11. 可选生成增强特征，包括过去变动、滚动波动、信用利差、期限利差。

## 命令

清洗单个历史 CSV：

```powershell
python -m creditbond_ai.cli clean-data --input 测试数据集.csv --out-dir data/cleaned --name 测试数据集
```

输出文件：

- `测试数据集_清洗后.csv`
- `测试数据集_增强特征.csv`
- `测试数据集_数据字典.csv`
- `测试数据集_增强特征字典.csv`
- `测试数据集_清洗报告.json`
- `测试数据集_数据字典校验.json`

批量清洗一个文件夹里的多个历史 CSV，并合并成模型总表：

```powershell
python -m creditbond_ai.cli clean-batch --input-dir raw_data --pattern "*.csv" --out-dir data/cleaned_batch --name 信用债总库
```

批量输出会包含：

- `per_file/`: 每个原始 CSV 的单独清洗结果和报告。
- `combined/信用债总库_历史合并.csv`: 按日期外连接后的历史总表，允许某些指标在起点前为空。
- `combined/信用债总库_模型就绪.csv`: 删除仍有缺失的开头区间，可以直接喂给神经网络。
- `combined/信用债总库_增强特征.csv`: 模型就绪表加上过去变化、波动、信用利差、期限利差等衍生特征。
- `combined/信用债总库_批量清洗报告.json`: 批量审计报告。

## 原则

清洗阶段宁可少用数据，也不能让模型看到历史时点不可获得的数据。

日常 DM 数据建议优先使用：

```powershell
python -m creditbond_ai.cli dm-fetch-data-batch --config configs/dm_edb_minimal_real.csv --out-dir data/dm_raw/edb_minimal_real
python -m creditbond_ai.cli dm-edb-to-wide --input data/dm_raw/edb_minimal_real --out data/dm_processed/edb_minimal_wide.csv --dictionary-out data/dm_processed/edb_minimal_dictionary.csv --model-ready-out data/dm_processed/edb_minimal_model_ready.csv
```
