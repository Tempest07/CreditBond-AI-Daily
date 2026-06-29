# DM 数据库接口接入笔记

参考文件已复制到 `references/dm_api`。

## 文件

- `DM PythonAPI数据调用用户手册V2.2(20260618).pdf`
- `dm_quant_api_client-0.2.3-py3-none-any.whl`
- `【字典】EDB指标层级.xlsx`
- `edb_indicator_levels.csv`

## SDK 信息

轮子包名：`dm_quant_api_client`

核心类：`DMQuantApiClient`

默认地址：`https://gapi-ext.innodealing.com`

认证参数：

- `app_key`
- `app_secret`

也支持环境变量：

- `INNO_APP_KEY`
- `INNO_APP_SECRET`
- `INNO_SM4_KEY`

依赖包括：

- `requests`
- `pandas`
- `gmssl`

当前本地环境已经可以导入 DM SDK，`gmssl` 也已安装。真实凭证只放在本地环境变量或 `scripts/set_dm_credentials.local.ps1`，不要写入代码和文档。

## 凭证设置

不要把 AppKey 和 AppSecret 写进代码。建议只在当前 PowerShell 会话里临时设置：

```powershell
$env:INNO_APP_KEY='你的AppKey'
$env:INNO_APP_SECRET='你的AppSecret'
```

查看提示：

```powershell
python -m creditbond_ai.cli dm-credentials-help
```

## EDB 层级搜索

本地搜索 EDB 层级：

```powershell
python -m creditbond_ai.cli dm-search-levels --keyword PMI --out reports/dm_search_pmi.csv
```

## EDB 接口

获取宏观 EDB 指标编号：

```text
/dm-quant-func-service/api/v1/edb/data-info/code
```

关键入参：

- `edbLevelIdList`: 指标层级编号列表
- `fieldNames`: 可选字段列表
- `offset`: 翻页游标

关键出参：

- `indicator_id`
- `indicator_name`
- `basic_indicator_unit`
- `data_status`
- `statistical_frequency`
- `region_name`
- `Max_Offset`

获取宏观 EDB 指标数值：

```text
/dm-quant-func-service/api/v1/edb/data-info/data
```

关键入参：

- `indicatorId`
- `startDate`
- `endDate`
- `fieldNames`

关键出参：

- `indicator_id`
- `indicator_name`
- `data_date`
- `data_value`
- `basic_indicator_unit`
- `publish_time`
- `data_source`

## 和我们现有框架的接法

后续主线是 DM API 全接管，不再依赖 Wind 手工导出。

1. 用 `dm-search-levels` 找到指标层级编号。
2. 调用指标编号接口，拿到具体 `indicator_id`。
3. 按配置表批量调用指标数值接口，保存 DM 原始数据。
4. 用 `dm-edb-to-wide` 转成按可获得日对齐的防穿越宽表。
5. 生成模型就绪表、增强特征和数据字典。
6. 进入 `auto-research` 或最新预测。

重点：EDB 数值接口返回 `publish_time`，后面要优先用它控制数据可获得时间，避免宏观数据穿越。

## 已接入的命令

按 EDB 层级编号拉具体指标 ID：

```powershell
python -m creditbond_ai.cli dm-fetch-codes --level-ids CN006008008 --out data/dm_raw/pmi_codes.csv
```

按层级配置批量拉具体指标 ID：

```powershell
python -m creditbond_ai.cli dm-fetch-codes-config --config configs/dm_edb_level_candidates_minimal.csv --out data/dm_raw/edb_codes_minimal.csv
```

按指标 ID 拉历史数值：

```powershell
python -m creditbond_ai.cli dm-fetch-data --indicator-id M00100066100000 --start-date 2020-01-01 --end-date 2026-06-23 --frequency 月 --out data/dm_raw/pmi_data.csv
```

按配置表批量拉历史数值：

```powershell
python -m creditbond_ai.cli dm-fetch-data-batch --config configs/dm_edb_minimal_template.csv --out-dir data/dm_raw/edb_minimal
```

配置表字段：

- `indicator_id`: DM EDB 指标 ID
- `alias`: 本地别名，可选
- `frequency`: 频率，用于自动分段请求
- `start_date`: 开始日期
- `end_date`: 结束日期

把 EDB 原始长表转换成防穿越宽表：

```powershell
python -m creditbond_ai.cli dm-edb-to-wide --input data/dm_raw/edb_minimal --out data/dm_processed/edb_wide.csv --dictionary-out data/dm_processed/edb_dictionary.csv --model-ready-out data/dm_processed/edb_model_ready.csv
```

这一步会优先使用 `publish_time` 作为可用日。也就是说，宏观数据只有在公告日之后才进入模型。

说明：

- 日度或不定期指标会自动按小于 1 年拆分请求。
- 周度、旬度指标会自动按小于 5 年拆分请求。
- 月度、季度、半年度、年度指标会自动按小于 10 年拆分请求。
- 抓到的 DM 原始数据后续优先进入 `dm-edb-to-wide`，不再走 Wind 手工导出合并逻辑。
