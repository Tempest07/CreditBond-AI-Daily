# DM 债券收益率曲线函数接入更新

更新日期：2026-06-24

## 一句话结论

信用债和国债曲线不要再只走 EDB 层级指标。DM 文档里的债券收益率曲线函数：

```text
/dm-quant-func-service/api/v1/bond/yield-curve/data
```

可以拉到 2020 年起的日频曲线数据。我们已经把它接入主配置 `configs/dm_master_indicators.csv` 和日更流程。

## 已验证结果

实测 2020-01-01 至 2020-01-31：

- `中债中短期票据收益率曲线(AAA)`：3年、5年、10年可返回，最早交易日为 2020-01-02。
- `中债中短期票据收益率曲线(AAA+)`：20年可返回，最早交易日为 2020-01-02。
- `中债国债收益率曲线`：3年、5年、10年、20年可返回，最早交易日为 2020-01-02。

注意：`中债中短期票据收益率曲线(AAA)` 当前不返回 20 年，所以 20 年信用债仍沿用 `AAA+` 口径。

## 接口参数

关键入参：

- `dataSource`: `18` 为中债，`19` 为中证。
- `curveName`: 曲线名称，例如 `中债中短期票据收益率曲线(AAA)`。
- `curveTermList`: 曲线期限列表，单次不超过 5 个。
- `curveType`: `1` 为到期收益率。
- `startDate` / `endDate`: 单次自然日跨度含首尾不能超过 30 天。

输出关键字段：

- `curveChName`: 曲线名称。
- `curveTerm`: 曲线期限。
- `valuationDate`: 估值日期。
- `yield`: 收益率，单位为 `%`。

## 我们的实现

新增能力：

- `creditbond_ai.dm_api.fetch_bond_yield_curve_data`
- `creditbond_ai.dm_api.curve_data_to_edb_like_raw`
- CLI 命令：`dm-fetch-yield-curve`
- 日更配置支持 `source=curve_func`

当前主配置已切换：

- 国债 1/3/5/7/10/20 年：走 `中债国债收益率曲线`。
- 中短期票据 AAA 1/3/5/7/10 年：走 `中债中短期票据收益率曲线(AAA)`。
- 中短期票据 AAA+ 20 年：走 `中债中短期票据收益率曲线(AAA+)`。
- 宏观月频指标仍走 EDB。

## 防穿越处理

文档说明该数据“次日凌晨更新”。因此转换为模型原始表时：

- `data_date` 使用估值日期。
- `publish_time` 保守设置为估值日期后 1 天。
- 后续宽表仍使用 `merge_asof` 向前持有，只允许向前填充。

这样做会比直接用估值日当天更保守，但能避免未来函数。

## 验证命令

用主日更流程验证 2020 年 6 月：

```powershell
.\scripts\set_dm_credentials.local.ps1
python -m creditbond_ai.cli daily-dm-update `
  --config configs\dm_master_indicators.csv `
  --out-dir data\dm_yield_curve_func_smoke2 `
  --start-date 2020-06-01 `
  --end-date 2020-06-30 `
  --model-ready-max-missing-ratio 0.5 `
  --timeout 60
```

验证结果：曲线 raw 文件均返回 2020-06-01 至 2020-06-30，共 21 个交易日。

## 2020 起全量刷新与重训

已完成 2020 起全量刷新：

```powershell
.\scripts\run_daily_dm_update.ps1 -OutDir data\dm_daily_master_curve_2020 -StartDate "2020-01-01" -FullRefresh -Device cuda
```

产物：
- 特征文件：`data\dm_daily_master_curve_2020\processed\dm_features_latest.csv`
- 数据区间：2020-03-09 至 2026-06-24
- 特征形状：1643 行、115 列
- 日报：`data\dm_daily_master_curve_2020\reports\daily_dm_update_report.html`

已完成四个期限专属模型重训，模型目录为：
- `models\curve_2020_AAA3Y_h5\01_full_features`
- `models\curve_2020_AAA5Y_h5\01_full_features`
- `models\curve_2020_AAA10Y_h5\01_full_features`
- `models\curve_2020_AAAp20Y_h5\01_full_features`

以后补数据后，可以用下面命令一键重训四个期限：

```powershell
.\scripts\train_curve_2020_models.ps1
```

也可以只训练某个期限，例如：

```powershell
.\scripts\train_curve_2020_models.ps1 --only AAA10Y
```

日常刷新脚本 `scripts\run_daily_dm_update.ps1` 已切换为使用上述 2020 曲线模型。
