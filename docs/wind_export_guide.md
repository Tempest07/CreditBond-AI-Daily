# Wind 手工导出指南

状态：历史兼容方案。

后续主线已经改为 DM API 全接管。本文只用于早期 Wind 手工文件、临时补数或外部 CSV 调试，不再作为日常数据更新流程。

你没有 Wind API 也没关系，第一版按“手工导出 CSV”设计。

## 推荐格式：一张宽表

CSV 第一列是日期，后面每列一个指标：

```csv
date,treasury_1y_yield,credit_aaa_1y_yield,credit_aa_1y_yield,pmi,cpi,usd_index,credit_trade_count
2020-01-02,2.35,3.05,3.48,50.2,4.5,96.7,1200
2020-01-03,2.34,3.04,3.49,50.2,4.5,96.8,1150
```

运行：

```powershell
python -m creditbond_ai.cli prepare-wide --input data/wind_raw.csv --out data/creditbond_prepared.csv
```

## 逐指标导出

如果你更习惯每个指标导出一个 CSV，也可以：

1. 每个文件至少包含日期列和一个数值列。
2. 用 `data/wind_mapping.json` 把文件名映射到模型列名。
3. 运行 `merge-exports`。

```powershell
python -m creditbond_ai.cli merge-exports --input-dir data/wind_exports --out data/creditbond_prepared.csv --mapping data/wind_mapping.json
```

## 建议先导出的指标

- 目标收益率：中债/中证 AAA 信用债 1Y、3Y 或你实际交易的期限。
- 无风险利率：国债 1Y、3Y、10Y。
- 信用层级：AAA、AA+、AA 同期限收益率或利差。
- 资金面：DR007、R007、SHIBOR、MLF/OMO 相关利率。
- 风险偏好：股指、美元指数、人民币汇率、商品指数。
- 宏观：PMI、CPI、PPI、社融、信贷、工业增加值。
- 市场微观：成交量、成交笔数、信用债净融资。

不要一次贪多。先用 10 到 30 个高质量指标跑通，再逐步扩展。
