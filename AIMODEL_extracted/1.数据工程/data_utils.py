# data_utils.py：所有数据处理的工具函数（只定义，不执行）
import pandas as pd
import numpy as np

def read_financial_data(file_path):
    """读取金融CSV数据，自动解析日期、排序、删除无效日期"""
    try:
        df = pd.read_csv(
            file_path,
            parse_dates=["date"],
            date_parser=lambda x: pd.to_datetime(x, errors="coerce")
        )
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        print(f"✅ 成功读取：{file_path}，共{len(df)}行数据")
        return df
    except Exception as e:
        print(f"❌ 读取{file_path}失败：{str(e)}")
        return None


def align_to_daily(df, value_col, method="linear", start_date="2018-01-01", end_date="2024-05-31"):
    """将非日频数据对齐到标准日频"""
    if df is None or value_col not in df.columns:
        print(f"❌ 对齐失败：数据为空或无{value_col}列")
        return None
    
    standard_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    standard_df = pd.DataFrame({"date": standard_dates})
    daily_df = pd.merge(standard_df, df[["date", value_col]], on="date", how="left")
    
    if method == "linear":
        daily_df[value_col] = daily_df[value_col].interpolate(method="linear")
    elif method == "ffill":
        daily_df[value_col] = daily_df[value_col].fillna(method="ffill")
    
    daily_df = daily_df.dropna(subset=[value_col])
    print(f"✅ 成功对齐{value_col}：日频数据共{len(daily_df)}行，方法={method}")
    return daily_df


def remove_outliers(df, value_col, business_rule=None):
    """处理异常值（3σ原则+可选业务逻辑）"""
    if df is None or value_col not in df.columns:
        print(f"❌ 异常值处理失败：数据为空或无{value_col}列")
        return None
    
    mean = df[value_col].mean()
    std = df[value_col].std()
    before_count = len(df)
    df_clean = df[(df[value_col] >= mean - 3 * std) & (df[value_col] <= mean + 3 * std)]
    outlier_count = before_count - len(df_clean)
    
    if business_rule is not None:
        df_clean = business_rule(df_clean)
        business_outlier_count = before_count - len(df_clean) - outlier_count
        outlier_count += business_outlier_count
    
    print(f"✅ 异常值处理{value_col}：删除{outlier_count}个异常值，剩余{len(df_clean)}行")
    return df_clean


def merge_all_data(data_list, on="date"):
    """合并所有日频数据（按日期）"""
    if len(data_list) == 0:
        print(f"❌ 合并失败：数据列表为空")
        return None
    
    final_data = data_list[0]
    for data in data_list[1:]:
        if data is not None:
            final_data = pd.merge(final_data, data, on=on, how="inner")
    
    print(f"✅ 所有数据合并完成：共{len(final_data)}行，{len(final_data.columns)}列")
    return final_data


def credit_business_rule(df, treasury_df, credit_yield_col, treasury_yield_col):
    """信用债业务逻辑：信用债收益率 ≥ 国债收益率"""
    merged = pd.merge(df, treasury_df[["date", treasury_yield_col]], on="date", how="inner")
    merged_clean = merged[merged[credit_yield_col] >= merged[treasury_yield_col]]
    return merged_clean[df.columns]  # 只返回信用债原始列