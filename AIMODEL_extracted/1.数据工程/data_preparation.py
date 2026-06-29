# data_preparation.py：动态读取原始数据的主流程（根据文件夹内容自动处理）
import os
from data_utils import (
    read_financial_data,
    align_to_daily,
    remove_outliers,
    merge_all_data,
    credit_business_rule
)

"""
PLEASE PAY ATTENTION!!
请注意！！
所有的csv文件内部均需遵循一定的格式！一般来说，一个CSV文件请仅包含一种factor（如中国GDP），并将日期置于A列，factor置于B列。
"""

if __name__ == "__main__":
    # --------------------------
    # 1. 配置路径（只改这里！）
    # --------------------------
    raw_data_path = "raw_data/"       # 原始数据文件夹
    output_path = "processed_data/credit_bond_final_data.csv"  # 输出路径

    # --------------------------
    # 2. 自动扫描并读取所有CSV文件
    # --------------------------
    # 2.1 列出文件夹中所有CSV文件
    csv_files = [f for f in os.listdir(raw_data_path) if f.endswith(".csv")]
    if not csv_files:
        print(f"❌ 原始数据文件夹'{raw_data_path}'中没有CSV文件，请检查路径")
        exit()  # 没有文件则退出程序

    # 2.2 定义“文件名关键词→数据类型”的映射（核心：通过文件名判断数据类型）
    # 格式：{关键词: (数值列名, 插值方法, 异常值处理规则)}
    # 可根据你的实际文件名扩展（比如加"pmi"、"export"等）
    data_type_mapping = {
        # 宏观经济数据（季度/月度）
        "gdp": ("gdp", "linear", None),           # 文件名含"gdp"→列名"gdp"，线性插值
        "cpi": ("cpi", "ffill", None),            # 文件名含"cpi"→列名"cpi"，前向填充
        "nonfarm": ("nonfarm", "ffill", None),    # 文件名含"nonfarm"→列名"nonfarm"
        
        # 收益率数据（日频，需业务逻辑）
        "treasury_1y": ("yield_1y", "linear", None),               # 1年期国债
        "credit_aaa_1y": ("yield_aaa_1y", "linear", "credit_aaa"),  # AAA级信用债
        "credit_aa_1y": ("yield_aa_1y", "linear", "credit_aa"),     # AA级信用债
        
        # 其他数据
        "volume": ("trade_count", "linear", None),  # 成交笔数（文件名含"volume"）
        "usd_index": ("usd_index", "linear", None)  # 美元指数
    }

    # 2.3 动态读取所有文件并分类存储
    raw_data = {}  # 字典：{数据类型: 原始DataFrame}
    for file in csv_files:
        file_path = os.path.join(raw_data_path, file)
        # 匹配数据类型（通过文件名是否包含关键词）
        matched_type = None
        for data_type, (value_col, _, _) in data_type_mapping.items():
            if data_type in file.lower():  # 不区分大小写（如GDP.csv和gdp.csv都能匹配）
                matched_type = data_type
                break
        
        if matched_type:
            # 读取数据并存储到字典
            df = read_financial_data(file_path)
            if df is not None:
                raw_data[matched_type] = {
                    "df": df,
                    "value_col": data_type_mapping[matched_type][0]  # 数值列名
                }
        else:
            print(f"⚠️ 未识别的文件：{file}（请检查文件名是否符合规则）")

    # 检查是否有必要数据（至少需要国债和信用债数据，否则无法生成标签）
    required_types = ["treasury_1y", "credit_aaa_1y"]
    if not all(t in raw_data for t in required_types):
        print(f"❌ 缺少必要数据：{[t for t in required_types if t not in raw_data]}")
        exit()

    # --------------------------
    # 3. 批量频率对齐（非日频→日频）
    # --------------------------
    daily_data = {}  # 字典：{数据类型: 日频DataFrame}
    for data_type, info in raw_data.items():
        _, align_method, _ = data_type_mapping[data_type]
        daily_df = align_to_daily(
            df=info["df"],
            value_col=info["value_col"],
            method=align_method
        )
        if daily_df is not None:
            daily_data[data_type] = daily_df

    # --------------------------
    # 4. 批量异常值处理
    # --------------------------
    clean_data = {}  # 字典：{数据类型: 清洗后DataFrame}
    # 先处理国债数据（信用债需要用）
    treasury_type = "treasury_1y"
    treasury_value_col = raw_data[treasury_type]["value_col"]
    clean_data[treasury_type] = remove_outliers(
        daily_data[treasury_type],
        value_col=treasury_value_col
    )

    # 处理其他数据
    for data_type in daily_data:
        if data_type == treasury_type:
            continue  # 已处理国债
        
        info = raw_data[data_type]
        _, _, outlier_rule = data_type_mapping[data_type]
        # 信用债需要额外业务逻辑
        business_rule = None
        if outlier_rule == "credit_aaa":
            business_rule = lambda x: credit_business_rule(
                x, 
                clean_data[treasury_type], 
                info["value_col"], 
                treasury_value_col
            )
        elif outlier_rule == "credit_aa":
            business_rule = lambda x: credit_business_rule(
                x, 
                clean_data[treasury_type], 
                info["value_col"], 
                treasury_value_col
            )
        
        # 处理异常值
        clean_df = remove_outliers(
            daily_data[data_type],
            value_col=info["value_col"],
            business_rule=business_rule
        )
        if clean_df is not None:
            clean_data[data_type] = clean_df

    # --------------------------
    # 5. 合并数据并保存（动态适配所有数据）
    # --------------------------
    # 定义列名映射（让输出列名更友好）
    column_rename_mapping = {
        "yield_1y": "treasury_1y_yield",
        "yield_aaa_1y": "credit_aaa_1y_yield",
        "yield_aa_1y": "credit_aa_1y_yield",
        "trade_count": "credit_trade_count"
    }

    # 准备合并的数据列表
    data_to_merge = []
    for data_type, df in clean_data.items():
        value_col = raw_data[data_type]["value_col"]
        # 重命名列（如果有映射）
        rename_col = column_rename_mapping.get(value_col, value_col)
        data_to_merge.append(
            df[["date", value_col]].rename(columns={value_col: rename_col})
        )

    # 合并并保存
    final_data = merge_all_data(data_to_merge)
    if final_data is not None:
        final_data.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\n🎉 数据准备完成！最终数据已保存到：{output_path}")
        print(f"📊 最终数据信息：")
        print(f"   - 时间范围：{final_data['date'].min()} ~ {final_data['date'].max()}")
        print(f"   - 数据规模：{len(final_data)}行 × {len(final_data.columns)}列")
        print(f"   - 列名：{final_data.columns.tolist()}")
        print(f"   - 缺失值：\n{final_data.isnull().sum()}")
    else:
        print("\n❌ 数据准备失败，请检查前面步骤的错误提示")
