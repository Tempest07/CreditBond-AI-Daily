# wind_data_fetcher.py：从Wind接口自动获取最新数据（宏观+债券指标）
import os
import pandas as pd
import datetime
from WindPy import w  # Wind Python API（需安装并授权）

# --------------------------
# 1. 配置参数（根据需求修改）
# --------------------------
# 输出路径（与预测脚本的新数据路径对应）
OUTPUT_PATH = "new_data/latest_credit_bond_data.csv"
# 需获取的指标（Wind代码→中文名称映射，与训练时的特征列对应）
INDICATORS = {
    # 债券相关指标
    "CBA00101.CS": "treasury_1y_yield",  # 1年期国债收益率
    "CBA00201.CS": "credit_aaa_1y_yield",  # AAA级1年期信用债收益率
    "CBA00202.CS": "credit_aa_1y_yield",   # AA级1年期信用债收益率
    "S0035626": "credit_trade_count",      # 信用债成交笔数
    # 宏观经济指标
    "M0000694": "gdp",  # 中国季度GDP同比增速（%）
    "M0000681": "cpi",  # 中国月度CPI同比（%）
    "US0000614": "nonfarm",  # 美国月度非农就业人数（千人）
    "M0000545": "usd_index"  # 美元指数
}
# 数据时间范围（获取最近90天数据，确保足够构建30天窗口）
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")


# --------------------------
# 2. 连接Wind接口
# --------------------------
def connect_wind():
    """连接Wind终端，返回连接状态"""
    if not w.isconnected():
        print("正在连接Wind终端...")
        wd = w.connect()  # 启动Wind连接（需提前登录Wind终端）
        if wd != 0:
            print("❌ Wind连接失败，请检查Wind终端是否登录或API权限是否开通")
            return False
    print("✅ Wind连接成功")
    return True


# --------------------------
# 3. 从Wind获取单指标数据
# --------------------------
def fetch_single_indicator(wind_code, indicator_name):
    """
    从Wind获取单个指标的时间序列数据
    :param wind_code: 指标的Wind代码（如"CBA00101.CS"）
    :param indicator_name: 自定义列名（与训练时的特征列一致）
    :return: 包含date和指标值的DataFrame
    """
    try:
        # 调用Wind的w.wsd函数（获取时间序列数据）
        # 参数说明：代码、字段、开始日、结束日、其他选项
        data = w.wsd(
            wind_code,
            "close",  # 取收盘价/最新值（根据指标类型自动适配）
            START_DATE,
            END_DATE,
            ""  # 其他参数（如复权方式，此处默认）
        )
        
        # 解析返回结果（Wind返回的数据格式为字典）
        if data.ErrorCode != 0:
            print(f"❌ 获取指标{indicator_name}失败：{data.Data[0]}")
            return None
        
        # 转换为DataFrame（date为索引，值为指标值）
        df = pd.DataFrame(
            data.Data[0],  # 指标数值列表
            index=data.Times,  # 时间列表
            columns=[indicator_name]
        )
        df.index.name = "date"  # 索引命名为date
        df = df.reset_index()   # 将date从索引转为列
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")  # 统一日期格式
        print(f"✅ 成功获取指标：{indicator_name}（{len(df)}条数据）")
        return df
    
    except Exception as e:
        print(f"❌ 获取指标{indicator_name}出错：{str(e)}")
        return None


# --------------------------
# 4. 合并多指标数据（统一日期索引）
# --------------------------
def merge_indicators(indicators_dict):
    """
    合并所有指标数据，按日期对齐（填充缺失值）
    :param indicators_dict: 指标Wind代码→自定义列名的映射
    :return: 合并后的DataFrame（日期+所有指标）
    """
    # 初始化合并结果（从第一个指标开始）
    first_code, first_name = next(iter(indicators_dict.items()))
    merged_df = fetch_single_indicator(first_code, first_name)
    if merged_df is None:
        print("❌ 无法获取首个指标数据，合并失败")
        return None
    
    # 逐个合并其他指标
    for wind_code, indicator_name in list(indicators_dict.items())[1:]:
        # 获取单个指标数据
        indicator_df = fetch_single_indicator(wind_code, indicator_name)
        if indicator_df is None:
            continue  # 跳过获取失败的指标
        
        # 按日期合并（左连接，保留所有日期）
        merged_df = pd.merge(
            merged_df,
            indicator_df,
            on="date",
            how="left"
        )
    
    # 处理缺失值（前向填充，与训练时的清洗逻辑一致）
    merged_df = merged_df.sort_values("date")
    merged_df = merged_df.fillna(method="ffill")  # 前向填充
    merged_df = merged_df.fillna(method="bfill")  # 后向填充（处理开头的缺失）
    
    print(f"\n✅ 所有指标合并完成：")
    print(f"   - 时间范围：{merged_df['date'].min()} ~ {merged_df['date'].max()}")
    print(f"   - 总数据条数：{len(merged_df)}")
    print(f"   - 包含指标：{merged_df.columns.tolist()[1:]}")  # 排除date列
    return merged_df


# --------------------------
# 5. 主流程：获取并保存数据
# --------------------------
def main():
    print("===== Wind数据自动获取工具 =====")
    
    # 步骤1：连接Wind
    if not connect_wind():
        return
    
    # 步骤2：获取并合并所有指标
    merged_df = merge_indicators(INDICATORS)
    if merged_df is None:
        return
    
    # 步骤3：保存数据（与预测脚本的新数据路径对应）
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\n🎉 数据已保存到：{OUTPUT_PATH}")
    
    # 断开Wind连接
    w.disconnect()
    print("✅ 已断开Wind连接")

if __name__ == "__main__":
    main()
