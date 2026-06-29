# feature_utils.py：原始数据直入LSTM的特征工程工具函数
# 核心：仅保留标签生成、数据拆分、标准化、时序窗口（无任何人工特征构造）
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import quantile_score


def create_labels(df, target_yield_col="credit_aaa_1y_yield", predict_days=5, theta_quantile=0.8):
    """
    核心1：生成LSTM的预测标签（看多/看空/震荡）
    :param df: 清洗后的原始数据（含date、信用债收益率、其他原始指标）
    :param target_yield_col: 目标信用债收益率列（如AAA级1年期）
    :param predict_days: 预测未来N天的走势
    :param theta_quantile: 阈值θ的分位数（用历史波动定阈值，避免主观）
    :return: 含标签的DataFrame、阈值θ
    """
    df_label = df.copy()
    
    # 1. 计算未来N天的收益率变化（Yx - Y0：X日后收益率 - 当前收益率）
    df_label["future_yield_change"] = df_label[target_yield_col].shift(-predict_days) - df_label[target_yield_col]
    
    # 2. 用历史波动的分位数确定阈值θ（避免手动设值）
    historical_changes = df_label["future_yield_change"].dropna().abs()  # 历史波动绝对值
    theta = quantile_score(historical_changes, historical_changes, quantile=theta_quantile)
    print(f"✅ 标签阈值θ（{theta_quantile*100}%分位数）：{theta:.4f}%")
    
    # 3. 定义标签：0=看空（收益率涨）、1=看多（收益率跌）、2=震荡
    def assign_label(change):
        if change > theta:       # 收益率上涨超θ→债券价格跌→看空
            return 0
        elif change < -theta:    # 收益率下跌超θ→债券价格涨→看多
            return 1
        else:                    # 波动在θ内→震荡
            return 2
    
    df_label["label"] = df_label["future_yield_change"].apply(assign_label)
    df_label = df_label.dropna(subset=["label"])  # 删除最后N行（无未来数据，无标签）
    return df_label, theta


def split_data_by_time(df, feature_cols, label_col="label", train_end="2021-12-31", val_end="2022-12-31"):
    """
    核心2：按时间拆分训练/验证/测试集（时序数据严禁随机拆分，防数据泄露）
    :param df: 含标签的原始数据
    :param feature_cols: 原始特征列名列表（无人工特征）
    :param train_end/val_end: 训练/验证集结束日期
    :return: 拆分后的特征矩阵（X）和标签向量（y）
    """
    # 确保日期格式正确
    df["date"] = pd.to_datetime(df["date"])
    
    # 按时间筛选掩码
    train_mask = df["date"] <= train_end
    val_mask = (df["date"] > train_end) & (df["date"] <= val_end)
    test_mask = df["date"] > val_end
    
    # 拆分特征（原始指标）和标签
    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, label_col].values
    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, label_col].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, label_col].values
    
    # 打印拆分结果（验证数据量合理性）
    print(f"✅ 数据拆分完成：")
    print(f"   - 训练集：{len(X_train)}样本 × {len(feature_cols)}原始特征")
    print(f"   - 验证集：{len(X_val)}样本")
    print(f"   - 测试集：{len(X_test)}样本")
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_features(X_train, X_val, X_test, save_scaler_path=None):
    """
    核心3：特征标准化（LSTM对数据量级敏感，必须用训练集拟合避免泄露）
    :param X_train/X_val/X_test: 拆分后的原始特征矩阵
    :param save_scaler_path: 标准化器保存路径（后续预测需复用）
    :return: 标准化后的特征矩阵 + 标准化器
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 仅训练集拟合（关键！）
    X_val_scaled = scaler.transform(X_val)          # 验证集：用训练集参数转换
    X_test_scaled = scaler.transform(X_test)        # 测试集：用训练集参数转换
    
    # 保存标准化器（后续预测新数据时需用同一 scaler）
    if save_scaler_path:
        import joblib
        joblib.dump(scaler, save_scaler_path)
        print(f"✅ 标准化器已保存到：{save_scaler_path}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def create_lstm_windows(X, y, window_size=30):
    """
    核心4：生成LSTM所需的时序窗口（输入格式：[样本数, 窗口大小, 特征数]）
    :param X: 标准化后的原始特征矩阵（2维：[样本数, 特征数]）
    :param y: 标签向量（1维：[样本数,]）
    :param window_size: 用过去N天的原始数据预测未来走势
    :return: 3维时序窗口特征 + 对应标签
    """
    X_windows = []
    y_windows = []
    
    # 滑动窗口：从第window_size个样本开始，截取前window_size天的特征
    for i in range(window_size, len(X)):
        X_window = X[i - window_size:i, :]  # 过去window_size天的原始特征（2维：[window_size, 特征数]）
        y_window = y[i]                     # 对应第i天的标签（预测未来N天走势）
        X_windows.append(X_window)
        y_windows.append(y_window)
    
    # 转为numpy数组（LSTM要求输入为numpy格式）
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    
    # 打印窗口结果（验证维度正确性）
    print(f"✅ 时序窗口生成完成：")
    print(f"   - 窗口大小：{window_size}天")
    print(f"   - 输入形状：{X_windows.shape}（样本数, 窗口大小, 原始特征数）")
    print(f"   - 标签形状：{y_windows.shape}（样本数,）")
    return X_windows, y_windows