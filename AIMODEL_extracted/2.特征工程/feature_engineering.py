# feature_engineering.py：原始数据直入LSTM的特征工程主流程
# 流程：加载清洗后原始数据 → 生成标签 → 筛选原始特征 → 时间拆分 → 标准化 → 生成LSTM窗口 → 保存结果
import pandas as pd
import os
import numpy as np
from feature_utils import (
    create_labels,
    split_data_by_time,
    standardize_features,
    create_lstm_windows
)

if __name__ == "__main__":
    # --------------------------
    # 1. 配置参数（用户仅需修改这里！）
    # --------------------------
    INPUT_PATH = "processed_data/credit_bond_final_data.csv"  # 数据准备输出的清洗后原始数据
    OUTPUT_DIR = "feature_data/"                              # 特征工程结果输出文件夹
    TARGET_YIELD_COL = "credit_aaa_1y_yield"                 # 目标预测的信用债收益率列（原始列名）
    PREDICT_DAYS = 5                                          # 预测未来N天的走势（如5天）
    WINDOW_SIZE = 30                                          # LSTM输入窗口：用过去N天原始数据（如30天）
    TRAIN_END = "2021-12-31"                                  # 训练集结束日期（按你的数据调整）
    VAL_END = "2022-12-31"                                    # 验证集结束日期

    # 创建输出文件夹（不存在则自动新建）
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------
    # 2. 加载数据准备阶段输出的“清洗后原始数据”
    # --------------------------
    try:
        # 读取数据，确保日期格式正确
        df_clean = pd.read_csv(INPUT_PATH, parse_dates=["date"])
        df_clean = df_clean.sort_values("date").reset_index(drop=True)  # 按时间排序（时序数据必须！）
        print(f"✅ 成功加载清洗后原始数据：")
        print(f"   - 数据时间范围：{df_clean['date'].min()} ~ {df_clean['date'].max()}")
        print(f"   - 原始指标列（{len(df_clean.columns)-1}个）：{[col for col in df_clean.columns if col != 'date']}")
    except Exception as e:
        print(f"❌ 加载清洗后数据失败：{str(e)}")
        print(f"   请检查路径'{INPUT_PATH}'是否正确，或数据准备阶段是否正常执行")
        exit()  # 数据加载失败则终止程序

    # --------------------------
    # 3. 生成预测标签（看多/看空/震荡）
    # --------------------------
    df_labeled, theta = create_labels(
        df=df_clean,
        target_yield_col=TARGET_YIELD_COL,
        predict_days=PREDICT_DAYS,
        theta_quantile=0.8  # 用历史波动80%分位数作为阈值θ
    )

    # 打印标签分布（验证是否存在类别不平衡）
    label_counts = df_labeled["label"].value_counts().sort_index()
    print(f"\n✅ 标签分布（共{len(df_labeled)}个样本）：")
    print(f"   - 看空（0）：{label_counts.get(0, 0)}个（{label_counts.get(0, 0)/len(df_labeled)*100:.1f}%）")
    print(f"   - 看多（1）：{label_counts.get(1, 0)}个（{label_counts.get(1, 0)/len(df_labeled)*100:.1f}%）")
    print(f"   - 震荡（2）：{label_counts.get(2, 0)}个（{label_counts.get(2, 0)/len(df_labeled)*100:.1f}%）")

    # 检查标签平衡：若震荡占比>80%，提示调整theta分位数
    if label_counts.get(2, 0)/len(df_labeled) > 0.8:
        print(f"⚠️ 警告：震荡类标签占比过高（>80%），可能导致模型偏向预测震荡！")
        print(f"   建议：将theta_quantile从0.8调低至0.7或0.6，增加看多/看空标签数量")

    # --------------------------
    # 4. 筛选“原始特征列”（排除非特征列）
    # --------------------------
    # 非特征列：date（时间）、future_yield_change（中间变量）、目标收益率列（仅用于生成标签，不做特征）
    exclude_cols = ["date", "future_yield_change", TARGET_YIELD_COL]
    feature_cols = [col for col in df_labeled.columns if col not in exclude_cols]

    # 验证特征列（确保无遗漏/错误）
    print(f"\n✅ 最终原始特征列（{len(feature_cols)}个）：")
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i:2d}. {col}")

    # --------------------------
    # 5. 按时间拆分训练/验证/测试集
    # --------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_time(
        df=df_labeled,
        feature_cols=feature_cols,
        train_end=TRAIN_END,
        val_end=VAL_END
    )

    # --------------------------
    # 6. 特征标准化（LSTM必须步骤）
    # --------------------------
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")  # 标准化器保存路径
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_features(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        save_scaler_path=scaler_path
    )

    # --------------------------
    # 7. 生成LSTM时序窗口（转换为3维输入格式）
    # --------------------------
    # 训练集窗口
    X_train_lstm, y_train_lstm = create_lstm_windows(
        X=X_train_scaled,
        y=y_train,
        window_size=WINDOW_SIZE
    )
    # 验证集窗口
    X_val_lstm, y_val_lstm = create_lstm_windows(
        X=X_val_scaled,
        y=y_val,
        window_size=WINDOW_SIZE
    )
    # 测试集窗口
    X_test_lstm, y_test_lstm = create_lstm_windows(
        X=X_test_scaled,
        y=y_test,
        window_size=WINDOW_SIZE
    )

    # --------------------------
    # 8. 保存所有结果（供后续LSTM训练使用）
    # --------------------------
    # 1. 保存LSTM输入数据（numpy格式，加载速度快）
    np.save(os.path.join(OUTPUT_DIR, "X_train_lstm.npy"), X_train_lstm)
    np.save(os.path.join(OUTPUT_DIR, "y_train_lstm.npy"), y_train_lstm)
    np.save(os.path.join(OUTPUT_DIR, "X_val_lstm.npy"), X_val_lstm)
    np.save(os.path.join(OUTPUT_DIR, "y_val_lstm.npy"), y_val_lstm)
    np.save(os.path.join(OUTPUT_DIR, "X_test_lstm.npy"), X_test_lstm)
    np.save(os.path.join(OUTPUT_DIR, "y_test_lstm.npy"), y_test_lstm)

    # 2. 保存特征列名（后续预测新数据时需用相同特征）
    with open(os.path.join(OUTPUT_DIR, "feature_cols.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feature_cols))

    # 3. 保存关键参数（方便后续追溯和复用）
    with open(os.path.join(OUTPUT_DIR, "params.txt"), "w", encoding="utf-8") as f:
        f.write(f"# 特征工程关键参数\n")
        f.write(f"target_yield_col={TARGET_YIELD_COL}\n")
        f.write(f"predict_days={PREDICT_DAYS}\n")
        f.write(f"window_size={WINDOW_SIZE}\n")
        f.write(f"theta={theta:.6f}\n")
        f.write(f"train_end={TRAIN_END}\n")
        f.write(f"val_end={VAL_END}\n")

    # --------------------------
    # 9. 输出最终总结
    # --------------------------
    print(f"\n🎉 特征工程全部完成！结果已保存到：{OUTPUT_DIR}")
    print(f"\n📁 输出文件清单：")
    print(f"   1. X_train_lstm.npy → 训练集LSTM窗口特征（3维）")
    print(f"   2. y_train_lstm.npy → 训练集标签")
    print(f"   3. X_val_lstm.npy   → 验证集LSTM窗口特征（3维）")
    print(f"   4. y_val_lstm.npy   → 验证集标签")
    print(f"   5. X_test_lstm.npy  → 测试集LSTM窗口特征（3维）")
    print(f"   6. y_test_lstm.npy  → 测试集标签")
    print(f"   7. scaler.pkl       → 特征标准化器（预测新数据需复用）")
    print(f"   8. feature_cols.txt → 原始特征列名（确保预测时特征顺序一致）")
    print(f"   9. params.txt       → 关键参数记录（追溯实验用）")