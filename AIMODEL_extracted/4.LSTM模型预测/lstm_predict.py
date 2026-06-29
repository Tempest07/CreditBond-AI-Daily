# lstm_predict.py：信用债走势预测（使用训练好的LSTM模型）
import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --------------------------
# 1. 配置参数（用户根据实际情况修改）
# --------------------------
# 模型和工具路径（指向训练阶段保存的资产）
MODEL_DIR = "saved_model/"
MODEL_PATH = os.path.join(MODEL_DIR, "best_lstm_model.keras")  # 用效果最好的模型
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")            # 标准化器（与训练一致）
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.txt")  # 特征列名（与训练一致）

# 新数据路径（输入最新的原始数据）
NEW_DATA_PATH = "new_data/latest_credit_bond_data.csv"  # 用户需准备此文件

# 预测参数（必须与训练阶段一致！）
WINDOW_SIZE = 30  # 用过去30天数据预测（与feature_engineering.py中的window_size一致）
PREDICT_DAYS = 5  # 预测未来5天走势（与训练时的predict_days一致）


# --------------------------
# 2. 加载训练好的模型和工具
# --------------------------
def load_prediction_assets(model_path, scaler_path, feature_cols_path):
    """加载预测所需的模型、标准化器和特征列"""
    try:
        # 1. 加载LSTM模型
        model = load_model(model_path)
        print(f"✅ 成功加载模型：{model_path}")
        
        # 2. 加载标准化器（必须用训练时的scaler，否则预测会偏差）
        scaler = joblib.load(scaler_path)
        print(f"✅ 成功加载标准化器：{scaler_path}")
        
        # 3. 加载特征列名（确保新数据的特征顺序与训练一致）
        with open(feature_cols_path, "r", encoding="utf-8") as f:
            feature_cols = f.read().splitlines()
        print(f"✅ 成功加载特征列（{len(feature_cols)}个）：{feature_cols}")
        
        return model, scaler, feature_cols
    
    except Exception as e:
        print(f"❌ 加载模型/工具失败：{str(e)}")
        print(f"   请检查'{MODEL_DIR}'文件夹是否存在训练好的模型")
        exit()


# --------------------------
# 3. 处理新数据（与训练时的清洗逻辑一致）
# --------------------------
def process_new_data(new_data_path, feature_cols):
    """
    处理新数据：
    1. 读取数据并按时间排序
    2. 确保特征列与训练时一致（顺序+名称）
    3. 处理缺失值（与训练时的逻辑一致：前向填充）
    """
    try:
        # 1. 读取新数据（需包含date列和所有特征列）
        new_df = pd.read_csv(new_data_path, parse_dates=["date"])
        new_df = new_df.sort_values("date").reset_index(drop=True)  # 按时间排序（关键！）
        print(f"\n✅ 成功加载新数据：")
        print(f"   - 数据时间范围：{new_df['date'].min()} ~ {new_df['date'].max()}")
        print(f"   - 数据条数：{len(new_df)}")
        
        # 2. 检查特征列是否完整（必须与训练时一致）
        missing_cols = [col for col in feature_cols if col not in new_df.columns]
        if missing_cols:
            print(f"❌ 新数据缺少必要特征列：{missing_cols}")
            print(f"   请补充这些列后重试（列名必须与训练时一致）")
            exit()
        
        # 3. 提取特征列（确保顺序与训练时一致）
        new_features = new_df[feature_cols].copy()
        
        # 4. 处理缺失值（与训练时一致：用前一天数据填充）
        new_features = new_features.fillna(method="ffill")  # 前向填充
        new_features = new_features.fillna(method="bfill")  # 若前向填充后仍有缺失，用后向填充
        if new_features.isnull().any().any():
            print(f"⚠️ 警告：新数据中仍有缺失值，可能影响预测结果")
        
        # 5. 返回处理后的特征和日期（日期用于输出预测对应的时间）
        return new_features, new_df["date"]
    
    except Exception as e:
        print(f"❌ 处理新数据失败：{str(e)}")
        print(f"   请检查'{new_data_path}'文件格式是否正确（需包含date列和所有特征列）")
        exit()


# --------------------------
# 4. 构建LSTM预测窗口（用最新的WINDOW_SIZE天数据）
# --------------------------
def create_prediction_window(processed_features, scaler, window_size):
    """
    构建预测窗口：
    1. 用训练时的scaler标准化新数据（必须！否则量级不一致）
    2. 取最新的window_size天数据作为预测窗口
    """
    # 1. 标准化新数据（用训练时的scaler，只做transform，不fit）
    scaled_features = scaler.transform(processed_features)
    
    # 2. 检查数据量是否足够（至少需要window_size天数据）
    if len(scaled_features) < window_size:
        print(f"❌ 新数据量不足（需至少{window_size}天，实际有{len(scaled_features)}天）")
        exit()
    
    # 3. 取最新的window_size天数据作为预测窗口
    # 格式：[1, window_size, n_features]（1个样本，window_size天，n个特征）
    latest_window = scaled_features[-window_size:].reshape(1, window_size, -1)
    print(f"\n✅ 成功构建预测窗口：")
    print(f"   - 窗口包含最新{window_size}天数据（截止到{processed_features.index[-1].strftime('%Y-%m-%d')}）")
    print(f"   - 窗口形状：{latest_window.shape}（符合LSTM输入要求）")
    
    return latest_window


# --------------------------
# 5. 预测并解析结果（输出易懂的结论）
# --------------------------
def predict_and_interpret(model, prediction_window, last_date, predict_days):
    """
    预测并解析结果：
    1. 模型输出3类概率（看空/看多/震荡）
    2. 转换为易懂的结论和投资建议
    """
    # 1. 模型预测（输出概率）
    pred_prob = model.predict(prediction_window, verbose=0)[0]  # [看空概率, 看多概率, 震荡概率]
    
    # 2. 确定预测标签（取概率最大的类别）
    pred_label = np.argmax(pred_prob)
    
    # 3. 解析标签含义（与训练时的定义一致）
    label_map = {
        0: "看空",
        1: "看多",
        2: "震荡"
    }
    pred_result = label_map[pred_label]
    
    # 4. 生成投资建议（结合信用债特性）
    advice_map = {
        0: "建议减持或规避信用债（预计收益率将上涨，价格可能下跌）",
        1: "建议增持信用债（预计收益率将下跌，价格可能上涨）",
        2: "建议观望或维持现有持仓（预计收益率波动较小，无明显趋势）"
    }
    investment_advice = advice_map[pred_label]
    
    # 5. 输出完整结论
    print(f"\n=== 信用债走势预测结果 ===")
    print(f"1. 预测基准日期：{last_date.strftime('%Y-%m-%d')}")
    print(f"2. 预测周期：未来{predict_days}天")
    print(f"3. 预测结论：{pred_result}")
    print(f"4. 各类别概率：")
    print(f"   - 看空：{pred_prob[0]:.2%}")
    print(f"   - 看多：{pred_prob[1]:.2%}")
    print(f"   - 震荡：{pred_prob[2]:.2%}")
    print(f"5. 投资建议：{investment_advice}")
    
    return {
        "prediction_date": last_date.strftime('%Y-%m-%d'),
        "predict_days": predict_days,
        "pred_result": pred_result,
        "probabilities": {
            "看空": float(pred_prob[0]),
            "看多": float(pred_prob[1]),
            "震荡": float(pred_prob[2])
        },
        "advice": investment_advice
    }


# --------------------------
# 6. 主流程：串联所有步骤
# --------------------------
def main():
    print("===== 信用债走势LSTM预测工具 =====")
    
    # 步骤1：加载模型和工具
    model, scaler, feature_cols = load_prediction_assets(
        MODEL_PATH, SCALER_PATH, FEATURE_COLS_PATH
    )
    
    # 步骤2：处理新数据
    processed_features, dates = process_new_data(NEW_DATA_PATH, feature_cols)
    last_date = dates.iloc[-1]  # 最新数据的日期
    
    # 步骤3：构建预测窗口
    prediction_window = create_prediction_window(
        processed_features, scaler, WINDOW_SIZE
    )
    
    # 步骤4：预测并输出结果
    result = predict_and_interpret(
        model, prediction_window, last_date, PREDICT_DAYS
    )
    
    # 可选：保存预测结果到文件
    result_df = pd.DataFrame([result])
    result_path = os.path.join(MODEL_DIR, f"prediction_result_{last_date.strftime('%Y%m%d')}.csv")
    result_df.to_csv(result_path, index=False, encoding="utf-8")
    print(f"\n✅ 预测结果已保存到：{result_path}")

if __name__ == "__main__":
    main()