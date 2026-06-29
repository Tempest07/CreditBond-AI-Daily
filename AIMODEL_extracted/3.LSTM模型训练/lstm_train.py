# lstm_train.py：信用债走势预测LSTM模型（训练+评估+保存）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# --------------------------
# 1. 配置参数（用户仅需改这里！）
# --------------------------
FEATURE_DATA_DIR = "feature_data/"  # 特征工程结果文件夹路径
MODEL_SAVE_DIR = "saved_model/"     # 模型保存文件夹
PREDICT_DAYS = 5                    # 预测未来N天（需和特征工程一致）
WINDOW_SIZE = 30                    # LSTM窗口大小（需和特征工程一致）

# 创建模型保存文件夹
import os
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# --------------------------
# 2. 加载特征工程输出的数据（核心：3维时序窗口）
# --------------------------
def load_lstm_data(data_dir):
    """加载LSTM训练所需的所有数据"""
    try:
        # 1. 加载时序窗口特征和标签（3维输入）
        X_train = np.load(os.path.join(data_dir, "X_train_lstm.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train_lstm.npy"))
        X_val = np.load(os.path.join(data_dir, "X_val_lstm.npy"))
        y_val = np.load(os.path.join(data_dir, "y_val_lstm.npy"))
        X_test = np.load(os.path.join(data_dir, "X_test_lstm.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test_lstm.npy"))
        
        # 2. 加载标准化器（后续预测新数据需复用）
        scaler = joblib.load(os.path.join(data_dir, "scaler.pkl"))
        
        # 3. 加载特征列名（验证特征数）
        with open(os.path.join(data_dir, "feature_cols.txt"), "r") as f:
            feature_cols = f.read().splitlines()
        n_features = len(feature_cols)
        
        # 打印数据信息（验证维度正确）
        print("✅ 数据加载完成！")
        print(f"   - 训练集窗口：{X_train.shape} → (样本数, 窗口大小, 特征数)")
        print(f"   - 测试集窗口：{X_test.shape}")
        print(f"   - 原始特征数：{n_features}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, scaler, n_features
    
    except Exception as e:
        print(f"❌ 数据加载失败：{str(e)}")
        print(f"   请检查'{data_dir}'文件夹是否存在，或特征工程是否正常执行")
        exit()

# 调用函数加载数据
X_train, y_train, X_val, y_val, X_test, y_test, scaler, n_features = load_lstm_data(FEATURE_DATA_DIR)


# --------------------------
# 3. 搭建极简LSTM模型（核心结构：输入→LSTM→Dropout→输出）
# --------------------------
def build_lstm_model(input_shape, n_classes=3):
    """
    搭建LSTM模型：
    - input_shape：(窗口大小, 特征数) → LSTM的输入维度
    - n_classes：3（看多/看空/震荡）
    """
    model = Sequential(name="CreditBond_LSTM")  # 模型命名，方便后续识别
    
    # LSTM层：32个神经元（新手友好值，数据多可调64，数据少可调16）
    model.add(LSTM(
        units=32, 
        activation="tanh",  # LSTM默认激活函数，无需修改
        input_shape=input_shape, 
        name="lstm_layer"
    ))
    
    # Dropout层：随机关闭20%神经元，防过拟合（核心！）
    model.add(Dropout(
        rate=0.2, 
        name="dropout_layer"
    ))
    
    # 输出层：3个神经元（对应3类），激活函数Softmax（输出概率）
    model.add(Dense(
        units=n_classes, 
        activation="softmax", 
        name="output_layer"
    ))
    
    # 配置模型训练参数
    model.compile(
        optimizer="adam",  # 常用优化器，无需手动调参
        loss="sparse_categorical_crossentropy",  # 标签是整数（0/1/2），用此损失
        metrics=["accuracy"]  # 监控准确率（参考用，重点看后续F1）
    )
    
    # 打印模型结构（可视化层信息）
    model.summary()
    return model

# 定义LSTM输入形状：(窗口大小, 特征数)
input_shape = (WINDOW_SIZE, n_features)
# 调用函数搭建模型
model = build_lstm_model(input_shape)


# --------------------------
# 4. 训练模型（加早停机制，防过拟合）
# --------------------------
def train_lstm_model(model, X_train, y_train, X_val, y_val, save_dir):
    """
    训练LSTM模型：
    - 早停机制：验证集损失3轮不下降就停止，避免无效训练
    - 模型 checkpoint：保存训练中效果最好的模型（不是最后一轮）
    """
    # 1. 早停机制（防过拟合核心手段）
    early_stopping = EarlyStopping(
        monitor="val_loss",  # 监控验证集损失
        patience=3,          # 3轮损失不下降就停止
        restore_best_weights=True,  # 恢复到损失最小的轮次权重
        verbose=1
    )
    
    # 2. 模型 checkpoint（保存最优模型）
    checkpoint_path = os.path.join(save_dir, "best_lstm_model.keras")
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,  # 只保存验证集损失最小的模型
        verbose=1
    )
    
    # 3. 开始训练（核心步骤）
    print("\n🚀 开始训练LSTM模型...")
    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=20,          # 最大训练轮次（早停会提前终止）
        batch_size=16,      # 每次喂16个样本（数据多可调32，数据少可调8）
        validation_data=(X_val, y_val),  # 用验证集监控
        callbacks=[early_stopping, model_checkpoint],  # 早停+保存最优模型
        verbose=1  # 显示训练进度条
    )
    
    # 4. 可视化训练过程（损失+准确率曲线）
    plot_training_history(history, save_dir)
    
    return model, history

def plot_training_history(history, save_dir):
    """绘制训练/验证的损失和准确率曲线，判断是否过拟合"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 1. 损失曲线（核心：验证集损失不上升说明没过度拟合）
    ax1.plot(history.history["loss"], label="训练集损失")
    ax1.plot(history.history["val_loss"], label="验证集损失")
    ax1.set_title("LSTM模型损失曲线")
    ax1.set_xlabel("训练轮次")
    ax1.set_ylabel("损失值")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    ax2.plot(history.history["accuracy"], label="训练集准确率")
    ax2.plot(history.history["val_accuracy"], label="验证集准确率")
    ax2.set_title("LSTM模型准确率曲线")
    ax2.set_xlabel("训练轮次")
    ax2.set_ylabel("准确率")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 保存图片
    plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 训练曲线已保存到：{os.path.join(save_dir, 'training_history.png')}")

# 调用函数训练模型
model, history = train_lstm_model(model, X_train, y_train, X_val, y_val, MODEL_SAVE_DIR)


# --------------------------
# 5. 评估模型（重点：金融场景有用的指标）
# --------------------------
def evaluate_lstm_model(model, X_test, y_test, save_dir, predict_days):
    """
    评估模型：
    1. 分类报告：看看多/看空的F1分数（比准确率更有用）
    2. 混淆矩阵：看各类别预测错误情况
    3. 模拟实盘收益：判断模型对投资的实际价值
    """
    # 1. 预测测试集（输出概率→转标签）
    y_pred_prob = model.predict(X_test, verbose=0)  # 预测3类概率（如[0.1, 0.8, 0.1]）
    y_pred = np.argmax(y_pred_prob, axis=1)  # 取概率最大的类作为预测标签（0/1/2）
    
    # 2. 分类报告（重点看F1分数）
    print("\n=== LSTM模型测试集评估报告 ===")
    report = classification_report(
        y_true=y_test,
        y_pred=y_pred,
        target_names=["看空（0）", "看多（1）", "震荡（2）"],  # 对应标签含义
        output_dict=False
    )
    print(report)
    
    # 3. 保存分类报告
    with open(os.path.join(save_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    
    # 4. 模拟实盘收益（核心：判断模型是否有投资价值）
    simulate_trading_return(y_test, y_pred, predict_days, save_dir)
    
    return y_pred

def simulate_trading_return(y_true, y_pred, predict_days, save_dir):
    """
    模拟实盘策略：
    - 当模型预测“看多（1）”时：买入信用债（假设持有predict_days天，收益率下跌→价格涨→盈利）
    - 当模型预测“看空（0）”时：卖出信用债（持有现金，收益率上涨→避免亏损）
    - 当模型预测“震荡（2）”时：不操作（收益为0）
    收益计算：基于“真实标签”对应的收益率变化（看多标签对应收益率下跌→收益为正）
    """
    # 构造“真实收益率变化”（简化：看多标签对应-θ收益，看空对应+θ，震荡0）
    # 注：实际应基于feature_data/params.txt中的theta，这里简化演示
    theta = 0.05  # 示例阈值（实际需从params.txt读取）
    true_return = np.where(
        y_true == 1,  # 真实看多（收益率跌）→ 收益为正
        theta,
        np.where(
            y_true == 0,  # 真实看空（收益率涨）→ 收益为负（若持有债券）
            -theta,
            0  # 震荡→收益0
        )
    )
    
    # 计算策略收益（模型预测正确则赚theta，错误则亏theta，震荡0）
    strategy_return = np.where(
        (y_pred == 1) & (y_true == 1),  # 预测看多且正确→赚theta
        theta,
        np.where(
            (y_pred == 0) & (y_true == 0),  # 预测看空且正确→赚theta（避免亏损）
            theta,
            np.where(
                (y_pred == 1) & (y_true == 0),  # 预测看多但错误→亏theta
                -theta,
                np.where(
                    (y_pred == 0) & (y_true == 1),  # 预测看空但错误→亏theta
                    -theta,
                    0  # 震荡→收益0
                )
            )
        )
    )
    
    # 计算累计收益
    cumulative_return = np.cumsum(strategy_return)
    total_return = cumulative_return[-1]
    win_rate = np.mean(strategy_return > 0)  # 盈利次数占比
    
    # 打印模拟结果
    print(f"\n=== 模拟实盘收益（持有{predict_days}天） ===")
    print(f"   - 累计收益：{total_return:.4f}%（按theta={theta}%计算）")
    print(f"   - 盈利次数占比：{win_rate:.2%}")
    print(f"   - 总交易次数：{len(strategy_return)}次（排除震荡）")
    
    # 绘制累计收益曲线
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(10, 4))
    plt.plot(cumulative_return, label=f"策略累计收益（总收益：{total_return:.4f}%）")
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="盈亏平衡线")
    plt.title(f"LSTM策略模拟累计收益（预测未来{predict_days}天）")
    plt.xlabel("交易次数")
    plt.ylabel("累计收益（%）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "simulated_return.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 模拟收益曲线已保存到：{os.path.join(save_dir, 'simulated_return.png')}")

# 调用函数评估模型
y_pred = evaluate_lstm_model(model, X_test, y_test, MODEL_SAVE_DIR, PREDICT_DAYS)


# --------------------------
# 6. 保存最终模型和关键信息（供后续预测新数据）
# --------------------------
def save_final_assets(model, scaler, feature_cols, save_dir):
    """保存模型、标准化器、特征列，方便后续用新数据预测"""
    # 1. 保存最终模型（已通过checkpoint保存最优模型，这里可省略）
    final_model_path = os.path.join(save_dir, "final_lstm_model.keras")
    model.save(final_model_path)
    
    # 2. 复制标准化器到模型文件夹（方便统一管理）
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    
    # 3. 保存特征列名
    with open(os.path.join(save_dir, "feature_cols.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feature_cols))
    
    print(f"\n🎉 所有资产保存完成！")
    print(f"   - 模型路径：{final_model_path}")
    print(f"   - 标准化器：{os.path.join(save_dir, 'scaler.pkl')}")

# 加载特征列名（用于保存）
with open(os.path.join(FEATURE_DATA_DIR, "feature_cols.txt"), "r") as f:
    feature_cols = f.read().splitlines()

# 调用函数保存资产
save_final_assets(model, scaler, feature_cols, MODEL_SAVE_DIR)