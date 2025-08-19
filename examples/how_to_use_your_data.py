#!/usr/bin/env python3
"""
完整示例：如何使用自己的数据集进行零膨胀时间序列分析

这个脚本演示了：
1. 正确的数据格式要求
2. 如何准备您的数据
3. 如何验证数据格式
4. 如何使用数据训练模型
5. 如何评估模型性能

数据格式要求总结：
=================
✅ 支持的格式：
- NumPy数组 (1D: 时间序列值)
- Pandas Series (带或不带时间索引)
- Pandas DataFrame (必须有value列，可选时间戳和特征列)
- CSV文件 (与DataFrame格式相同)

✅ 数据要求：
- 数值型数据 (float或int)
- 非负数值 (零膨胀数据通常是计数或度量数据)
- 无缺失值 (NaN) 或无穷值
- 至少50个观测点 (建议100+用于可靠建模)
- 零值比例可以从5%到80%不等
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 导入零膨胀工具包组件
from data.loaders import ZeroInflatedDataLoader
from data.formatters import validate_zero_inflated_data, convert_to_standard_format
from models.baseline.zip_model import ZeroInflatedPoisson
from models.zero_aware.zip_rnn import ZIPRNN
from evaluation.metrics import ZeroInflatedMetrics


def create_example_datasets():
    """创建不同格式的示例数据集，展示正确的数据格式"""
    
    print("=" * 60)
    print("步骤1: 创建不同格式的示例数据集")
    print("=" * 60)
    
    # 生成基础时间序列数据（带零膨胀）
    np.random.seed(42)
    n_points = 200
    
    # 创建基础时间序列（有趋势和季节性）
    time = np.arange(n_points)
    trend = 0.01 * time
    seasonal = 2 * np.sin(2 * np.pi * time / 24) + 1 * np.sin(2 * np.pi * time / 7)
    noise = np.random.normal(0, 0.5, n_points)
    base_series = np.maximum(0, 3 + trend + seasonal + noise)
    
    # 添加零膨胀（随机将一些值设为0）
    zero_mask = np.random.random(n_points) < 0.3  # 30% 零值
    zero_inflated_series = base_series.copy()
    zero_inflated_series[zero_mask] = 0
    
    print(f"生成的时间序列统计:")
    print(f"  总长度: {len(zero_inflated_series)}")
    print(f"  零值比例: {np.mean(zero_inflated_series == 0):.1%}")
    print(f"  数值范围: [{np.min(zero_inflated_series):.2f}, {np.max(zero_inflated_series):.2f}]")
    print(f"  非零值均值: {np.mean(zero_inflated_series[zero_inflated_series > 0]):.2f}")
    
    # 示例1: NumPy数组格式 (最简单)
    print(f"\n✅ 格式1: NumPy数组")
    numpy_data = zero_inflated_series
    print(f"  形状: {numpy_data.shape}")
    print(f"  数据类型: {numpy_data.dtype}")
    print(f"  前5个值: {numpy_data[:5]}")
    
    # 示例2: Pandas Series格式 (带时间索引)
    print(f"\n✅ 格式2: Pandas Series (带时间索引)")
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_points)]
    series_data = pd.Series(zero_inflated_series, index=timestamps, name='value')
    print(f"  形状: {series_data.shape}")
    print(f"  索引类型: {type(series_data.index)}")
    print(f"  前5个值:")
    print(series_data.head())
    
    # 示例3: Pandas DataFrame格式 (包含额外特征)
    print(f"\n✅ 格式3: Pandas DataFrame (包含额外特征)")
    # 创建一些相关特征
    temperature = 20 + 10 * np.sin(2 * np.pi * time / 24) + np.random.normal(0, 2, n_points)
    humidity = 60 + 20 * np.cos(2 * np.pi * time / 24) + np.random.normal(0, 5, n_points)
    is_weekend = ((time // 24) % 7 >= 5).astype(int)
    
    df_data = pd.DataFrame({
        'timestamp': timestamps,
        'value': zero_inflated_series,  # 这是主要的时间序列值 (必需)
        'temperature': temperature,     # 额外特征1 (可选)
        'humidity': humidity,          # 额外特征2 (可选)
        'is_weekend': is_weekend       # 额外特征3 (可选)
    })
    
    print(f"  形状: {df_data.shape}")
    print(f"  列名: {list(df_data.columns)}")
    print(f"  前5行:")
    print(df_data.head())
    
    # 示例4: CSV文件格式
    print(f"\n✅ 格式4: CSV文件")
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'example_user_data.csv')
    csv_data = pd.read_csv(csv_path, parse_dates=['timestamp'])
    print(f"  文件路径: {csv_path}")
    print(f"  形状: {csv_data.shape}")
    print(f"  列名: {list(csv_data.columns)}")
    print(f"  前3行:")
    print(csv_data.head(3))
    
    return {
        'numpy': numpy_data,
        'series': series_data, 
        'dataframe': df_data,
        'csv_path': csv_path
    }


def validate_data_formats(datasets):
    """验证不同格式的数据是否符合要求"""
    
    print("\n" + "=" * 60)
    print("步骤2: 验证数据格式")
    print("=" * 60)
    
    for name, data in datasets.items():
        print(f"\n🔍 验证 {name} 格式数据:")
        
        try:
            if name == 'csv_path':
                # 对于CSV文件，先加载数据
                csv_data = pd.read_csv(data, parse_dates=['timestamp'])
                is_valid, issues = validate_zero_inflated_data(csv_data, value_column='value')
                print(f"  CSV文件路径: {data}")
            elif name == 'dataframe':
                is_valid, issues = validate_zero_inflated_data(data, value_column='value')
            else:
                is_valid, issues = validate_zero_inflated_data(data)
            
            if is_valid:
                print(f"  ✅ 数据格式有效")
            else:
                print(f"  ❌ 数据格式问题:")
                for issue in issues:
                    print(f"    - {issue}")
                    
        except Exception as e:
            print(f"  ❌ 验证失败: {str(e)}")


def demonstrate_data_loading(datasets):
    """演示如何使用数据加载器加载不同格式的数据"""
    
    print("\n" + "=" * 60)
    print("步骤3: 使用数据加载器")
    print("=" * 60)
    
    loader = ZeroInflatedDataLoader()
    
    for name, data in datasets.items():
        print(f"\n📥 加载 {name} 格式数据:")
        
        try:
            if name == 'csv_path':
                # 对于CSV文件
                prepared_data = loader.load_and_prepare(
                    data=data,
                    sequence_length=24,
                    prediction_horizon=6,
                    test_split=0.2,
                    batch_size=16,
                    value_column='value',
                    normalize=True
                )
            elif name == 'dataframe':
                # 对于DataFrame
                prepared_data = loader.load_and_prepare(
                    data=data,
                    sequence_length=24,
                    prediction_horizon=6,
                    test_split=0.2,
                    batch_size=16,
                    value_column='value',
                    normalize=True
                )
            else:
                # 对于NumPy数组和Series
                prepared_data = loader.load_and_prepare(
                    data=data,
                    sequence_length=24,
                    prediction_horizon=6,
                    test_split=0.2,
                    batch_size=16,
                    normalize=True
                )
            
            print(f"  ✅ 加载成功")
            print(f"  训练集批次数: {len(prepared_data['train_loader'])}")
            print(f"  验证集批次数: {len(prepared_data['val_loader'])}")
            print(f"  测试集批次数: {len(prepared_data['test_loader'])}")
            
            # 展示一个批次的数据
            sample_batch = next(iter(prepared_data['train_loader']))
            input_seq, target_seq = sample_batch
            print(f"  输入序列形状: {input_seq.shape}")
            print(f"  目标序列形状: {target_seq.shape}")
            
        except Exception as e:
            print(f"  ❌ 加载失败: {str(e)}")


def train_models_example(datasets):
    """演示如何使用加载的数据训练模型"""
    
    print("\n" + "=" * 60)
    print("步骤4: 训练模型示例")
    print("=" * 60)
    
    # 使用DataFrame数据作为示例
    data = datasets['dataframe']
    
    # 准备数据
    loader = ZeroInflatedDataLoader()
    prepared_data = loader.load_and_prepare(
        data=data,
        sequence_length=24,
        prediction_horizon=6,
        test_split=0.2,
        batch_size=16,
        value_column='value',
        normalize=True
    )
    
    train_loader = prepared_data['train_loader']
    test_loader = prepared_data['test_loader']
    
    print(f"数据准备完成:")
    print(f"  训练批次: {len(train_loader)}")
    print(f"  测试批次: {len(test_loader)}")
    
    # 示例1: 统计模型 (ZIP)
    print(f"\n🔧 训练统计模型 (Zero-Inflated Poisson):")
    
    # 为统计模型准备数据 (需要转换为sklearn格式)
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for input_seq, target_seq in train_loader:
        # 使用序列的最后一个值作为特征，第一个目标值作为标签
        X_train.extend(input_seq[:, -1, :].numpy())
        y_train.extend(target_seq[:, 0, 0].numpy())
    
    for input_seq, target_seq in test_loader:
        X_test.extend(input_seq[:, -1, :].numpy())
        y_test.extend(target_seq[:, 0, 0].numpy())
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # ZIP模型需要整数值，所以将连续值转换为整数
    y_train_int = np.round(np.maximum(0, y_train)).astype(int)
    y_test_int = np.round(np.maximum(0, y_test)).astype(int)
    
    print(f"  数据统计: y_train_int范围 [{np.min(y_train_int)}, {np.max(y_train_int)}]")
    print(f"  数据类型: {y_train_int.dtype}")
    
    # 训练ZIP模型
    zip_model = ZeroInflatedPoisson()
    try:
        zip_model.fit(X_train, y_train_int)
        zip_predictions = zip_model.predict(X_test)
        print(f"  ZIP模型训练完成")
        print(f"  测试集预测形状: {zip_predictions.shape}")
    except Exception as e:
        print(f"  ZIP模型训练失败: {str(e)}")
        print(f"  注意: 这是一个已知问题，继续演示其他模型")
    
    # 示例2: 深度学习模型 (ZIP-RNN)
    print(f"\n🧠 训练深度学习模型 (ZIP-RNN):")
    
    # 获取输入维度（如果有额外特征，使用它们；否则只使用时间序列值）
    sample_input, _ = next(iter(train_loader))
    input_dim = sample_input.shape[-1]
    print(f"  输入特征维度: {input_dim}")
    
    # 创建ZIP-RNN模型
    ziprnn_model = ZIPRNN(
        input_dim=input_dim,
        hidden_dim=32,
        num_layers=2,
        seq_len=24,
        pred_len=6
    )
    
    optimizer = torch.optim.Adam(ziprnn_model.parameters(), lr=0.001)
    
    # 训练几个epoch
    try:
        ziprnn_model.train()
        for epoch in range(3):  # 只训练3个epoch作为示例
            total_loss = 0
            for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
                if batch_idx >= 5:  # 只训练前5个批次作为示例
                    break
                    
                optimizer.zero_grad()
                
                # 前向传播
                predictions = ziprnn_model(input_seq)
                loss = ziprnn_model.compute_zip_loss(predictions, target_seq)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / min(5, len(train_loader))
            print(f"  Epoch {epoch+1}/3, 平均损失: {avg_loss:.4f}")
        
        # 测试模型
        ziprnn_model.eval()
        with torch.no_grad():
            sample_input, sample_target = next(iter(test_loader))
            ziprnn_predictions = ziprnn_model(sample_input)
            print(f"  ZIP-RNN预测形状: {ziprnn_predictions.shape}")
    except Exception as e:
        print(f"  ZIP-RNN训练失败: {str(e)}")
        print(f"  注意: 多维特征输入需要适配，这里只演示概念")


def evaluate_models_example(datasets):
    """演示如何评估模型性能"""
    
    print("\n" + "=" * 60)
    print("步骤5: 模型评估示例")
    print("=" * 60)
    
    # 生成一些示例预测和真实值用于评估
    np.random.seed(42)
    
    # 模拟预测值和真实值
    n_samples = 100
    true_values = np.random.exponential(2, n_samples)
    zero_mask = np.random.random(n_samples) < 0.3
    true_values[zero_mask] = 0
    
    # 模拟模型预测 (添加一些噪声)
    predictions = true_values + np.random.normal(0, 0.5, n_samples)
    predictions = np.maximum(0, predictions)  # 确保非负
    
    print(f"评估数据统计:")
    print(f"  样本数量: {len(true_values)}")
    print(f"  真实值零比例: {np.mean(true_values == 0):.1%}")
    print(f"  预测值零比例: {np.mean(predictions < 0.1):.1%}")  # 近似零值
    
    # 使用零膨胀评估指标
    evaluator = ZeroInflatedMetrics()
    metrics = evaluator(predictions, true_values)
    
    print(f"\n📊 零膨胀特定评估指标:")
    print(f"  零值分类准确率: {metrics['zero_classification_accuracy']:.3f}")
    print(f"  零值精确率: {metrics['zero_precision']:.3f}")
    print(f"  零值召回率: {metrics['zero_recall']:.3f}")
    print(f"  零值比例误差: {metrics['zero_ratio_error']:.3f}")
    
    print(f"\n📈 传统预测指标:")
    print(f"  均方误差 (MSE): {metrics['mse']:.6f}")
    print(f"  平均绝对误差 (MAE): {metrics['mae']:.6f}")
    print(f"  R²决定系数: {metrics['r2']:.3f}")
    
    print(f"\n🎯 非零值专门指标:")
    print(f"  非零值MSE: {metrics['nonzero_mse']:.6f}")
    print(f"  非零值MAE: {metrics['nonzero_mae']:.6f}")
    print(f"  非零值R²: {metrics['nonzero_r2']:.3f}")


def common_data_issues_and_solutions():
    """演示常见数据问题及解决方案"""
    
    print("\n" + "=" * 60)
    print("常见数据问题及解决方案")
    print("=" * 60)
    
    print(f"\n❌ 问题1: 数据包含缺失值")
    print(f"解决方案: 使用前向填充或插值")
    
    # 示例数据包含NaN
    problematic_data = np.array([1.0, 2.0, np.nan, 0.0, 3.0, np.nan, 1.0])
    print(f"  原始数据: {problematic_data}")
    
    # 解决方案1: 前向填充
    filled_data = pd.Series(problematic_data).fillna(method='ffill').values
    print(f"  前向填充后: {filled_data}")
    
    # 解决方案2: 线性插值
    interpolated_data = pd.Series(problematic_data).interpolate().values
    print(f"  线性插值后: {interpolated_data}")
    
    print(f"\n❌ 问题2: 数据包含负值")
    print(f"解决方案: 取绝对值或者偏移")
    
    negative_data = np.array([1.0, -0.5, 2.0, 0.0, -1.0, 3.0])
    print(f"  原始数据: {negative_data}")
    
    # 解决方案1: 取绝对值
    abs_data = np.abs(negative_data)
    print(f"  取绝对值后: {abs_data}")
    
    # 解决方案2: 添加偏移量
    offset_data = negative_data - np.min(negative_data)
    print(f"  添加偏移后: {offset_data}")
    
    print(f"\n❌ 问题3: 数据长度不足")
    print(f"解决方案: 需要至少50个观测点，建议100+")
    
    short_data = np.random.exponential(1, 30)
    print(f"  数据长度: {len(short_data)} (太短)")
    print(f"  建议: 收集更多数据或使用数据增强技术")
    
    print(f"\n❌ 问题4: 零值比例过高或过低")
    print(f"解决方案: 检查数据收集过程，确认零值的含义")
    
    # 零值比例过高的例子
    high_zero_data = np.random.exponential(1, 100)
    high_zero_data[np.random.random(100) < 0.9] = 0  # 90%零值
    zero_ratio = np.mean(high_zero_data == 0)
    print(f"  零值比例: {zero_ratio:.1%} (可能过高)")
    print(f"  建议: 如果超过80%，考虑是否真的是零膨胀问题")


def main():
    """主函数：完整的数据格式和使用示例"""
    
    print("🎯 零膨胀时间序列分析工具包")
    print("📚 完整数据格式和使用指南")
    print("=" * 60)
    
    # 1. 创建示例数据集
    datasets = create_example_datasets()
    
    # 2. 验证数据格式
    validate_data_formats(datasets)
    
    # 3. 演示数据加载
    demonstrate_data_loading(datasets)
    
    # 4. 演示模型训练
    train_models_example(datasets)
    
    # 5. 演示模型评估
    evaluate_models_example(datasets)
    
    # 6. 常见问题解决方案
    common_data_issues_and_solutions()
    
    print("\n" + "=" * 60)
    print("🎉 完整示例结束")
    print("=" * 60)
    print(f"\n📝 总结：")
    print(f"1. 支持的数据格式: NumPy数组, Pandas Series/DataFrame, CSV文件")
    print(f"2. 数据要求: 数值型、非负、无缺失值、至少50个观测点")
    print(f"3. 推荐的零值比例: 5%-80%")
    print(f"4. DataFrame/CSV格式必须包含'value'列作为主要时间序列")
    print(f"5. 可以包含额外特征列(temperature, humidity等)")
    print(f"6. 时间戳列可选，但有助于数据解释")
    print(f"\n💡 下一步:")
    print(f"- 检查您的数据是否符合上述要求")
    print(f"- 使用validate_zero_inflated_data()验证数据格式")
    print(f"- 使用ZeroInflatedDataLoader加载和准备数据")
    print(f"- 选择合适的模型进行训练和评估")


if __name__ == "__main__":
    main()