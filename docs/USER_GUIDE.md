# 用户指南：如何使用自己的数据集

这个指南将详细说明如何准备和使用您自己的数据集进行零膨胀时间序列分析。

## 📋 目录

1. [数据格式要求](#数据格式要求)
2. [准备您的数据](#准备您的数据)
3. [数据验证和转换](#数据验证和转换)
4. [完整工作流程示例](#完整工作流程示例)
5. [常见问题和解决方案](#常见问题和解决方案)
6. [最佳实践](#最佳实践)

## 📊 数据格式要求

### 基本要求

您的数据需要满足以下基本要求：

| 要求 | 说明 | 示例 |
|------|------|------|
| **数据类型** | 数值型（整数或浮点数） | `1.2, 0.0, 3.5, 0.0` |
| **非负值** | 所有值 ≥ 0 | ✅ `[0, 1.2, 3.0]` ❌ `[0, -1.2, 3.0]` |
| **无缺失值** | 不能有 NaN 或空值 | ✅ `[0, 1.2, 3.0]` ❌ `[0, NaN, 3.0]` |
| **最小长度** | 至少 50 个观测点 | 建议 100+ 用于可靠建模 |
| **零值意义** | 零值应该是有意义的 | 销量为0、访问量为0等 |

### 零膨胀特征

您的数据应该表现出零膨胀特征：

- **零值比例**: 通常在 10% - 80% 之间
- **零值不是错误**: 零值代表真实的"无事件"状态
- **零值模式**: 零值可能有时间模式（如夜间、周末等）

### 时间序列特征

- **时间顺序**: 数据应该按时间顺序排列
- **规律间隔**: 最好有规律的时间间隔（小时、天、周等）
- **足够历史**: 足够的历史数据来学习模式

## 🔧 准备您的数据

### 格式1: NumPy数组

最简单的格式，适用于单变量时间序列：

```python
import numpy as np

# 示例：商店每日销量数据
daily_sales = np.array([
    12.5, 0.0, 15.2, 0.0, 8.7, 23.1, 0.0,  # 第1周
    18.3, 0.0, 21.4, 0.0, 9.5, 25.8, 0.0,  # 第2周
    # ... 更多数据
])

# 检查零值比例
zero_ratio = np.mean(daily_sales == 0)
print(f"零值比例: {zero_ratio:.3f}")  # 应该在 0.1-0.8 之间
```

### 格式2: Pandas Series

适用于有时间索引的单变量时间序列：

```python
import pandas as pd
import numpy as np

# 创建时间索引
dates = pd.date_range('2023-01-01', periods=365, freq='D')

# 创建Series
sales_data = pd.Series(
    data=your_sales_values,  # 您的数据
    index=dates,
    name='daily_sales'
)

print(f"数据长度: {len(sales_data)}")
print(f"零值比例: {(sales_data == 0).mean():.3f}")
print(f"时间范围: {sales_data.index.min()} 到 {sales_data.index.max()}")
```

### 格式3: Pandas DataFrame（推荐）

最灵活的格式，支持多个特征：

```python
import pandas as pd
import numpy as np

# 创建DataFrame
data = pd.DataFrame({
    # 必需：时间戳列
    'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
    
    # 必需：目标变量（要预测的零膨胀时间序列）
    'sales': your_target_values,  # 这是主要的预测目标
    
    # 可选：额外特征
    'temperature': np.random.normal(20, 10, 1000),  # 温度
    'is_weekend': is_weekend_array,                  # 是否周末
    'promotion': promotion_indicator,                # 促销标记
    'store_id': store_identifier,                    # 店铺ID
    'product_category': category_codes               # 产品类别
})

print(data.info())
print(f"\n零值比例: {(data['sales'] == 0).mean():.3f}")
```

### 格式4: CSV文件

适用于数据存储在文件中的情况：

```csv
timestamp,sales,temperature,is_weekend,promotion
2023-01-01 00:00:00,12.5,18.2,0,0
2023-01-01 01:00:00,0.0,17.8,0,0
2023-01-01 02:00:00,0.0,17.5,0,0
2023-01-01 03:00:00,0.0,17.1,0,0
2023-01-01 04:00:00,0.0,16.9,0,0
2023-01-01 05:00:00,0.0,16.8,0,0
2023-01-01 06:00:00,8.3,17.2,0,0
2023-01-01 07:00:00,15.7,18.1,0,0
2023-01-01 08:00:00,23.4,19.5,0,1
2023-01-01 09:00:00,18.9,20.8,0,1
...
```

**CSV文件要求**：
- 第一行必须是列名
- 时间戳列应该可以被 `pd.to_datetime()` 解析
- 数值列不能包含非数字字符
- 编码建议使用 UTF-8

## ✅ 数据验证和转换

使用内置工具验证和转换您的数据：

### 步骤1: 数据验证

```python
from data.formatters import validate_zero_inflated_data

# 验证不同格式的数据
# 对于NumPy数组
is_valid, issues = validate_zero_inflated_data(your_numpy_array)

# 对于Pandas DataFrame
is_valid, issues = validate_zero_inflated_data(
    your_dataframe, 
    value_column='sales'  # 指定目标列
)

# 对于CSV文件
is_valid, issues = validate_zero_inflated_data(
    'path/to/your/data.csv',
    value_column='sales'
)

# 检查验证结果
if is_valid:
    print("✅ 数据格式正确！")
else:
    print("❌ 数据格式问题：")
    for issue in issues:
        print(f"   - {issue}")
```

### 步骤2: 转换为标准格式

```python
from data.formatters import convert_to_standard_format

# 转换为标准格式
standard_data = convert_to_standard_format(
    data=your_data,                    # 您的数据
    value_column='sales',              # 目标列名（DataFrame/CSV用）
    timestamp_column='timestamp',      # 时间列名（可选）
    feature_columns=[                  # 特征列名（可选）
        'temperature', 
        'is_weekend', 
        'promotion'
    ],
    zero_threshold=1e-6               # 零值阈值
)

# 查看转换结果
summary = standard_data.get_summary_stats()
print("数据摘要：")
for key, value in summary.items():
    print(f"  {key}: {value}")
```

## 🚀 完整工作流程示例

以下是一个完整的端到端示例，展示如何从原始数据到模型训练：

### 示例场景：在线商店每小时销量预测

```python
import pandas as pd
import numpy as np
from data.loaders import ZeroInflatedDataLoader
from models.zero_aware.zip_rnn import ZIPRNN
from evaluation.metrics import ZeroInflatedMetrics

# 第1步：加载您的数据
# 假设您有一个CSV文件包含商店销售数据
data_file = 'your_store_sales.csv'

# 或者您有一个DataFrame
data = pd.DataFrame({
    'datetime': pd.date_range('2023-01-01', periods=8760, freq='H'),  # 一年的小时数据
    'sales': your_sales_data,        # 每小时销量（可能为0）
    'temperature': temperature_data,  # 温度
    'is_weekend': weekend_flags,     # 周末标记
    'hour_of_day': hour_data,       # 一天中的小时
    'month': month_data,            # 月份
    'promotion': promotion_flags     # 促销活动
})

# 第2步：验证数据
from data.formatters import validate_zero_inflated_data

is_valid, issues = validate_zero_inflated_data(
    data, 
    value_column='sales'
)

if not is_valid:
    print("数据问题：")
    for issue in issues:
        print(f"  - {issue}")
    # 处理数据问题...

# 第3步：准备训练数据
loader = ZeroInflatedDataLoader(zero_threshold=1e-6)

prepared_data = loader.load_and_prepare(
    data=data,
    sequence_length=24,              # 使用24小时历史数据
    prediction_horizon=6,            # 预测未来6小时
    test_split=0.2,                 # 20%作为测试集
    validation_split=0.15,           # 15%作为验证集
    batch_size=32,                  # 批次大小
    value_column='sales',           # 目标列
    timestamp_column='datetime',    # 时间列
    feature_columns=[               # 特征列
        'temperature', 
        'is_weekend', 
        'hour_of_day', 
        'month', 
        'promotion'
    ],
    normalize=True                  # 标准化
)

# 第4步：获取数据加载器
train_loader = prepared_data['train_loader']
val_loader = prepared_data['val_loader']
test_loader = prepared_data['test_loader']

print(f"训练数据: {len(train_loader)} 批次")
print(f"验证数据: {len(val_loader)} 批次")
print(f"测试数据: {len(test_loader)} 批次")

# 第5步：创建和训练模型
feature_dim = prepared_data['data_info']['feature_dimension']

model = ZIPRNN(
    input_dim=feature_dim,
    hidden_dim=64,
    num_layers=2,
    seq_len=24,
    pred_len=6,
    rnn_type='LSTM'
)

# 训练模型
import torch
import torch.nn as nn

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        # 前向传播
        predictions = model(input_seq)
        
        # 计算损失
        components = model(input_seq, return_components=True)
        loss = model.compute_zip_loss(
            predictions, target_seq, 
            components['pi'], components['lambda']
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 第6步：评估模型
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for input_seq, target_seq in test_loader:
        predictions = model(input_seq)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(target_seq.cpu().numpy())

# 合并所有预测结果
predictions = np.concatenate(all_predictions, axis=0)
targets = np.concatenate(all_targets, axis=0)

# 计算评估指标
evaluator = ZeroInflatedMetrics(zero_threshold=1e-6)
metrics = evaluator(predictions.flatten(), targets.flatten())

print("\n评估结果:")
print(f"整体MSE: {metrics['mse']:.6f}")
print(f"整体MAE: {metrics['mae']:.6f}")
print(f"零值分类准确率: {metrics['zero_classification_accuracy']:.3f}")
print(f"零值F1得分: {metrics['zero_f1']:.3f}")
print(f"非零值R²: {metrics['nonzero_r2']:.3f}")
print(f"零值比例误差: {metrics['zero_ratio_error']:.3f}")
```

## 🔍 常见问题和解决方案

### 问题1: 数据包含负值

**问题**: 数据验证失败，提示包含负值。

**解决方案**:
```python
# 检查负值
negative_mask = data['sales'] < 0
print(f"负值数量: {negative_mask.sum()}")
print(f"负值位置: {np.where(negative_mask)[0]}")

# 解决方法1: 如果负值是错误，设为0
data.loc[data['sales'] < 0, 'sales'] = 0

# 解决方法2: 如果负值有意义，考虑数据变换
# 注意：这会改变零膨胀的性质
data['sales'] = data['sales'] - data['sales'].min()

# 解决方法3: 使用允许负值的验证
is_valid, issues = validate_zero_inflated_data(
    data, 
    value_column='sales',
    allow_negative=True
)
```

### 问题2: 零值比例过高或过低

**问题**: 零值比例不在合理范围内。

**解决方案**:
```python
zero_ratio = (data['sales'] == 0).mean()
print(f"当前零值比例: {zero_ratio:.3f}")

if zero_ratio < 0.05:
    print("警告: 零值比例过低，可能不需要零膨胀模型")
    # 考虑使用传统时间序列模型
    
elif zero_ratio > 0.9:
    print("警告: 零值比例过高，数据可能有问题")
    # 检查数据收集过程
    # 考虑是否需要聚合数据（如小时变为天）
    
else:
    print("零值比例在合理范围内")
```

### 问题3: 时间序列不规律

**问题**: 时间间隔不规律或有缺失的时间点。

**解决方案**:
```python
# 检查时间间隔
time_diffs = data['timestamp'].diff()
print(f"时间间隔统计:")
print(time_diffs.describe())

# 重新采样到规律间隔
data.set_index('timestamp', inplace=True)
regular_data = data.resample('H').agg({  # 按小时重新采样
    'sales': 'sum',        # 销量求和
    'temperature': 'mean', # 温度取平均
    'promotion': 'max'     # 促销取最大值（是否有促销）
}).reset_index()

# 填充缺失值
regular_data['sales'].fillna(0, inplace=True)  # 销量缺失设为0
regular_data['temperature'].interpolate(inplace=True)  # 温度插值
```

### 问题4: 特征列包含缺失值

**问题**: 额外特征包含NaN值。

**解决方案**:
```python
# 检查缺失值
print("缺失值统计:")
print(data.isnull().sum())

# 处理缺失值
for column in ['temperature', 'humidity', 'other_features']:
    if column in data.columns:
        if data[column].dtype in ['float64', 'int64']:
            # 数值型：用均值或中位数填充
            data[column].fillna(data[column].median(), inplace=True)
        else:
            # 分类型：用众数填充
            data[column].fillna(data[column].mode()[0], inplace=True)

# 或者删除包含缺失值的行
# data.dropna(inplace=True)  # 但要确保数据仍然足够
```

### 问题5: 内存不足

**问题**: 数据太大，内存不足。

**解决方案**:
```python
# 减少数据精度
data = data.astype({
    'sales': 'float32',      # 从float64减少到float32
    'temperature': 'float32',
    'other_numeric': 'float32'
})

# 使用数据分块处理
def process_in_chunks(data, chunk_size=10000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        # 处理chunk
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    return pd.concat(results, ignore_index=True)

# 减少数据长度
# 只使用最近的数据
recent_data = data.tail(50000)  # 只使用最后50000个点
```

## 💡 最佳实践

### 数据准备最佳实践

1. **了解您的数据**
   ```python
   # 全面的数据探索
   print("数据基本信息:")
   print(f"数据长度: {len(data)}")
   print(f"时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
   print(f"零值比例: {(data['sales'] == 0).mean():.3f}")
   print(f"非零值统计:")
   print(data[data['sales'] > 0]['sales'].describe())
   ```

2. **可视化您的数据**
   ```python
   import matplotlib.pyplot as plt
   
   fig, axes = plt.subplots(3, 1, figsize=(15, 12))
   
   # 时间序列图
   axes[0].plot(data['timestamp'], data['sales'])
   axes[0].set_title('时间序列')
   axes[0].set_ylabel('销量')
   
   # 零值分布
   zero_positions = data[data['sales'] == 0]['timestamp']
   axes[1].scatter(zero_positions, [1]*len(zero_positions), alpha=0.5)
   axes[1].set_title('零值出现时间')
   axes[1].set_ylabel('零值')
   
   # 非零值分布
   non_zero_data = data[data['sales'] > 0]['sales']
   axes[2].hist(non_zero_data, bins=50)
   axes[2].set_title('非零值分布')
   axes[2].set_xlabel('销量')
   
   plt.tight_layout()
   plt.show()
   ```

3. **特征工程**
   ```python
   # 添加有用的时间特征
   data['hour'] = data['timestamp'].dt.hour
   data['day_of_week'] = data['timestamp'].dt.dayofweek
   data['month'] = data['timestamp'].dt.month
   data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
   
   # 添加滞后特征
   data['sales_lag_1'] = data['sales'].shift(1)
   data['sales_lag_24'] = data['sales'].shift(24)  # 24小时前的销量
   
   # 添加滚动统计特征
   data['sales_rolling_mean_7d'] = data['sales'].rolling(window=24*7).mean()
   data['sales_rolling_std_7d'] = data['sales'].rolling(window=24*7).std()
   ```

4. **数据分割策略**
   ```python
   # 对于时间序列，使用时间顺序分割
   total_len = len(data)
   train_end = int(0.7 * total_len)
   val_end = int(0.85 * total_len)
   
   train_data = data[:train_end]
   val_data = data[train_end:val_end] 
   test_data = data[val_end:]
   
   print(f"训练集: {len(train_data)} 样本")
   print(f"验证集: {len(val_data)} 样本")
   print(f"测试集: {len(test_data)} 样本")
   ```

### 模型选择建议

根据您的数据特征选择合适的模型：

| 数据特征 | 推荐模型 | 原因 |
|----------|----------|------|
| 零值比例 < 20% | Tweedie GLM | 轻微零膨胀，传统方法有效 |
| 零值比例 20-50% | ZIP, ZINB | 中等零膨胀，专业模型更好 |
| 零值比例 > 50% | Hurdle Model, Dual Branch | 严重零膨胀，需要两阶段建模 |
| 复杂时间模式 | ZIP-RNN, Transformer | 神经网络能捕获复杂模式 |
| 多个特征 | Dual Branch, Weighted Transformer | 能有效利用多元信息 |
| 数据量大 | 深度学习模型 | 有足够数据训练复杂模型 |
| 数据量小 | 统计模型 | 参数少，不容易过拟合 |

### 性能优化建议

1. **序列长度选择**
   ```python
   # 实验不同序列长度
   sequence_lengths = [12, 24, 48, 96, 168]  # 12小时到1周
   
   for seq_len in sequence_lengths:
       prepared_data = loader.load_and_prepare(
           data=data,
           sequence_length=seq_len,
           prediction_horizon=6,
           # ... 其他参数
       )
       # 训练和评估模型
       # 选择性能最好的序列长度
   ```

2. **批次大小优化**
   ```python
   # 根据内存和性能选择批次大小
   batch_sizes = [16, 32, 64, 128]
   
   for batch_size in batch_sizes:
       # 测试训练速度和内存使用
       # 选择平衡点
   ```

3. **特征选择**
   ```python
   from sklearn.feature_selection import mutual_info_regression
   
   # 计算特征重要性
   feature_cols = ['temperature', 'is_weekend', 'hour', 'promotion']
   X_features = data[feature_cols].fillna(0)
   y_target = data['sales']
   
   importance = mutual_info_regression(X_features, y_target)
   feature_importance = pd.DataFrame({
       'feature': feature_cols,
       'importance': importance
   }).sort_values('importance', ascending=False)
   
   print("特征重要性:")
   print(feature_importance)
   
   # 只使用重要的特征
   important_features = feature_importance.head(3)['feature'].tolist()
   ```

这个用户指南应该能够帮助您成功地使用自己的数据集进行零膨胀时间序列分析。如果您有任何问题，请参考项目的其他文档或创建Issue。