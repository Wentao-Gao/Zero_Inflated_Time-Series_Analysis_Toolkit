# 零膨胀时间序列分析综合工具包

这是一个全面的零膨胀时间序列分析工具包，提供从数据预处理到模型训练评估的完整解决方案。该工具包专门设计用于处理包含大量零值的时间序列数据，这在现实世界的许多应用中都很常见。

## 🌟 主要特性

### 📊 数据处理
- **多格式支持**: NumPy数组、Pandas Series/DataFrame、CSV文件
- **自动验证**: 数据格式和质量自动检查
- **智能转换**: 自动转换为标准时间序列格式
- **专业预处理**: 专门针对零膨胀数据的预处理工具

### 🔬 零膨胀机制
- **Threshold机制**: 基于阈值的零膨胀
- **Mixture机制**: 混合分布零膨胀
- **Tweedie机制**: 基于Tweedie分布的自然零膨胀
- **Hurdle机制**: 两阶段跨栏模型

### 📈 统计模型
- **ZIP (Zero-Inflated Poisson)**: 零膨胀泊松回归
- **ZINB (Zero-Inflated Negative Binomial)**: 零膨胀负二项回归
- **Tweedie GLM**: Tweedie广义线性模型
- **Hurdle Model**: 跨栏模型

### 🧠 深度学习模型
- **ZIP-RNN**: 结合RNN和零膨胀泊松分布的神经网络
- **Dual Branch Network**: 双分支网络分别建模零值和非零值
- **Weighted Loss Transformer**: 带加权损失的Transformer
- **Enhanced Tweedie Transformer**: 增强的Tweedie Transformer

### 🎯 评估系统
- **专业指标**: 专门为零膨胀数据设计的评估指标
- **交叉验证**: 时间序列感知的交叉验证
- **基准测试**: 标准化的基准测试套件
- **模型比较**: 自动化模型比较和报告

## 🚀 快速开始

### 安装依赖

```bash
pip install numpy pandas scikit-learn torch scipy matplotlib
```

### 基本使用示例

```python
import numpy as np
import pandas as pd
from data.loaders import ZeroInflatedDataLoader
from models.baseline.zip_model import ZeroInflatedPoisson
from evaluation.metrics import ZeroInflatedMetrics

# 1. 准备数据 - 支持多种格式
# 方式1: NumPy数组
data = np.array([1.2, 0.0, 2.1, 0.0, 1.8, 3.2, 0.0, 2.5])

# 方式2: Pandas DataFrame
data = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
    'value': [...],  # 您的时间序列数据
    'feature1': [...],  # 可选的额外特征
    'feature2': [...]   # 可选的额外特征
})

# 方式3: CSV文件
# data = '/path/to/your/data.csv'

# 2. 加载和准备数据
loader = ZeroInflatedDataLoader()
prepared_data = loader.load_and_prepare(
    data=data,
    sequence_length=24,      # 输入序列长度
    prediction_horizon=6,    # 预测时间跨度
    test_split=0.2,         # 测试集比例
    batch_size=32,          # 批次大小
    value_column='value',   # 目标列名称 (用于DataFrame/CSV)
    normalize=True          # 是否标准化
)

# 3. 获取数据加载器
train_loader = prepared_data['train_loader']
val_loader = prepared_data['val_loader']
test_loader = prepared_data['test_loader']

# 4. 训练模型
model = ZeroInflatedPoisson()
# 训练模型...

# 5. 评估模型
evaluator = ZeroInflatedMetrics()
metrics = evaluator(predictions, targets)
print(f"零值分类准确率: {metrics['zero_classification_accuracy']:.3f}")
print(f"整体MSE: {metrics['mse']:.6f}")
print(f"非零值R²: {metrics['nonzero_r2']:.3f}")
```

## 📋 数据格式要求

### 支持的数据格式

#### 1. NumPy数组格式
```python
# 1D数组：时间序列值
data = np.array([1.2, 0.0, 2.1, 0.0, 1.8, 3.2, 0.0, 2.5])
```

#### 2. Pandas Series格式
```python
# 带时间索引的Series
data = pd.Series(
    [1.2, 0.0, 2.1, 0.0, 1.8, 3.2, 0.0, 2.5], 
    index=pd.date_range('2023-01-01', periods=8, freq='D'),
    name='value'
)
```

#### 3. Pandas DataFrame格式
```python
data = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),  # 时间戳列
    'value': [...],        # 主要的时间序列值（必需）
    'temperature': [...],  # 额外特征1（可选）
    'humidity': [...],     # 额外特征2（可选）
    'is_weekend': [...]    # 额外特征3（可选）
})
```

#### 4. CSV文件格式
```csv
timestamp,value,temperature,humidity,is_weekend
2023-01-01 00:00:00,1.2,12.1,65.5,0
2023-01-01 01:00:00,0.0,11.8,66.2,0
2023-01-01 02:00:00,2.1,13.2,64.1,0
2023-01-01 03:00:00,0.0,12.5,65.8,0
...
```

### 数据要求

✅ **必需要求**:
- 数值型时间序列数据（float或int）
- 非负数值（零膨胀数据通常是计数或度量数据）
- 无缺失值（NaN）或无穷值
- 至少50个观测点（建议100+用于可靠建模）

✅ **可选要素**:
- 时间戳列或索引
- 额外的特征列
- 规则的时间间隔

✅ **零值处理**:
- 零值是预期的和有意义的
- 零值比例可以从5%到80%不等
- 工具包会自动检测和处理零膨胀程度

### 数据验证

使用内置验证器检查您的数据：

```python
from data.formatters import validate_zero_inflated_data

# 验证数据格式
is_valid, issues = validate_zero_inflated_data(
    your_data, 
    value_column='value'  # 用于DataFrame/CSV
)

if is_valid:
    print("✓ 数据格式有效")
else:
    print("✗ 数据格式问题:")
    for issue in issues:
        print(f"  - {issue}")
```

## 🎯 应用场景

该工具包特别适用于以下场景：

### 商业应用
- **销售预测**: 商品销量预测（许多时期销量为零）
- **客户行为**: 用户活动预测（用户可能在某些时期不活跃）
- **需求预测**: 服务或产品需求预测

### 工程应用
- **故障检测**: 设备故障次数预测
- **质量控制**: 缺陷数量预测
- **维护计划**: 维护需求预测

### 科学研究
- **生态学**: 物种出现次数预测
- **医学**: 疾病发病次数预测
- **社会科学**: 事件发生次数预测

### 互联网应用
- **网站流量**: 访问量预测（某些时段可能为零）
- **广告点击**: 点击率预测
- **内容消费**: 观看或阅读次数预测

## 🏗️ 项目结构

```
zero_inflated_comprehensive/
├── data/                    # 数据处理模块
│   ├── formatters.py       # 数据格式验证和转换
│   ├── loaders.py         # 数据加载器
│   └── preprocessors.py   # 数据预处理器
├── generation/             # 数据生成模块
│   ├── inject_zeros.py    # 零膨胀注入函数
│   └── zero_mechanisms.py # 零膨胀机制
├── models/                 # 模型模块
│   ├── baseline/          # 基准统计模型
│   │   ├── zip_model.py   # ZIP模型
│   │   ├── zinb_model.py  # ZINB模型
│   │   ├── tweedie_glm.py # Tweedie GLM
│   │   └── hurdle_model.py # 跨栏模型
│   ├── zero_aware/        # 零感知深度学习模型
│   │   ├── zip_rnn.py     # ZIP-RNN
│   │   ├── dual_branch_network.py # 双分支网络
│   │   ├── weighted_loss_transformer.py # 加权Transformer
│   │   └── tweedie_transformer.py # Tweedie Transformer
│   └── losses/            # 损失函数
│       ├── tweedie_loss.py
│       └── zero_aware_losses.py
├── evaluation/            # 评估模块
│   ├── metrics.py        # 评估指标
│   ├── evaluator.py      # 评估器
│   └── benchmarks.py     # 基准测试
├── experiments/          # 实验配置
└── docs/                # 文档
```

## 📚 文档

- **[English Documentation](docs/en/)** - 完整英文文档
- **[中文文档](docs/cn/)** - 完整中文文档  
- **[User Guide](docs/en/USER_GUIDE_EN.md)** - 如何使用自己的数据集（英文）
- **[API Reference](docs/en/API_REFERENCE_EN.md)** - 完整API文档（英文）
- **[用户指南](docs/cn/USER_GUIDE.md)** - 如何使用自己的数据集
- **[API参考](docs/cn/API_REFERENCE.md)** - 完整API文档

## 📚 详细文档

### 模型使用指南

#### 统计模型

**零膨胀泊松模型 (ZIP)**
```python
from models.baseline.zip_model import ZeroInflatedPoisson

model = ZeroInflatedPoisson()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**零膨胀负二项模型 (ZINB)**
```python
from models.baseline.zinb_model import ZeroInflatedNegativeBinomial

model = ZeroInflatedNegativeBinomial()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 深度学习模型

**ZIP-RNN模型**
```python
from models.zero_aware.zip_rnn import ZIPRNN
import torch

model = ZIPRNN(
    input_dim=1,
    hidden_dim=64,
    num_layers=2,
    seq_len=24,
    pred_len=6
)

# 训练循环
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(100):
    for batch in train_loader:
        input_seq, target_seq = batch
        predictions = model(input_seq)
        loss = model.compute_zip_loss(predictions, target_seq)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 评估指标说明

该工具包提供专门针对零膨胀数据的评估指标：

#### 标准预测指标
- **MSE/RMSE**: 均方误差和均方根误差
- **MAE**: 平均绝对误差  
- **MAPE**: 平均绝对百分比误差
- **R²**: 决定系数

#### 零膨胀特定指标
- **零值分类准确率**: 正确预测零/非零的比例
- **零值精确率/召回率**: 二分类性能指标
- **零值比例误差**: 预测和实际零值比例的差异
- **非零值性能**: 仅在非零值上的预测性能

#### 分布指标
- **KS统计量**: 预测和实际分布的差异
- **分位数误差**: 各分位数的预测误差
- **偏度/峰度误差**: 高阶矩的差异

### 基准测试

运行标准基准测试：

```python
from evaluation.benchmarks import StandardBenchmarks

# 准备模型字典
models = {
    'ZIP': ZeroInflatedPoisson(),
    'ZINB': ZeroInflatedNegativeBinomial(),
    # 添加更多模型...
}

# 运行快速基准测试
results = StandardBenchmarks.quick_benchmark(models)

# 或运行全面基准测试
results = StandardBenchmarks.comprehensive_benchmark(models)
```

## 🔧 高级用法

### 自定义零膨胀机制

```python
from generation.zero_mechanisms import ThresholdZeroInflation

# 创建自定义零膨胀机制
zi_mechanism = ThresholdZeroInflation(
    threshold_value=2.0,
    threshold_prob=0.8
)

# 应用到数据
zero_inflated_data = zi_mechanism.apply(original_data)
```

### 自定义评估指标

```python
from evaluation.metrics import ZeroInflatedMetrics

# 创建自定义评估器
evaluator = ZeroInflatedMetrics(zero_threshold=1e-6)

# 计算所有指标
all_metrics = evaluator(predictions, targets, return_components=True)

# 仅计算特定类型指标
forecasting_metrics = evaluator.compute_forecasting_metrics(predictions, targets)
zero_metrics = evaluator.compute_zero_inflation_metrics(predictions, targets)
```

### 批量实验

```python
from evaluation.evaluator import ComprehensiveEvaluation

# 准备多个模型
models = {
    'ZIP': create_zip_model,
    'ZINB': create_zinb_model,
    'RNN': create_rnn_model
}

# 运行完整评估
evaluator = ComprehensiveEvaluation()
results = evaluator.full_evaluation(
    models=models,
    X=X_data,
    y=y_data,
    test_split=0.2,
    perform_cv=True  # 执行交叉验证
)

# 生成报告
report = evaluator.generate_report(results)
print(report)
```

## 🎨 可视化

### 数据探索
```python
import matplotlib.pyplot as plt

# 零膨胀数据可视化
def plot_zero_inflation_analysis(data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 时间序列图
    axes[0, 0].plot(data)
    axes[0, 0].set_title('时间序列')
    
    # 零值分布
    zero_positions = np.where(data == 0)[0]
    axes[0, 1].hist(zero_positions, bins=50)
    axes[0, 1].set_title('零值位置分布')
    
    # 非零值直方图
    non_zero_data = data[data > 0]
    axes[1, 0].hist(non_zero_data, bins=30)
    axes[1, 0].set_title('非零值分布')
    
    # 零膨胀程度随时间变化
    window_size = len(data) // 20
    rolling_zero_ratio = pd.Series(data == 0).rolling(window_size).mean()
    axes[1, 1].plot(rolling_zero_ratio)
    axes[1, 1].set_title('滚动零值比例')
    
    plt.tight_layout()
    plt.show()

# 使用示例
plot_zero_inflation_analysis(your_data)
```

## 🤝 贡献指南

我们欢迎贡献！请按照以下步骤：

1. Fork该项目
2. 创建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个Pull Request

### 开发指南

- 所有新功能都应该包含测试
- 遵循现有的代码风格
- 更新相关文档
- 确保所有测试都通过

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢scikit-learn团队提供的机器学习基础设施
- 感谢PyTorch团队提供的深度学习框架
- 感谢所有为零膨胀建模研究做出贡献的研究人员

## 📞 联系方式

如果您有任何问题或建议，请：

- 创建一个Issue
- 发送邮件到 [your-email@example.com]
- 访问我们的文档网站

---

**开始使用零膨胀时间序列分析，让您的预测更加准确！** 🎯