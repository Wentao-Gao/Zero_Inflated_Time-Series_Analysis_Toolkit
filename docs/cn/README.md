# 中文文档

欢迎使用零膨胀时间序列分析工具包的中文文档。

## 📚 文档内容

### 核心指南
- **[用户指南](USER_GUIDE.md)** - 如何准备和使用自己数据集的完整指南
- **[API参考](API_REFERENCE.md)** - 所有模块的全面API文档

### 入门指南
1. **数据准备**: 学习如何正确格式化数据
2. **模型选择**: 为您的用例选择合适的模型
3. **训练**: 在零膨胀时间序列上训练模型
4. **评估**: 使用专门指标评估模型性能

## 🚀 快速导航

### 新用户
从[用户指南](USER_GUIDE.md)开始了解：
- 数据格式要求
- 支持的数据类型
- 基本工作流程示例
- 常见问题和解决方案

### 开发者
参考[API参考](API_REFERENCE.md)获取：
- 详细的函数签名
- 参数描述
- 返回值规范
- 每个模块的代码示例

## 📊 涵盖的关键主题

### 数据处理
- 多种输入格式（NumPy、Pandas、CSV）
- 数据验证和质量检查
- 自动格式转换
- 时间序列预处理

### 可用模型
- **统计模型**: ZIP、ZINB、Tweedie GLM、跨栏模型
- **深度学习模型**: ZIP-RNN、双分支网络、Transformer
- **零膨胀机制**: 阈值、混合、Tweedie

### 评估指标
- 标准预测指标（MSE、MAE、RMSE）
- 零膨胀特定指标
- 分布比较指标
- 交叉验证策略

## 💡 使用示例

### 基础示例
```python
from data.loaders import ZeroInflatedDataLoader
from models.baseline.zip_model import ZeroInflatedPoisson

# 加载数据
loader = ZeroInflatedDataLoader()
data = loader.load_and_prepare(your_data, sequence_length=24)

# 训练模型
model = ZeroInflatedPoisson()
model.fit(X_train, y_train)

# 进行预测
predictions = model.predict(X_test)
```

### 高级示例
```python
from evaluation.benchmarks import StandardBenchmarks

# 比较多个模型
models = {
    'ZIP': ZeroInflatedPoisson(),
    'ZINB': ZeroInflatedNegativeBinomial(),
    'Hurdle': HurdleModel()
}

results = StandardBenchmarks.comprehensive_benchmark(models)
```

## 🔍 查找所需信息

| 如果您想要... | 请查看... |
|--------------|----------|
| 了解数据要求 | [用户指南 - 数据格式要求](USER_GUIDE.md#数据格式要求) |
| 学习模型API | [API参考 - 模型模块](API_REFERENCE.md#模型模块-models) |
| 查看评估指标 | [API参考 - 评估模块](API_REFERENCE.md#评估模块-evaluation) |
| 处理数据加载 | [API参考 - 数据模块](API_REFERENCE.md#数据模块-data) |
| 生成零膨胀数据 | [API参考 - 生成模块](API_REFERENCE.md#生成模块-generation) |

## 🆘 需要帮助？

- **常见问题**: 查看[故障排除部分](USER_GUIDE.md#常见问题和解决方案)
- **最佳实践**: 查看[最佳实践指南](USER_GUIDE.md#最佳实践)
- **示例**: 浏览[示例目录](../examples/)
- **教程**: 跟随分步[教程](../tutorials/)

## 🌐 其他语言

- [English Documentation](../en/README.md) - 英文文档