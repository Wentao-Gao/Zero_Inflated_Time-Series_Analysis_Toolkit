# English Documentation

Welcome to the English documentation for the Zero-Inflated Time Series Analysis Toolkit.

## 📚 Documentation Contents

### Core Guides
- **[User Guide](USER_GUIDE_EN.md)** - Complete guide on preparing and using your own datasets
- **[API Reference](API_REFERENCE_EN.md)** - Comprehensive API documentation for all modules

### Getting Started
1. **Data Preparation**: Learn how to format your data correctly
2. **Model Selection**: Choose the right model for your use case
3. **Training**: Train models on your zero-inflated time series
4. **Evaluation**: Assess model performance with specialized metrics

## 🚀 Quick Navigation

### For New Users
Start with the [User Guide](USER_GUIDE_EN.md) to understand:
- Data format requirements
- Supported data types
- Basic workflow examples
- Common issues and solutions

### For Developers
Refer to the [API Reference](API_REFERENCE_EN.md) for:
- Detailed function signatures
- Parameter descriptions
- Return value specifications
- Code examples for each module

## 📊 Key Topics Covered

### Data Handling
- Multiple input formats (NumPy, Pandas, CSV)
- Data validation and quality checks
- Automatic format conversion
- Time series preprocessing

### Models Available
- **Statistical Models**: ZIP, ZINB, Tweedie GLM, Hurdle
- **Deep Learning Models**: ZIP-RNN, Dual Branch Network, Transformers
- **Zero-Inflation Mechanisms**: Threshold, Mixture, Tweedie

### Evaluation Metrics
- Standard forecasting metrics (MSE, MAE, RMSE)
- Zero-inflation specific metrics
- Distribution comparison metrics
- Cross-validation strategies

## 💡 Usage Examples

### Basic Example
```python
from data.loaders import ZeroInflatedDataLoader
from models.baseline.zip_model import ZeroInflatedPoisson

# Load data
loader = ZeroInflatedDataLoader()
data = loader.load_and_prepare(your_data, sequence_length=24)

# Train model
model = ZeroInflatedPoisson()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Advanced Example
```python
from evaluation.benchmarks import StandardBenchmarks

# Compare multiple models
models = {
    'ZIP': ZeroInflatedPoisson(),
    'ZINB': ZeroInflatedNegativeBinomial(),
    'Hurdle': HurdleModel()
}

results = StandardBenchmarks.comprehensive_benchmark(models)
```

## 🔍 Finding What You Need

| If you want to... | Go to... |
|------------------|----------|
| Understand data requirements | [User Guide - Data Format Requirements](USER_GUIDE_EN.md#data-format-requirements) |
| Learn model APIs | [API Reference - Models Module](API_REFERENCE_EN.md#models-module-models) |
| See evaluation metrics | [API Reference - Evaluation Module](API_REFERENCE_EN.md#evaluation-module-evaluation) |
| Handle data loading | [API Reference - Data Module](API_REFERENCE_EN.md#data-module-data) |
| Generate zero-inflated data | [API Reference - Generation Module](API_REFERENCE_EN.md#generation-module-generation) |

## 🆘 Need Help?

- **Common Issues**: Check the [troubleshooting section](USER_GUIDE_EN.md#common-issues-and-solutions)
- **Best Practices**: Review the [best practices guide](USER_GUIDE_EN.md#best-practices)
- **Examples**: Browse the [examples directory](../examples/)
- **Tutorials**: Follow step-by-step [tutorials](../tutorials/)

## 🌐 Other Languages

- [中文文档](../cn/README.md) - Chinese documentation