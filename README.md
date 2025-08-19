# 🚀 Zero-Inflated Time Series Analysis Toolkit

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)
![Build](https://img.shields.io/badge/build-passing-success.svg)

**🎯 A comprehensive toolkit for zero-inflated time series analysis**

[🇨🇳 中文说明](README_CN.md) | [🇺🇸 English](README.md) | [📚 Documentation](docs/) | [🔬 Examples](examples/)

---

*✨ From data preprocessing to model training and evaluation - everything you need for zero-inflated time series! ✨*

</div>

## 🌟 Key Features

### 📊 **Data Processing**
- 🔄 **Multi-format Support**: NumPy arrays, Pandas Series/DataFrame, CSV files
- ✅ **Automatic Validation**: Data format and quality checks
- 🔀 **Smart Conversion**: Automatic conversion to standard time series format  
- 🎯 **Specialized Preprocessing**: Tools specifically designed for zero-inflated data

### 🧬 **Zero-Inflation Mechanisms**
- 📏 **Threshold Mechanism**: Threshold-based zero inflation
- 🎭 **Mixture Mechanism**: Mixture distribution zero inflation
- 🌊 **Tweedie Mechanism**: Natural zero inflation based on Tweedie distribution
- 🚪 **Hurdle Mechanism**: Two-stage hurdle model

### 📈 **Statistical Models**
- 🎲 **ZIP** *(Zero-Inflated Poisson)*: Zero-inflated Poisson regression
- 🎯 **ZINB** *(Zero-Inflated Negative Binomial)*: Zero-inflated negative binomial regression
- 🌊 **Tweedie GLM**: Tweedie generalized linear model
- 🚪 **Hurdle Model**: Two-stage hurdle model

### 🧠 **Deep Learning Models**
- 🔗 **ZIP-RNN**: Neural network combining RNN and zero-inflated Poisson distribution
- 🌿 **Dual Branch Network**: Dual-branch network modeling zeros and non-zeros separately  
- ⚖️ **Weighted Loss Transformer**: Transformer with weighted loss
- ⚡ **Enhanced Tweedie Transformer**: Enhanced Tweedie Transformer

### 🎯 **Evaluation System**
- 📊 **Specialized Metrics**: Evaluation metrics designed specifically for zero-inflated data
- 🔄 **Cross Validation**: Time series-aware cross validation
- 🏆 **Benchmarking**: Standardized benchmark suite
- 📋 **Model Comparison**: Automated model comparison and reporting

---

## 🚀 Quick Start

### 📦 Installation

```bash
# 🔽 Install from source
git clone https://github.com/your-username/zero-inflated-comprehensive.git
cd zero-inflated-comprehensive
pip install -r requirements.txt
pip install -e .

# 🔧 Or install dependencies manually
pip install numpy pandas scikit-learn torch scipy matplotlib
```

### 💡 Basic Usage Example

```python
import numpy as np
import pandas as pd
from data.loaders import ZeroInflatedDataLoader
from models.baseline.zip_model import ZeroInflatedPoisson
from evaluation.metrics import ZeroInflatedMetrics

# 🔹 1. Prepare data - supports multiple formats
# 📊 Method 1: NumPy array
data = np.array([1.2, 0.0, 2.1, 0.0, 1.8, 3.2, 0.0, 2.5])

# 📋 Method 2: Pandas DataFrame
data = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
    'value': [...],      # 🎯 Your time series data
    'feature1': [...],   # 🔧 Optional additional features
    'feature2': [...]    # 🔧 Optional additional features
})

# 📁 Method 3: CSV file
# data = '/path/to/your/data.csv'

# 🔹 2. Load and prepare data
loader = ZeroInflatedDataLoader()
prepared_data = loader.load_and_prepare(
    data=data,
    sequence_length=24,      # 📏 Input sequence length
    prediction_horizon=6,    # 🔮 Prediction time horizon
    test_split=0.2,         # ✂️ Test split ratio
    batch_size=32,          # 📦 Batch size
    value_column='value',   # 🎯 Target column name (for DataFrame/CSV)
    normalize=True          # 📊 Whether to normalize
)

# 🔹 3. Get data loaders
train_loader = prepared_data['train_loader']
val_loader = prepared_data['val_loader']
test_loader = prepared_data['test_loader']

# 🔹 4. Train model
model = ZeroInflatedPoisson()
# 🏋️ Train the model...

# 🔹 5. Evaluate model
evaluator = ZeroInflatedMetrics()
metrics = evaluator(predictions, targets)
print(f"✨ Zero classification accuracy: {metrics['zero_classification_accuracy']:.3f}")
print(f"📊 Overall MSE: {metrics['mse']:.6f}")
print(f"🎯 Non-zero R²: {metrics['nonzero_r2']:.3f}")
```

---

## 📋 Data Format Requirements

### 🔄 **Supported Data Formats**

#### 1️⃣ **NumPy Array Format**
```python
# 📈 1D array: time series values
data = np.array([1.2, 0.0, 2.1, 0.0, 1.8, 3.2, 0.0, 2.5])
```

#### 2️⃣ **Pandas Series Format**
```python
# 📅 Series with time index
data = pd.Series(
    [1.2, 0.0, 2.1, 0.0, 1.8, 3.2, 0.0, 2.5], 
    index=pd.date_range('2023-01-01', periods=8, freq='D'),
    name='value'
)
```

#### 3️⃣ **Pandas DataFrame Format**
```python
data = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
    'value': [...],        # 🎯 Main time series values (required)
    'temperature': [...],  # 🌡️ Additional feature 1 (optional)
    'humidity': [...],     # 💧 Additional feature 2 (optional)  
    'is_weekend': [...]    # 📅 Additional feature 3 (optional)
})
```

#### 4️⃣ **CSV File Format**
```csv
timestamp,value,temperature,humidity,is_weekend
2023-01-01 00:00:00,1.2,12.1,65.5,0
2023-01-01 01:00:00,0.0,11.8,66.2,0
2023-01-01 02:00:00,2.1,13.2,64.1,0
2023-01-01 03:00:00,0.0,12.5,65.8,0
...
```

### 📌 **Data Requirements**

<table>
<tr>
<td><strong>✅ Required</strong></td>
<td>
• 🔢 Numeric time series data (float or int)<br>
• ➕ Non-negative values (zero-inflated data is typically count/measurement data)<br>
• 🚫 No missing values (NaN) or infinite values<br>
• 📊 At least 50 observations (100+ recommended for reliable modeling)
</td>
</tr>
<tr>
<td><strong>🔧 Optional</strong></td>
<td>
• 📅 Timestamp column or index<br>
• 🔧 Additional feature columns<br>
• ⏰ Regular time intervals
</td>
</tr>
<tr>
<td><strong>🎯 Zero Handling</strong></td>
<td>
• 0️⃣ Zero values are expected and meaningful<br>
• 📈 Zero ratio can range from 5% to 80%<br>
• 🤖 Toolkit automatically detects and handles zero-inflation levels
</td>
</tr>
</table>

### 🔍 **Data Validation**

Use the built-in validator to check your data:

```python
from data.formatters import validate_zero_inflated_data

# 🔍 Validate data format
is_valid, issues = validate_zero_inflated_data(
    your_data, 
    value_column='value'  # For DataFrame/CSV
)

if is_valid:
    print("✅ Data format is valid")
else:
    print("❌ Data format issues:")
    for issue in issues:
        print(f"  ⚠️ {issue}")
```

---

## 🎯 Application Scenarios

This toolkit is particularly suitable for:

### 💼 **Business Applications**
- 🛒 **Sales Forecasting**: Product sales prediction (many periods with zero sales)
- 👥 **Customer Behavior**: User activity prediction (users may be inactive during certain periods)  
- 📊 **Demand Forecasting**: Service or product demand prediction

### ⚙️ **Engineering Applications**
- 🔧 **Fault Detection**: Equipment failure count prediction
- 🎯 **Quality Control**: Defect count prediction
- 🔧 **Maintenance Planning**: Maintenance demand prediction

### 🔬 **Scientific Research**
- 🌿 **Ecology**: Species occurrence count prediction
- 🏥 **Medicine**: Disease incidence count prediction
- 👥 **Social Sciences**: Event occurrence count prediction

### 🌐 **Internet Applications**
- 🌐 **Website Traffic**: Traffic prediction (some periods may have zero visits)
- 🖱️ **Ad Clicks**: Click rate prediction  
- 📖 **Content Consumption**: View or read count prediction

---

## 🏗️ Project Structure

```
📁 zero_inflated_comprehensive/
├── 📂 data/                    # 🔄 Data processing module
│   ├── 📝 formatters.py       # 🔄 Data format validation and conversion
│   ├── 📥 loaders.py         # 📥 Data loaders
│   └── 🔧 preprocessors.py   # 🔧 Data preprocessors
├── 📂 generation/             # 🎲 Data generation module
│   ├── 💉 inject_zeros.py    # 💉 Zero inflation injection functions
│   └── 🧬 zero_mechanisms.py # 🧬 Zero inflation mechanisms  
├── 📂 models/                 # 🤖 Model module
│   ├── 📂 baseline/          # 📊 Baseline statistical models
│   │   ├── 🎲 zip_model.py   # 🎲 ZIP model
│   │   ├── 🎯 zinb_model.py  # 🎯 ZINB model
│   │   ├── 🌊 tweedie_glm.py # 🌊 Tweedie GLM
│   │   └── 🚪 hurdle_model.py # 🚪 Hurdle model
│   ├── 📂 zero_aware/        # 🧠 Zero-aware deep learning models
│   │   ├── 🔗 zip_rnn.py     # 🔗 ZIP-RNN
│   │   ├── 🌿 dual_branch_network.py # 🌿 Dual-branch network
│   │   ├── ⚖️ weighted_loss_transformer.py # ⚖️ Weighted Transformer
│   │   └── ⚡ tweedie_transformer.py # ⚡ Tweedie Transformer
│   └── 📂 losses/            # 💥 Loss functions
│       ├── 🌊 tweedie_loss.py
│       └── 🎯 zero_aware_losses.py
├── 📂 evaluation/            # 📊 Evaluation module
│   ├── 📊 metrics.py        # 📊 Evaluation metrics
│   ├── 🔍 evaluator.py      # 🔍 Evaluator
│   └── 🏆 benchmarks.py     # 🏆 Benchmarks
├── 📂 experiments/          # 🧪 Experiment configurations
└── 📂 docs/                # 📚 Documentation
```

---

## 📚 Model Usage Guide

### 📊 **Statistical Models**

**🎲 Zero-Inflated Poisson Model (ZIP)**
```python
from models.baseline.zip_model import ZeroInflatedPoisson

model = ZeroInflatedPoisson()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**🎯 Zero-Inflated Negative Binomial Model (ZINB)**  
```python
from models.baseline.zinb_model import ZeroInflatedNegativeBinomial

model = ZeroInflatedNegativeBinomial()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 🧠 **Deep Learning Models**

**🔗 ZIP-RNN Model**
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

# 🏋️ Training loop
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

---

## 📊 Evaluation Metrics

The toolkit provides evaluation metrics specifically designed for zero-inflated data:

### 📈 **Standard Prediction Metrics**
- 📊 **MSE/RMSE**: Mean squared error and root mean squared error
- 📉 **MAE**: Mean absolute error
- 📊 **MAPE**: Mean absolute percentage error  
- 📈 **R²**: Coefficient of determination

### 🎯 **Zero-Inflation Specific Metrics**
- ✅ **Zero Classification Accuracy**: Proportion of correctly predicted zero/non-zero
- 🎯 **Zero Precision/Recall**: Binary classification performance metrics
- 📊 **Zero Ratio Error**: Difference between predicted and actual zero ratios
- 🎯 **Non-zero Performance**: Prediction performance on non-zero values only

### 📊 **Distribution Metrics**
- 📊 **KS Statistic**: Difference between predicted and actual distributions
- 📈 **Quantile Errors**: Prediction errors at various quantiles
- 📊 **Skewness/Kurtosis Errors**: Higher-order moment differences

---

## 🔧 Advanced Usage

### 🧬 **Custom Zero-Inflation Mechanisms**

```python
from generation.zero_mechanisms import ThresholdZeroInflation

# 🔧 Create custom zero-inflation mechanism
zi_mechanism = ThresholdZeroInflation(
    threshold_value=2.0,
    threshold_prob=0.8
)

# 💉 Apply to data
zero_inflated_data = zi_mechanism.apply(original_data)
```

### 🏆 **Benchmarking**

Run standard benchmarks:

```python
from evaluation.benchmarks import StandardBenchmarks

# 🗂️ Prepare model dictionary
models = {
    '🎲 ZIP': ZeroInflatedPoisson(),
    '🎯 ZINB': ZeroInflatedNegativeBinomial(),
    # ➕ Add more models...
}

# ⚡ Run quick benchmark
results = StandardBenchmarks.quick_benchmark(models)

# 🔬 Or run comprehensive benchmark
results = StandardBenchmarks.comprehensive_benchmark(models)
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. 🍴 **Fork** the project
2. 🌿 **Create** your feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. 🚀 **Push** to the branch (`git push origin feature/AmazingFeature`)
5. 📥 **Open** a Pull Request

### 📋 **Development Guidelines**

- ✅ All new features should include tests
- 🎨 Follow existing code style  
- 📝 Update relevant documentation
- 🧪 Ensure all tests pass

---

<div align="center">

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- 🙏 Thanks to the **scikit-learn** team for machine learning infrastructure
- 🙏 Thanks to the **PyTorch** team for the deep learning framework  
- 🙏 Thanks to all researchers contributing to **zero-inflated modeling** research

---

## 📞 Contact & Support

🚀 **Get Started**: [Quick Start Guide](docs/quickstart.md)  
📚 **Documentation**: [Full Documentation](docs/)  
🐛 **Issues**: [Create an Issue](issues/)  
💬 **Discussions**: [Join the Discussion](discussions/)  

---

<div align="center">

**🎯 Start using zero-inflated time series analysis for more accurate predictions! 🚀**

![Stars](https://img.shields.io/github/stars/your-username/zero-inflated-comprehensive?style=social)
![Forks](https://img.shields.io/github/forks/your-username/zero-inflated-comprehensive?style=social)
![Issues](https://img.shields.io/github/issues/your-username/zero-inflated-comprehensive)
![Contributors](https://img.shields.io/github/contributors/your-username/zero-inflated-comprehensive)

</div>

</div>
