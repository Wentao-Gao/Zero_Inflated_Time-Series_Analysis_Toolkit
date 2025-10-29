
---

# 🚀 Zero-Inflated Time Series Analysis Toolkit

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)
![Build](https://img.shields.io/badge/build-passing-success.svg)

**A comprehensive toolkit for zero-inflated time series analysis**

[🇨🇳 中文说明](README_CN.md) | [📚 Documentation](docs/) | [🧪 Examples](examples/)

---

*From data preprocessing to model training and evaluation — everything you need for zero-inflated time series.*

</div>

---

## 🌟 Key Features

### 📊 Data Processing

* **Multi-format support**: NumPy, Pandas, CSV
* **Automatic validation** and quality checks
* **Zero-aware preprocessing** with normalization and handling of extreme zeros

### 🧬 Zero-Inflation Mechanisms

* Threshold-based zero inflation
* Mixture-distribution mechanism
* Tweedie-based natural zero inflation
* Hurdle (two-stage) mechanism

### 📈 Statistical Models

* Zero-Inflated Poisson (ZIP)
* Zero-Inflated Negative Binomial (ZINB)
* Tweedie GLM
* Hurdle Model

### 🤖 Deep Learning Models

* **ZIP-RNN** — neural zero-inflated Poisson network
* **Dual-Branch Network** — separate modeling of zero and non-zero parts
* **Weighted-Loss Transformer** — Transformer with zero-aware weighting
* **Enhanced Tweedie Transformer** — Tweedie-based probabilistic Transformer

### 🎯 Evaluation System

* Metrics designed for zero-inflated data
* Time-series cross-validation
* Automated benchmarking and comparison

---

## ⚡ Quick Start

### 📦 Installation

```bash
git clone https://github.com/your-username/zero-inflated-comprehensive.git
cd zero-inflated-comprehensive
pip install -r requirements.txt
pip install -e .
```

Or manually install dependencies:

```bash
pip install numpy pandas scikit-learn torch scipy matplotlib
```

### 💡 Basic Example

```python
import numpy as np
import pandas as pd
from data.loaders import ZeroInflatedDataLoader
from models.baseline.zip_model import ZeroInflatedPoisson
from evaluation.metrics import ZeroInflatedMetrics

# 1️⃣ Prepare data
data = np.array([1.2, 0.0, 2.1, 0.0, 1.8, 3.2, 0.0, 2.5])

loader = ZeroInflatedDataLoader()
prepared = loader.load_and_prepare(
    data=data,
    sequence_length=24,
    prediction_horizon=6,
    test_split=0.2,
    batch_size=32,
    normalize=True
)

train_loader = prepared['train_loader']
val_loader = prepared['val_loader']
test_loader = prepared['test_loader']

# 2️⃣ Train model
model = ZeroInflatedPoisson()

# 3️⃣ Evaluate
evaluator = ZeroInflatedMetrics()
metrics = evaluator(predictions, targets)
print(f"Zero accuracy: {metrics['zero_classification_accuracy']:.3f}")
print(f"MSE: {metrics['mse']:.6f}")
print(f"Non-zero R²: {metrics['nonzero_r2']:.3f}")
```

---

## 📁 Project Structure

```
zero_inflated_comprehensive/
├── data/
│   ├── formatters.py
│   ├── loaders.py
│   └── preprocessors.py
├── generation/
│   ├── inject_zeros.py
│   └── zero_mechanisms.py
├── models/
│   ├── baseline/
│   │   ├── zip_model.py
│   │   ├── zinb_model.py
│   │   ├── tweedie_glm.py
│   │   └── hurdle_model.py
│   ├── zero_aware/
│   │   ├── zip_rnn.py
│   │   ├── dual_branch_network.py
│   │   ├── weighted_loss_transformer.py
│   │   └── tweedie_transformer.py
│   └── losses/
│       ├── tweedie_loss.py
│       └── zero_aware_losses.py
├── evaluation/
│   ├── metrics.py
│   ├── evaluator.py
│   └── benchmarks.py
└── docs/
```

---

## 🔍 Evaluation Metrics

| Category      | Metric                                                     | Description                     |
| ------------- | ---------------------------------------------------------- | ------------------------------- |
| Standard      | MSE / RMSE, MAE, MAPE, R²                                  | Conventional regression metrics |
| Zero-specific | Zero accuracy, Zero precision/recall, Zero ratio error     | Evaluate zero detection quality |
| Distribution  | KS statistic, Quantile error, Skewness/Kurtosis difference | Capture distribution similarity |

---

## 💼 Typical Applications

| Domain      | Example                                         |
| ----------- | ----------------------------------------------- |
| Business    | Sales forecasting, inactive customer prediction |
| Engineering | Equipment faults, maintenance demand            |
| Science     | Species count, disease incidence                |
| Internet    | Website traffic, ad click rate                  |

---

## 🤝 Contributing

1. Fork this repository
2. Create a branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push and open a Pull Request

**Guidelines**

* Add unit tests for new features
* Keep consistent code style
* Update docs and verify all tests pass

---

## 📄 License

Released under the **MIT License** — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

* scikit-learn for ML infrastructure
* PyTorch for deep learning backend
* All contributors to zero-inflated modeling research

---

## 📬 Contact

* [Quick Start Guide](docs/quickstart.md)
* [Documentation](docs/)
* [Report Issues](issues/)
* [Join Discussions](discussions/)

---

<div align="center">

**🌦️ Start exploring zero-inflated time series for better predictions!**

![Stars](https://img.shields.io/github/stars/your-username/zero-inflated-comprehensive?style=social)
![Forks](https://img.shields.io/github/forks/your-username/zero-inflated-comprehensive?style=social)
![Issues](https://img.shields.io/github/issues/your-username/zero-inflated-comprehensive)
![Contributors](https://img.shields.io/github/contributors/your-username/zero-inflated-comprehensive)

</div>

---

