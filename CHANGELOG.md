# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-19

### Added
- Initial release of Zero-Inflated Time Series Analysis Toolkit
- **Data Processing Module**:
  - Multi-format data support (NumPy arrays, Pandas Series/DataFrame, CSV files)
  - Automatic data validation and format conversion
  - Specialized preprocessing for zero-inflated data
  - PyTorch-integrated data loaders with time series awareness

- **Statistical Models**:
  - Zero-Inflated Poisson (ZIP) regression model
  - Zero-Inflated Negative Binomial (ZINB) regression model
  - Tweedie Generalized Linear Model (GLM)
  - Hurdle model implementation

- **Deep Learning Models**:
  - ZIP-RNN: RNN with zero-inflated Poisson distribution
  - Dual Branch Network: Separate modeling for zeros and non-zeros
  - Weighted Loss Transformer: Transformer with specialized loss weighting
  - Enhanced Tweedie Transformer: Advanced Tweedie distribution modeling

- **Zero-Inflation Mechanisms**:
  - Threshold-based zero inflation
  - Mixture distribution zero inflation
  - Tweedie-based natural zero inflation
  - Hurdle mechanism implementation

- **Evaluation System**:
  - Specialized metrics for zero-inflated time series
  - Zero classification accuracy and precision/recall
  - Distribution-based evaluation metrics
  - Comprehensive benchmarking suite
  - Time series-aware cross validation

- **Documentation**:
  - Comprehensive English and Chinese README files
  - Detailed API reference documentation
  - User guide with practical examples
  - Tutorial notebooks and example scripts

- **Package Infrastructure**:
  - Standard Python packaging with setup.py and pyproject.toml
  - Comprehensive dependency management
  - MIT license
  - Automated testing framework setup

### Features
- Support for time series with 5-80% zero inflation
- Automatic detection and handling of zero-inflation patterns
- Batch processing capabilities for large datasets
- Integration with popular ML libraries (scikit-learn, PyTorch)
- Visualization utilities for zero-inflated data analysis
- Configurable preprocessing pipelines
- Model comparison and benchmarking tools

### Technical Details
- Python 3.8+ compatibility
- NumPy, Pandas, and PyTorch integration
- Memory-efficient data processing
- GPU acceleration support for deep learning models
- Extensible architecture for custom models and metrics

### Applications
- Sales forecasting with intermittent demand
- Web traffic analysis with zero-visit periods
- Equipment failure prediction
- Ecological occurrence modeling
- Medical event prediction
- Quality control and defect analysis

[1.0.0]: https://github.com/your-username/zero-inflated-comprehensive/releases/tag/v1.0.0