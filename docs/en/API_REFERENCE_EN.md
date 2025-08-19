# API Reference Documentation

[English](API_REFERENCE_EN.md) | [中文](API_REFERENCE.md)

This document provides complete API reference for the Zero-Inflated Time Series Analysis Toolkit.

## 📋 Table of Contents

- [Data Module (data)](#data-module-data)
- [Models Module (models)](#models-module-models)
- [Evaluation Module (evaluation)](#evaluation-module-evaluation)
- [Generation Module (generation)](#generation-module-generation)

## Data Module (data)

### data.formatters

#### `StandardTimeSeriesFormat`

Standard time series data format class.

```python
class StandardTimeSeriesFormat:
    def __init__(self,
                 values: np.ndarray,
                 timestamps: Optional[np.ndarray] = None,
                 features: Optional[np.ndarray] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 zero_threshold: float = 1e-6,
                 frequency: Optional[str] = None)
```

**Parameters**:
- `values`: Time series values (1D or 2D array)
- `timestamps`: Optional timestamp array
- `features`: Optional additional feature array
- `metadata`: Optional metadata dictionary
- `zero_threshold`: Zero value threshold
- `frequency`: Time series frequency

**Methods**:

##### `get_zero_ratio() -> float`
Calculate the ratio of zero values.

**Returns**: Float between 0 and 1 representing the proportion of zero values.

##### `get_summary_stats() -> Dict[str, Any]`
Get summary statistics of the time series.

**Returns**: Dictionary containing length, shape, zero ratio, and other information.

##### `validate() -> Tuple[bool, List[str]]`
Validate the data format and quality.

**Returns**: Tuple of (is_valid, list_of_issues).

#### `DataFormatValidator`

Data format validation class.

```python
class DataFormatValidator:
    @staticmethod
    def validate_numpy_array(data: np.ndarray) -> Tuple[bool, List[str]]
    
    @staticmethod
    def validate_pandas_series(data: pd.Series) -> Tuple[bool, List[str]]
    
    @staticmethod
    def validate_pandas_dataframe(data: pd.DataFrame, 
                                  value_column: str) -> Tuple[bool, List[str]]
```

**Methods**:

##### `validate_numpy_array(data)`
Validate NumPy array format.

**Parameters**:
- `data`: NumPy array to validate

**Returns**: Tuple of (is_valid, list_of_issues).

##### `validate_pandas_series(data)`
Validate Pandas Series format.

**Parameters**:
- `data`: Pandas Series to validate

**Returns**: Tuple of (is_valid, list_of_issues).

##### `validate_pandas_dataframe(data, value_column)`
Validate Pandas DataFrame format.

**Parameters**:
- `data`: Pandas DataFrame to validate
- `value_column`: Name of the target value column

**Returns**: Tuple of (is_valid, list_of_issues).

#### `validate_zero_inflated_data(data, value_column=None) -> Tuple[bool, List[str]]`

Main validation function for zero-inflated data.

**Parameters**:
- `data`: Data to validate (NumPy array, Pandas Series, or DataFrame)
- `value_column`: Target column name (required for DataFrame)

**Returns**: Tuple of (is_valid, list_of_issues).

**Example**:
```python
from data.formatters import validate_zero_inflated_data

# For DataFrame
is_valid, issues = validate_zero_inflated_data(df, value_column='sales')

# For NumPy array
is_valid, issues = validate_zero_inflated_data(array)
```

#### `convert_to_standard_format(data, **kwargs) -> StandardTimeSeriesFormat`

Convert various data formats to standard internal format.

**Parameters**:
- `data`: Input data (various formats supported)
- `value_column`: Target column name (for DataFrame/CSV)
- `timestamp_column`: Timestamp column name (optional)
- `feature_columns`: List of feature column names (optional)

**Returns**: StandardTimeSeriesFormat object.

### data.loaders

#### `ZeroInflatedDataLoader`

Main data loader class for zero-inflated time series.

```python
class ZeroInflatedDataLoader:
    def __init__(self, random_state: int = 42)
```

**Methods**:

##### `load_and_prepare(data, sequence_length, prediction_horizon, **kwargs) -> Dict`

Load and prepare data for modeling.

**Parameters**:
- `data`: Input data (various formats)
- `sequence_length`: Input sequence length
- `prediction_horizon`: Prediction time horizon
- `test_split`: Test set ratio (default: 0.2)
- `val_split`: Validation set ratio (default: 0.1)
- `batch_size`: Batch size (default: 32)
- `normalize`: Whether to normalize data (default: True)
- `value_column`: Target column name (for DataFrame/CSV)
- `timestamp_column`: Timestamp column name (optional)
- `feature_columns`: Feature column names (optional)

**Returns**: Dictionary with keys:
- `train_loader`: Training data loader
- `val_loader`: Validation data loader  
- `test_loader`: Test data loader
- `scaler`: Data scaler object
- `metadata`: Dataset metadata

**Example**:
```python
from data.loaders import ZeroInflatedDataLoader

loader = ZeroInflatedDataLoader()
prepared_data = loader.load_and_prepare(
    data=df,
    sequence_length=24,
    prediction_horizon=6,
    test_split=0.2,
    batch_size=32,
    value_column='sales'
)

train_loader = prepared_data['train_loader']
```

##### `create_sequences(values, sequence_length, prediction_horizon) -> Tuple`

Create input-output sequences for time series modeling.

**Parameters**:
- `values`: Time series values
- `sequence_length`: Length of input sequences
- `prediction_horizon`: Length of output sequences

**Returns**: Tuple of (input_sequences, target_sequences).

#### `load_csv_data(file_path, **kwargs) -> StandardTimeSeriesFormat`

Convenient function to load CSV files.

**Parameters**:
- `file_path`: Path to CSV file
- `value_column`: Target column name
- `timestamp_column`: Timestamp column name (optional)
- `feature_columns`: Feature column names (optional)

**Returns**: StandardTimeSeriesFormat object.

### data.preprocessors

#### `ZeroInflatedPreprocessor`

Specialized preprocessor for zero-inflated data.

```python
class ZeroInflatedPreprocessor:
    def __init__(self, zero_threshold: float = 1e-6)
```

**Methods**:

##### `fit(X) -> 'ZeroInflatedPreprocessor'`
Fit the preprocessor to the data.

##### `transform(X) -> np.ndarray`
Transform the data.

##### `fit_transform(X) -> np.ndarray`
Fit and transform the data.

##### `inverse_transform(X) -> np.ndarray`
Inverse transform the data.

#### `TimeSeriesScaler`

Time series-aware scaler.

```python
class TimeSeriesScaler:
    def __init__(self, method: str = 'standard')
```

**Parameters**:
- `method`: Scaling method ('standard', 'minmax', 'robust')

## Models Module (models)

### models.baseline

#### `ZeroInflatedPoisson`

Zero-Inflated Poisson regression model.

```python
class ZeroInflatedPoisson:
    def __init__(self, max_iter: int = 100, tol: float = 1e-6)
```

**Methods**:

##### `fit(X, y) -> 'ZeroInflatedPoisson'`
Fit the ZIP model.

**Parameters**:
- `X`: Feature matrix (n_samples, n_features)
- `y`: Target values (n_samples,)

**Returns**: Fitted model.

##### `predict(X) -> np.ndarray`
Make predictions.

**Parameters**:
- `X`: Feature matrix

**Returns**: Predicted values.

##### `predict_proba(X) -> Tuple[np.ndarray, np.ndarray]`
Predict class probabilities.

**Returns**: Tuple of (zero_probabilities, count_probabilities).

##### `get_params() -> Dict`
Get model parameters.

#### `ZeroInflatedNegativeBinomial`

Zero-Inflated Negative Binomial regression model.

```python
class ZeroInflatedNegativeBinomial:
    def __init__(self, max_iter: int = 100, tol: float = 1e-6)
```

Similar interface to ZeroInflatedPoisson.

#### `TweedieGLM`

Tweedie Generalized Linear Model.

```python
class TweedieGLM:
    def __init__(self, power: float = 1.5, max_iter: int = 100)
```

**Parameters**:
- `power`: Tweedie power parameter (1 < power < 2)

#### `HurdleModel`

Two-part hurdle model.

```python
class HurdleModel:
    def __init__(self, binary_model=None, count_model=None)
```

### models.zero_aware

#### `ZIPRNN`

Zero-Inflated Poisson RNN model.

```python
class ZIPRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, 
                 num_layers: int, seq_len: int, pred_len: int)
```

**Methods**:

##### `forward(x) -> torch.Tensor`
Forward pass.

##### `compute_zip_loss(predictions, targets) -> torch.Tensor`
Compute ZIP loss.

#### `DualBranchNetwork`

Dual-branch network for separate zero/non-zero modeling.

```python
class DualBranchNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int, pred_len: int)
```

#### `WeightedLossTransformer`

Transformer with weighted loss for zero-inflated data.

```python
class WeightedLossTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int)
```

## Evaluation Module (evaluation)

### evaluation.metrics

#### `ZeroInflatedMetrics`

Comprehensive metrics for zero-inflated time series evaluation.

```python
class ZeroInflatedMetrics:
    def __init__(self, zero_threshold: float = 1e-6)
```

**Methods**:

##### `__call__(predictions, targets) -> Dict[str, float]`
Compute all metrics.

**Parameters**:
- `predictions`: Predicted values
- `targets`: True values

**Returns**: Dictionary of metric names and values.

**Metrics included**:
- `mse`: Mean Squared Error
- `mae`: Mean Absolute Error
- `rmse`: Root Mean Squared Error
- `mape`: Mean Absolute Percentage Error
- `r2`: R-squared
- `zero_classification_accuracy`: Zero/non-zero classification accuracy
- `zero_precision`: Precision for zero prediction
- `zero_recall`: Recall for zero prediction
- `zero_f1`: F1-score for zero prediction
- `nonzero_mse`: MSE on non-zero values only
- `nonzero_mae`: MAE on non-zero values only
- `nonzero_r2`: R-squared on non-zero values only
- `zero_ratio_error`: Difference in zero ratios

##### `compute_forecasting_metrics(predictions, targets) -> Dict[str, float]`
Compute standard forecasting metrics.

##### `compute_zero_inflation_metrics(predictions, targets) -> Dict[str, float]`
Compute zero-inflation specific metrics.

##### `compute_distribution_metrics(predictions, targets) -> Dict[str, float]`
Compute distribution comparison metrics.

### evaluation.evaluator

#### `ComprehensiveEvaluation`

Comprehensive evaluation framework.

```python
class ComprehensiveEvaluation:
    def __init__(self, metrics: Optional[ZeroInflatedMetrics] = None)
```

**Methods**:

##### `evaluate_model(model, X_test, y_test) -> Dict`
Evaluate a single model.

##### `compare_models(models, X_test, y_test) -> pd.DataFrame`
Compare multiple models.

##### `cross_validate(model, X, y, cv=5) -> Dict`
Perform cross-validation.

### evaluation.benchmarks

#### `StandardBenchmarks`

Standard benchmarking suite.

```python
class StandardBenchmarks:
    @staticmethod
    def quick_benchmark(models: Dict) -> pd.DataFrame
    
    @staticmethod
    def comprehensive_benchmark(models: Dict) -> Dict
```

## Generation Module (generation)

### generation.zero_mechanisms

#### `ThresholdZeroInflation`

Threshold-based zero inflation mechanism.

```python
class ThresholdZeroInflation:
    def __init__(self, threshold_value: float, threshold_prob: float)
```

**Methods**:

##### `apply(data) -> np.ndarray`
Apply zero inflation to data.

#### `MixtureZeroInflation`

Mixture distribution zero inflation.

```python
class MixtureZeroInflation:
    def __init__(self, zero_prob: float)
```

#### `TweedieZeroInflation`

Natural zero inflation based on Tweedie distribution.

```python
class TweedieZeroInflation:
    def __init__(self, power: float = 1.5, mu: float = 1.0)
```

### generation.inject_zeros

#### `inject_zeros_threshold(data, threshold, prob) -> np.ndarray`

Inject zeros based on threshold.

#### `inject_zeros_random(data, zero_ratio) -> np.ndarray`

Inject zeros randomly.

#### `inject_zeros_pattern(data, pattern) -> np.ndarray`

Inject zeros following a pattern.

## Usage Examples

### Basic Usage

```python
# Data loading and preparation
from data.loaders import ZeroInflatedDataLoader
from data.formatters import validate_zero_inflated_data

# Validate data
is_valid, issues = validate_zero_inflated_data(your_data, value_column='target')

# Load and prepare
loader = ZeroInflatedDataLoader()
data = loader.load_and_prepare(
    data=your_data,
    sequence_length=24,
    prediction_horizon=6,
    value_column='target'
)

# Model training
from models.baseline.zip_model import ZeroInflatedPoisson

model = ZeroInflatedPoisson()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluation
from evaluation.metrics import ZeroInflatedMetrics

evaluator = ZeroInflatedMetrics()
metrics = evaluator(predictions, y_test)
print(f"MSE: {metrics['mse']:.4f}")
print(f"Zero Accuracy: {metrics['zero_classification_accuracy']:.3f}")
```

### Advanced Usage

```python
# Model comparison
from evaluation.evaluator import ComprehensiveEvaluation
from models.baseline import ZeroInflatedPoisson, HurdleModel

models = {
    'ZIP': ZeroInflatedPoisson(),
    'Hurdle': HurdleModel()
}

evaluator = ComprehensiveEvaluation()
results = evaluator.compare_models(models, X_test, y_test)
print(results)

# Benchmarking
from evaluation.benchmarks import StandardBenchmarks

benchmark_results = StandardBenchmarks.comprehensive_benchmark(models)
```

This API reference provides comprehensive documentation for all public interfaces in the Zero-Inflated Time Series Analysis Toolkit. For usage examples and tutorials, refer to the User Guide and example notebooks.