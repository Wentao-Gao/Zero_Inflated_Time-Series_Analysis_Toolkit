# User Guide: How to Use Your Own Dataset

[English](USER_GUIDE_EN.md) | [中文](USER_GUIDE.md)

This guide provides detailed instructions on how to prepare and use your own datasets for zero-inflated time series analysis.

## 📋 Table of Contents

1. [Data Format Requirements](#data-format-requirements)
2. [Preparing Your Data](#preparing-your-data)
3. [Data Validation and Conversion](#data-validation-and-conversion)
4. [Complete Workflow Example](#complete-workflow-example)
5. [Common Issues and Solutions](#common-issues-and-solutions)
6. [Best Practices](#best-practices)

## 📊 Data Format Requirements

### Basic Requirements

Your data must satisfy the following basic requirements:

| Requirement | Description | Example |
|-------------|-------------|---------|
| **Data Type** | Numeric (integer or float) | `1.2, 0.0, 3.5, 0.0` |
| **Non-negative** | All values ≥ 0 | ✅ `[0, 1.2, 3.0]` ❌ `[0, -1.2, 3.0]` |
| **No Missing Values** | No NaN or null values | ✅ `[0, 1.2, 3.0]` ❌ `[0, NaN, 3.0]` |
| **Minimum Length** | At least 50 observations | 100+ recommended for reliable modeling |
| **Meaningful Zeros** | Zero values should be meaningful | Sales=0, visits=0, etc. |

### Zero-Inflation Characteristics

Your data should exhibit zero-inflation characteristics:

- **Zero Ratio**: Typically between 10% - 80%
- **Zeros Are Not Errors**: Zero values represent genuine "no event" states
- **Zero Patterns**: Zeros may have temporal patterns (e.g., nighttime, weekends)

### Time Series Characteristics

- **Temporal Order**: Data should be arranged in chronological order
- **Regular Intervals**: Preferably regular time intervals (hourly, daily, weekly, etc.)
- **Sufficient History**: Enough historical data to learn patterns

## 🔧 Preparing Your Data

### Format 1: NumPy Array

Simplest format for univariate time series:

```python
import numpy as np

# Example: Daily sales data for a store
daily_sales = np.array([
    12.5, 0.0, 15.2, 0.0, 8.7, 23.1, 0.0,  # Week 1
    18.3, 0.0, 21.4, 0.0, 9.5, 25.8, 0.0,  # Week 2
    # ... more data
])

# Check zero ratio
zero_ratio = np.mean(daily_sales == 0)
print(f"Zero ratio: {zero_ratio:.3f}")  # Should be between 0.1-0.8
```

### Format 2: Pandas Series

Suitable for univariate time series with time index:

```python
import pandas as pd
import numpy as np

# Create time index
dates = pd.date_range('2023-01-01', periods=365, freq='D')

# Create Series
sales_data = pd.Series(
    data=your_sales_values,  # Your data
    index=dates,
    name='daily_sales'
)

print(f"Data length: {len(sales_data)}")
print(f"Zero ratio: {(sales_data == 0).mean():.3f}")
print(f"Time range: {sales_data.index.min()} to {sales_data.index.max()}")
```

### Format 3: Pandas DataFrame (Recommended)

Most flexible format supporting multiple features:

```python
import pandas as pd
import numpy as np

# Create DataFrame
data = pd.DataFrame({
    # Required: timestamp column
    'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
    
    # Required: target variable (zero-inflated time series to predict)
    'sales': your_target_values,  # Main prediction target
    
    # Optional: additional features
    'temperature': your_temperature_data,  # Weather feature
    'is_weekend': your_weekend_indicator,  # Categorical feature
    'hour_of_day': range(24) * (1000 // 24 + 1),  # Cyclic feature
    'promotion': your_promotion_data,  # Business feature
})

print(f"DataFrame shape: {data.shape}")
print(f"Zero ratio in target: {(data['sales'] == 0).mean():.3f}")
print("Column types:")
print(data.dtypes)
```

### Format 4: CSV File

Standard file format for data storage:

```csv
timestamp,sales,temperature,is_weekend,hour_of_day,promotion
2023-01-01 00:00:00,12.5,15.2,0,0,0
2023-01-01 01:00:00,0.0,14.8,0,1,0
2023-01-01 02:00:00,8.7,14.1,0,2,0
2023-01-01 03:00:00,0.0,13.5,0,3,0
2023-01-01 04:00:00,15.3,13.2,0,4,1
...
```

**Loading CSV data:**
```python
import pandas as pd

# Load CSV file
data = pd.read_csv('your_data.csv')

# Parse timestamp column
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Set timestamp as index (optional)
data.set_index('timestamp', inplace=True)

print("Data loaded successfully!")
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
```

## ✅ Data Validation and Conversion

### Step 1: Validate Data Format

Use built-in validation functions:

```python
from data.formatters import validate_zero_inflated_data

# For DataFrame/CSV
is_valid, issues = validate_zero_inflated_data(
    data, 
    value_column='sales'
)

# For NumPy array or Series
is_valid, issues = validate_zero_inflated_data(data)

if is_valid:
    print("✓ Data format is valid and ready for analysis")
else:
    print("✗ Data format issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

### Step 2: Convert to Standard Format

```python
from data.formatters import convert_to_standard_format

# Convert to standard internal format
standard_data = convert_to_standard_format(
    data,
    value_column='sales',  # For DataFrame/CSV
    timestamp_column='timestamp',  # Optional
    feature_columns=['temperature', 'is_weekend', 'hour_of_day']  # Optional
)

# Check conversion results
summary = standard_data.get_summary_stats()
print("Conversion successful!")
print(f"Length: {summary['length']}")
print(f"Zero ratio: {summary['zero_ratio']:.3f}")
print(f"Has timestamps: {summary['has_timestamps']}")
print(f"Has features: {summary['has_features']}")
```

## 🔄 Complete Workflow Example

Here's a complete example using real-world e-commerce data:

```python
import numpy as np
import pandas as pd
from data.loaders import ZeroInflatedDataLoader
from models.baseline.zip_model import ZeroInflatedPoisson
from models.zero_aware.zip_rnn import ZIPRNN
from evaluation.metrics import ZeroInflatedMetrics

# Step 1: Load your data
print("Step 1: Loading data...")
data = pd.read_csv('ecommerce_hourly_sales.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

print(f"Loaded data shape: {data.shape}")
print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")

# Step 2: Data validation
print("\nStep 2: Validating data format...")
from data.formatters import validate_zero_inflated_data

is_valid, issues = validate_zero_inflated_data(
    data, 
    value_column='sales'
)

if not is_valid:
    print("Data issues found:")
    for issue in issues:
        print(f"  - {issue}")
    exit()

# Step 3: Prepare data for modeling
print("\nStep 3: Preparing data for modeling...")
loader = ZeroInflatedDataLoader()

prepared_data = loader.load_and_prepare(
    data=data,
    sequence_length=48,        # Use 48 hours of history
    prediction_horizon=12,     # Predict next 12 hours
    test_split=0.2,           # 20% for testing
    batch_size=32,            # Batch size for training
    value_column='sales',     # Target variable
    timestamp_column='timestamp',
    feature_columns=['temperature', 'is_weekend', 'hour_of_day'],
    normalize=True            # Normalize features
)

# Get data loaders
train_loader = prepared_data['train_loader']
val_loader = prepared_data['val_loader']
test_loader = prepared_data['test_loader']

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Step 4: Train statistical model
print("\nStep 4: Training statistical model (ZIP)...")
# First, prepare data for statistical models
X_train, y_train = prepared_data['X_train'], prepared_data['y_train']
X_test, y_test = prepared_data['X_test'], prepared_data['y_test']

zip_model = ZeroInflatedPoisson()
zip_model.fit(X_train, y_train)
zip_predictions = zip_model.predict(X_test)

# Step 5: Train deep learning model
print("\nStep 5: Training deep learning model (ZIP-RNN)...")
import torch

rnn_model = ZIPRNN(
    input_dim=1 + len(['temperature', 'is_weekend', 'hour_of_day']),  # Features + target
    hidden_dim=64,
    num_layers=2,
    seq_len=48,
    pred_len=12
)

# Training loop
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)
num_epochs = 50

for epoch in range(num_epochs):
    rnn_model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        
        outputs = rnn_model(inputs)
        loss = rnn_model.compute_zip_loss(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Step 6: Evaluate models
print("\nStep 6: Evaluating models...")
evaluator = ZeroInflatedMetrics()

# Evaluate ZIP model
zip_metrics = evaluator(zip_predictions, y_test)
print("ZIP Model Results:")
print(f"  MSE: {zip_metrics['mse']:.4f}")
print(f"  MAE: {zip_metrics['mae']:.4f}")
print(f"  Zero Classification Accuracy: {zip_metrics['zero_classification_accuracy']:.3f}")

# Evaluate RNN model
rnn_model.eval()
rnn_predictions = []

with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = rnn_model(inputs)
        rnn_predictions.append(outputs.cpu().numpy())

rnn_predictions = np.concatenate(rnn_predictions, axis=0)
rnn_metrics = evaluator(rnn_predictions, y_test[:len(rnn_predictions)])

print("ZIP-RNN Model Results:")
print(f"  MSE: {rnn_metrics['mse']:.4f}")
print(f"  MAE: {rnn_metrics['mae']:.4f}")
print(f"  Zero Classification Accuracy: {rnn_metrics['zero_classification_accuracy']:.3f}")

# Step 7: Model comparison
print("\nStep 7: Model comparison...")
print("Model Performance Summary:")
print(f"{'Model':<15} {'MSE':<8} {'MAE':<8} {'Zero Acc':<10}")
print("-" * 45)
print(f"{'ZIP':<15} {zip_metrics['mse']:<8.4f} {zip_metrics['mae']:<8.4f} {zip_metrics['zero_classification_accuracy']:<10.3f}")
print(f"{'ZIP-RNN':<15} {rnn_metrics['mse']:<8.4f} {rnn_metrics['mae']:<8.4f} {rnn_metrics['zero_classification_accuracy']:<10.3f}")

print("\n✅ Workflow completed successfully!")
```

## ❓ Common Issues and Solutions

### Issue 1: High Zero Ratio (>80%)

**Problem**: Too many zeros may indicate data quality issues.

**Solutions**:
```python
# Check data distribution
zero_ratio = (data['sales'] == 0).mean()
if zero_ratio > 0.8:
    print(f"Warning: Very high zero ratio ({zero_ratio:.3f})")
    
    # Possible solutions:
    # 1. Aggregate to larger time intervals
    data_daily = data.groupby(data['timestamp'].dt.date)['sales'].sum()
    
    # 2. Filter out periods with systematic zeros
    business_hours = data['hour_of_day'].between(9, 18)
    data_filtered = data[business_hours]
    
    # 3. Use specialized models for extreme zero inflation
    from models.baseline.hurdle_model import HurdleModel
    hurdle_model = HurdleModel()
```

### Issue 2: Irregular Time Intervals

**Problem**: Inconsistent time gaps between observations.

**Solutions**:
```python
# Check time intervals
time_diffs = data['timestamp'].diff().dt.total_seconds()
print(f"Time interval stats (seconds):")
print(time_diffs.describe())

# Resample to regular intervals
data_resampled = data.set_index('timestamp').resample('1H').agg({
    'sales': 'sum',  # Sum sales within each hour
    'temperature': 'mean',  # Average temperature
    'is_weekend': 'first'  # Take first value
}).fillna(0)
```

### Issue 3: Missing Values

**Problem**: NaN or null values in the dataset.

**Solutions**:
```python
# Check for missing values
missing_info = data.isnull().sum()
print("Missing values per column:")
print(missing_info[missing_info > 0])

# Handle missing values
# Option 1: Forward fill (for time series)
data['sales'].fillna(method='ffill', inplace=True)

# Option 2: Interpolation
data['temperature'].interpolate(method='linear', inplace=True)

# Option 3: Remove rows with missing target values
data.dropna(subset=['sales'], inplace=True)
```

### Issue 4: Negative Values

**Problem**: Negative values in target variable.

**Solutions**:
```python
# Check for negative values
negative_count = (data['sales'] < 0).sum()
if negative_count > 0:
    print(f"Found {negative_count} negative values")
    
    # Option 1: Set negative values to zero
    data['sales'] = np.maximum(0, data['sales'])
    
    # Option 2: Use absolute values if appropriate
    data['sales'] = np.abs(data['sales'])
    
    # Option 3: Remove negative values
    data = data[data['sales'] >= 0]
```

## 💡 Best Practices

### 1. Data Quality Checks

Always perform comprehensive data quality checks:

```python
def perform_data_quality_checks(data, target_column):
    """Comprehensive data quality assessment"""
    
    print("=== Data Quality Report ===")
    
    # Basic statistics
    print(f"Dataset shape: {data.shape}")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Zero inflation analysis
    zero_ratio = (data[target_column] == 0).mean()
    print(f"Zero ratio: {zero_ratio:.3f}")
    
    if zero_ratio < 0.05:
        print("⚠️  Warning: Very low zero ratio - may not be zero-inflated")
    elif zero_ratio > 0.9:
        print("⚠️  Warning: Extremely high zero ratio - check data quality")
    
    # Missing values
    missing = data.isnull().sum()
    if missing.any():
        print("Missing values found:")
        print(missing[missing > 0])
    
    # Negative values
    if (data[target_column] < 0).any():
        negative_count = (data[target_column] < 0).sum()
        print(f"⚠️  Warning: {negative_count} negative values found")
    
    # Time gaps
    if 'timestamp' in data.columns:
        time_diffs = data['timestamp'].diff().dt.total_seconds()
        irregular_gaps = time_diffs.std() / time_diffs.mean() > 0.1
        if irregular_gaps:
            print("⚠️  Warning: Irregular time intervals detected")
    
    print("=== End Report ===\n")

# Use the function
perform_data_quality_checks(data, 'sales')
```

### 2. Feature Engineering

Create meaningful features for better model performance:

```python
def create_time_features(data, timestamp_col='timestamp'):
    """Create temporal features from timestamp"""
    
    data = data.copy()
    
    # Cyclic time features
    data['hour'] = data[timestamp_col].dt.hour
    data['day_of_week'] = data[timestamp_col].dt.dayofweek
    data['day_of_year'] = data[timestamp_col].dt.dayofyear
    data['month'] = data[timestamp_col].dt.month
    
    # Cyclic encoding (preserves cyclical nature)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    # Binary features
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    data['is_business_hour'] = data['hour'].between(9, 17).astype(int)
    
    return data

# Apply feature engineering
data_with_features = create_time_features(data)
```

### 3. Model Selection Guidelines

Choose models based on your data characteristics:

```python
def recommend_model(data, target_column):
    """Recommend appropriate models based on data characteristics"""
    
    zero_ratio = (data[target_column] == 0).mean()
    n_samples = len(data)
    has_features = len(data.columns) > 2  # More than timestamp and target
    
    recommendations = []
    
    if zero_ratio < 0.2:
        recommendations.append("Tweedie GLM - handles natural zeros well")
    elif 0.2 <= zero_ratio <= 0.6:
        recommendations.append("ZIP Model - good for moderate zero inflation")
        recommendations.append("ZINB Model - if overdispersion is present")
    else:  # zero_ratio > 0.6
        recommendations.append("Hurdle Model - excellent for high zero inflation")
    
    if n_samples > 1000 and has_features:
        recommendations.append("ZIP-RNN - for complex temporal patterns")
        recommendations.append("Dual Branch Network - separates zero/non-zero modeling")
    
    return recommendations

# Get model recommendations
recommendations = recommend_model(data, 'sales')
print("Recommended models:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")
```

### 4. Validation Strategy

Use appropriate validation techniques for time series:

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_validation(X, y, model, n_splits=5):
    """Time series cross-validation"""
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        
        # Evaluate
        mse = np.mean((predictions - y_val) ** 2)
        scores.append(mse)
        
        print(f"Fold {fold + 1}: MSE = {mse:.4f}")
    
    print(f"Average MSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    return scores

# Example usage
# scores = time_series_validation(X, y, ZeroInflatedPoisson())
```

This comprehensive user guide provides everything needed to successfully use your own datasets with the zero-inflated time series toolkit. For more advanced topics, refer to the API reference documentation.