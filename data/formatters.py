"""
Data format validation and conversion utilities for zero-inflated time series.

This module defines the standard data formats expected by the library
and provides tools for validating and converting user data to these formats.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
from datetime import datetime, timedelta


@dataclass 
class StandardTimeSeriesFormat:
    """
    Standard format specification for zero-inflated time series data.
    
    This class defines the expected format for all time series data
    used throughout the library.
    """
    
    # Data requirements
    values: np.ndarray  # Time series values (1D or 2D array)
    timestamps: Optional[np.ndarray] = None  # Optional timestamps
    features: Optional[np.ndarray] = None  # Optional additional features
    metadata: Optional[Dict[str, Any]] = None  # Optional metadata
    
    # Data properties
    zero_threshold: float = 1e-6  # Threshold for considering values as zero
    frequency: Optional[str] = None  # Time series frequency ('D', 'H', 'M', etc.)
    
    def __post_init__(self):
        """Validate the data format after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate that the data conforms to the standard format.
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check values array
        if not isinstance(self.values, np.ndarray):
            raise ValueError("Values must be a numpy array")
        
        if self.values.ndim == 0:
            raise ValueError("Values cannot be a scalar")
        
        if self.values.ndim > 2:
            raise ValueError("Values must be 1D or 2D array")
        
        # Check for non-negative values (zero-inflated data should be non-negative)
        if np.any(self.values < 0):
            warnings.warn("Found negative values in zero-inflated time series data")
        
        # Check timestamps if provided
        if self.timestamps is not None:
            if not isinstance(self.timestamps, np.ndarray):
                raise ValueError("Timestamps must be a numpy array")
            
            if len(self.timestamps) != len(self.values):
                raise ValueError("Timestamps length must match values length")
        
        # Check features if provided
        if self.features is not None:
            if not isinstance(self.features, np.ndarray):
                raise ValueError("Features must be a numpy array")
            
            if len(self.features) != len(self.values):
                raise ValueError("Features length must match values length")
        
        # Check zero threshold
        if self.zero_threshold < 0:
            raise ValueError("Zero threshold must be non-negative")
        
        return True
    
    def get_zero_ratio(self) -> float:
        """Calculate the ratio of zero values in the time series."""
        return np.mean(np.abs(self.values) <= self.zero_threshold)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the time series."""
        return {
            'length': len(self.values),
            'shape': self.values.shape,
            'zero_ratio': self.get_zero_ratio(),
            'min_value': np.min(self.values),
            'max_value': np.max(self.values),
            'mean_value': np.mean(self.values),
            'std_value': np.std(self.values),
            'has_timestamps': self.timestamps is not None,
            'has_features': self.features is not None,
            'frequency': self.frequency
        }


class DataFormatValidator:
    """
    Validator for checking if data conforms to expected formats.
    
    This class provides comprehensive validation for various input data types
    and ensures they can be converted to the standard format.
    """
    
    def __init__(self, zero_threshold: float = 1e-6, allow_negative: bool = False):
        """
        Initialize the validator.
        
        Args:
            zero_threshold: Threshold for considering values as zero
            allow_negative: Whether to allow negative values
        """
        self.zero_threshold = zero_threshold
        self.allow_negative = allow_negative
    
    def validate_numpy_array(self, data: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate a numpy array for zero-inflated time series use.
        
        Args:
            data: Numpy array to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check type
        if not isinstance(data, np.ndarray):
            issues.append("Data must be a numpy array")
            return False, issues
        
        # Check dimensions
        if data.ndim == 0:
            issues.append("Data cannot be a scalar")
        elif data.ndim > 2:
            issues.append("Data must be 1D or 2D array (got {}D)".format(data.ndim))
        
        # Check for invalid values
        if np.any(np.isnan(data)):
            issues.append("Data contains NaN values")
        
        if np.any(np.isinf(data)):
            issues.append("Data contains infinite values")
        
        # Check for negative values
        if not self.allow_negative and np.any(data < 0):
            issues.append("Data contains negative values (use allow_negative=True if intentional)")
        
        # Check data type
        if not np.issubdtype(data.dtype, np.number):
            issues.append("Data must be numeric")
        
        # Warning for very sparse data
        zero_ratio = np.mean(np.abs(data) <= self.zero_threshold)
        if zero_ratio > 0.8:
            issues.append("WARNING: Data is extremely sparse ({}% zeros)".format(zero_ratio * 100))
        
        return len(issues) == 0, issues
    
    def validate_pandas_series(self, data: pd.Series) -> Tuple[bool, List[str]]:
        """
        Validate a pandas Series.
        
        Args:
            data: Pandas Series to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not isinstance(data, pd.Series):
            issues.append("Data must be a pandas Series")
            return False, issues
        
        # Convert to numpy and validate
        try:
            numpy_data = data.values
            numpy_valid, numpy_issues = self.validate_numpy_array(numpy_data)
            issues.extend(numpy_issues)
        except Exception as e:
            issues.append(f"Error converting Series to numpy: {str(e)}")
        
        # Check index
        if data.index.duplicated().any():
            issues.append("Series index contains duplicates")
        
        return len(issues) == 0, issues
    
    def validate_pandas_dataframe(self, data: pd.DataFrame, 
                                 value_column: str = 'value') -> Tuple[bool, List[str]]:
        """
        Validate a pandas DataFrame.
        
        Args:
            data: Pandas DataFrame to validate
            value_column: Name of the column containing time series values
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not isinstance(data, pd.DataFrame):
            issues.append("Data must be a pandas DataFrame")
            return False, issues
        
        # Check if value column exists
        if value_column not in data.columns:
            issues.append(f"Value column '{value_column}' not found in DataFrame")
            return False, issues
        
        # Validate the value column
        series_valid, series_issues = self.validate_pandas_series(data[value_column])
        issues.extend(series_issues)
        
        # Check for time column
        potential_time_columns = ['timestamp', 'time', 'date', 'datetime']
        time_column = None
        for col in potential_time_columns:
            if col in data.columns:
                time_column = col
                break
        
        if time_column:
            try:
                pd.to_datetime(data[time_column])
            except:
                issues.append(f"Time column '{time_column}' cannot be converted to datetime")
        
        return len(issues) == 0, issues
    
    def validate_csv_file(self, file_path: str, 
                         value_column: str = 'value') -> Tuple[bool, List[str]]:
        """
        Validate a CSV file for zero-inflated time series data.
        
        Args:
            file_path: Path to the CSV file
            value_column: Name of the column containing time series values
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Try to read the CSV
            data = pd.read_csv(file_path)
            df_valid, df_issues = self.validate_pandas_dataframe(data, value_column)
            issues.extend(df_issues)
            
        except FileNotFoundError:
            issues.append(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            issues.append("CSV file is empty")
        except Exception as e:
            issues.append(f"Error reading CSV file: {str(e)}")
        
        return len(issues) == 0, issues


def validate_zero_inflated_data(data: Union[np.ndarray, pd.Series, pd.DataFrame, str],
                               value_column: str = 'value',
                               zero_threshold: float = 1e-6,
                               allow_negative: bool = False) -> Tuple[bool, List[str]]:
    """
    Convenient function to validate zero-inflated time series data.
    
    Args:
        data: Data to validate (numpy array, pandas Series/DataFrame, or CSV file path)
        value_column: Column name for DataFrame or CSV
        zero_threshold: Threshold for considering values as zero
        allow_negative: Whether to allow negative values
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    validator = DataFormatValidator(zero_threshold=zero_threshold, 
                                   allow_negative=allow_negative)
    
    if isinstance(data, np.ndarray):
        return validator.validate_numpy_array(data)
    elif isinstance(data, pd.Series):
        return validator.validate_pandas_series(data)
    elif isinstance(data, pd.DataFrame):
        return validator.validate_pandas_dataframe(data, value_column)
    elif isinstance(data, str):
        return validator.validate_csv_file(data, value_column)
    else:
        return False, [f"Unsupported data type: {type(data)}"]


def convert_to_standard_format(data: Union[np.ndarray, pd.Series, pd.DataFrame, str],
                              value_column: str = 'value',
                              timestamp_column: Optional[str] = None,
                              feature_columns: Optional[List[str]] = None,
                              zero_threshold: float = 1e-6) -> StandardTimeSeriesFormat:
    """
    Convert various data formats to the standard time series format.
    
    Args:
        data: Input data in various formats
        value_column: Name of the value column (for DataFrame/CSV)
        timestamp_column: Name of the timestamp column (optional)
        feature_columns: List of feature column names (optional)
        zero_threshold: Threshold for considering values as zero
        
    Returns:
        StandardTimeSeriesFormat object
        
    Raises:
        ValueError: If data cannot be converted or is invalid
    """
    # First validate the data
    is_valid, issues = validate_zero_inflated_data(
        data, value_column, zero_threshold, allow_negative=False
    )
    
    if not is_valid:
        raise ValueError(f"Data validation failed: {'; '.join(issues)}")
    
    # Convert based on input type
    if isinstance(data, np.ndarray):
        values = data.copy()
        timestamps = None
        features = None
        frequency = None
        
    elif isinstance(data, pd.Series):
        values = data.values
        timestamps = data.index.values if not data.index.equals(pd.RangeIndex(len(data))) else None
        features = None
        frequency = getattr(data.index, 'freq', None)
        
    elif isinstance(data, pd.DataFrame):
        # Load DataFrame
        df = data.copy()
        
        values = df[value_column].values
        
        # Handle timestamps
        if timestamp_column and timestamp_column in df.columns:
            timestamps = pd.to_datetime(df[timestamp_column]).values
        elif not df.index.equals(pd.RangeIndex(len(df))):
            timestamps = df.index.values
        else:
            timestamps = None
        
        # Handle features
        if feature_columns:
            available_features = [col for col in feature_columns if col in df.columns]
            if available_features:
                features = df[available_features].values
            else:
                features = None
        else:
            # Auto-detect numeric columns as features (exclude value column)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if value_column in numeric_cols:
                numeric_cols.remove(value_column)
            if numeric_cols:
                features = df[numeric_cols].values
            else:
                features = None
        
        frequency = getattr(df.index, 'freq', None) if hasattr(df.index, 'freq') else None
        
    elif isinstance(data, str):  # CSV file path
        df = pd.read_csv(data)
        return convert_to_standard_format(
            df, value_column, timestamp_column, feature_columns, zero_threshold
        )
    
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    # Create metadata
    metadata = {
        'original_type': type(data).__name__,
        'conversion_timestamp': datetime.now().isoformat(),
        'zero_threshold': zero_threshold
    }
    
    return StandardTimeSeriesFormat(
        values=values,
        timestamps=timestamps,
        features=features,
        metadata=metadata,
        zero_threshold=zero_threshold,
        frequency=str(frequency) if frequency else None
    )


def create_sample_data_format() -> Dict[str, Any]:
    """
    Create sample data showing the expected format for user data.
    
    Returns:
        Dictionary with sample data in various formats
    """
    # Generate sample time series with zero-inflation
    np.random.seed(42)
    n_samples = 100
    
    # Time stamps
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    # Base time series with trend and seasonality
    t = np.arange(n_samples)
    trend = 0.01 * t
    seasonal = 2 * np.sin(2 * np.pi * t / 30)  # Monthly seasonality
    noise = np.random.normal(0, 0.5, n_samples)
    base_values = 5 + trend + seasonal + noise
    base_values = np.maximum(base_values, 0)  # Ensure non-negative
    
    # Add zero-inflation
    zero_mask = np.random.random(n_samples) < 0.2  # 20% zeros
    values = base_values * ~zero_mask
    
    # Additional features
    feature1 = np.random.normal(10, 2, n_samples)  # Temperature
    feature2 = np.random.poisson(3, n_samples)     # Count feature
    
    return {
        'numpy_array_format': {
            'description': '1D numpy array with time series values',
            'example': values,
            'usage': 'data = np.array([1.2, 0.0, 2.1, 0.0, 1.8, ...])'
        },
        
        'pandas_series_format': {
            'description': 'Pandas Series with optional datetime index',
            'example': pd.Series(values, index=timestamps, name='value'),
            'usage': """
data = pd.Series(
    [1.2, 0.0, 2.1, 0.0, 1.8, ...], 
    index=pd.date_range('2023-01-01', periods=100, freq='D'),
    name='value'
)"""
        },
        
        'pandas_dataframe_format': {
            'description': 'DataFrame with value column and optional timestamp/features',
            'example': pd.DataFrame({
                'timestamp': timestamps,
                'value': values,
                'temperature': feature1,
                'count': feature2
            }),
            'usage': """
data = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
    'value': [1.2, 0.0, 2.1, 0.0, 1.8, ...],
    'feature1': [...],  # Optional additional features
    'feature2': [...]   # Optional additional features
})"""
        },
        
        'csv_format': {
            'description': 'CSV file with columns for timestamp, value, and features',
            'example_content': """timestamp,value,temperature,count
2023-01-01,1.2,12.1,3
2023-01-02,0.0,11.8,2
2023-01-03,2.1,13.2,4
2023-01-04,0.0,12.5,1
2023-01-05,1.8,11.9,3
...""",
            'usage': 'Save data as CSV with appropriate column names'
        },
        
        'data_requirements': {
            'values': 'Non-negative numeric values (zeros allowed and expected)',
            'timestamps': 'Optional datetime index or column',
            'features': 'Optional additional numeric features',
            'format': 'No missing values (NaN) or infinite values',
            'size': 'At least 50 observations recommended for modeling'
        }
    }