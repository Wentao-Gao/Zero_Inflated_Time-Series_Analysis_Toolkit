"""
Data loading utilities for zero-inflated time series analysis.

This module provides convenient data loaders for various formats
and standardized dataset classes for use with the library's models.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, Optional, Tuple, List, Dict, Any
import warnings
from pathlib import Path

from .formatters import StandardTimeSeriesFormat, convert_to_standard_format


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data with sequence generation.
    
    This dataset creates input-output sequences for time series forecasting
    and is compatible with PyTorch DataLoaders.
    """
    
    def __init__(self, 
                 data: Union[np.ndarray, StandardTimeSeriesFormat],
                 sequence_length: int = 96,
                 prediction_horizon: int = 24,
                 stride: int = 1,
                 include_features: bool = True,
                 normalize: bool = False):
        """
        Initialize the dataset.
        
        Args:
            data: Time series data (numpy array or StandardTimeSeriesFormat)
            sequence_length: Length of input sequences
            prediction_horizon: Length of prediction sequences
            stride: Step size between sequences
            include_features: Whether to include additional features
            normalize: Whether to normalize the data
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.include_features = include_features
        
        # Convert data to standard format if needed
        if isinstance(data, StandardTimeSeriesFormat):
            self.data_format = data
        elif isinstance(data, np.ndarray):
            self.data_format = StandardTimeSeriesFormat(values=data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Extract values and features
        self.values = self.data_format.values
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)
        
        self.features = self.data_format.features
        if self.features is not None and self.features.ndim == 1:
            self.features = self.features.reshape(-1, 1)
        
        # Normalization
        self.normalize = normalize
        self.value_mean = None
        self.value_std = None
        self.feature_mean = None
        self.feature_std = None
        
        if self.normalize:
            self._fit_normalization()
            self._apply_normalization()
        
        # Generate sequences
        self.sequences = self._generate_sequences()
        
    def _fit_normalization(self):
        """Fit normalization parameters."""
        self.value_mean = np.mean(self.values, axis=0)
        self.value_std = np.std(self.values, axis=0)
        
        # Avoid division by zero
        self.value_std = np.where(self.value_std == 0, 1, self.value_std)
        
        if self.features is not None:
            self.feature_mean = np.mean(self.features, axis=0)
            self.feature_std = np.std(self.features, axis=0)
            self.feature_std = np.where(self.feature_std == 0, 1, self.feature_std)
    
    def _apply_normalization(self):
        """Apply normalization to data."""
        if self.value_mean is not None:
            self.values = (self.values - self.value_mean) / self.value_std
        
        if self.features is not None and self.feature_mean is not None:
            self.features = (self.features - self.feature_mean) / self.feature_std
    
    def _generate_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate input-output sequences."""
        sequences = []
        
        total_length = self.sequence_length + self.prediction_horizon
        max_start = len(self.values) - total_length + 1
        
        for i in range(0, max_start, self.stride):
            # Input sequence
            input_values = self.values[i:i+self.sequence_length]
            
            # Include features if available and requested
            if self.include_features and self.features is not None:
                input_features = self.features[i:i+self.sequence_length]
                input_seq = np.concatenate([input_values, input_features], axis=1)
            else:
                input_seq = input_values
            
            # Target sequence
            target_seq = self.values[i+self.sequence_length:i+total_length]
            
            # Convert to tensors
            input_tensor = torch.FloatTensor(input_seq)
            target_tensor = torch.FloatTensor(target_seq)
            
            sequences.append((input_tensor, target_tensor))
        
        return sequences
    
    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence pair by index."""
        return self.sequences[idx]
    
    def get_feature_dimension(self) -> int:
        """Get the total feature dimension (values + features)."""
        base_dim = self.values.shape[1]
        if self.include_features and self.features is not None:
            base_dim += self.features.shape[1]
        return base_dim
    
    def denormalize_values(self, normalized_values: np.ndarray) -> np.ndarray:
        """Denormalize values back to original scale."""
        if not self.normalize or self.value_mean is None:
            return normalized_values
        
        return normalized_values * self.value_std + self.value_mean
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        return {
            'total_sequences': len(self.sequences),
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'feature_dimension': self.get_feature_dimension(),
            'includes_features': self.include_features and self.features is not None,
            'normalized': self.normalize,
            'stride': self.stride,
            'original_data_length': len(self.data_format.values),
            'zero_ratio': self.data_format.get_zero_ratio()
        }


class ZeroInflatedDataLoader:
    """
    Convenient data loader for zero-inflated time series data.
    
    This class provides easy loading and preprocessing of data in various formats
    for use with zero-inflated time series models.
    """
    
    def __init__(self, zero_threshold: float = 1e-6):
        """
        Initialize the data loader.
        
        Args:
            zero_threshold: Threshold for considering values as zero
        """
        self.zero_threshold = zero_threshold
    
    def load_and_prepare(self,
                        data: Union[str, np.ndarray, pd.Series, pd.DataFrame],
                        sequence_length: int = 96,
                        prediction_horizon: int = 24,
                        test_split: float = 0.2,
                        validation_split: float = 0.1,
                        batch_size: int = 32,
                        value_column: str = 'value',
                        timestamp_column: Optional[str] = None,
                        feature_columns: Optional[List[str]] = None,
                        normalize: bool = True,
                        **kwargs) -> Dict[str, Any]:
        """
        Load and prepare data for training.
        
        Args:
            data: Input data in various formats
            sequence_length: Length of input sequences
            prediction_horizon: Length of prediction sequences  
            test_split: Fraction of data for testing
            validation_split: Fraction of training data for validation
            batch_size: Batch size for DataLoaders
            value_column: Name of value column (for DataFrame/CSV)
            timestamp_column: Name of timestamp column (optional)
            feature_columns: List of feature column names (optional)
            normalize: Whether to normalize the data
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with train/val/test dataloaders and metadata
        """
        # Convert to standard format
        standard_data = convert_to_standard_format(
            data=data,
            value_column=value_column,
            timestamp_column=timestamp_column,
            feature_columns=feature_columns,
            zero_threshold=self.zero_threshold
        )
        
        print(f"Loaded data: {len(standard_data.values)} samples")
        print(f"Zero ratio: {standard_data.get_zero_ratio():.3f}")
        if standard_data.features is not None:
            print(f"Additional features: {standard_data.features.shape[1]}")
        
        # Split data
        n_total = len(standard_data.values)
        n_test = int(n_total * test_split)
        n_train_val = n_total - n_test
        n_val = int(n_train_val * validation_split)
        n_train = n_train_val - n_val
        
        # Create splits
        train_data = StandardTimeSeriesFormat(
            values=standard_data.values[:n_train],
            timestamps=standard_data.timestamps[:n_train] if standard_data.timestamps is not None else None,
            features=standard_data.features[:n_train] if standard_data.features is not None else None,
            zero_threshold=self.zero_threshold
        )
        
        val_data = StandardTimeSeriesFormat(
            values=standard_data.values[n_train:n_train+n_val],
            timestamps=standard_data.timestamps[n_train:n_train+n_val] if standard_data.timestamps is not None else None,
            features=standard_data.features[n_train:n_train+n_val] if standard_data.features is not None else None,
            zero_threshold=self.zero_threshold
        )
        
        test_data = StandardTimeSeriesFormat(
            values=standard_data.values[n_train+n_val:],
            timestamps=standard_data.timestamps[n_train+n_val:] if standard_data.timestamps is not None else None,
            features=standard_data.features[n_train+n_val:] if standard_data.features is not None else None,
            zero_threshold=self.zero_threshold
        )
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_data, sequence_length, prediction_horizon, 
            normalize=normalize, **kwargs
        )
        
        val_dataset = TimeSeriesDataset(
            val_data, sequence_length, prediction_horizon, 
            normalize=normalize, **kwargs
        )
        
        test_dataset = TimeSeriesDataset(
            test_data, sequence_length, prediction_horizon, 
            normalize=normalize, **kwargs
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'data_info': {
                'total_samples': n_total,
                'train_samples': n_train,
                'val_samples': n_val,
                'test_samples': n_test,
                'feature_dimension': train_dataset.get_feature_dimension(),
                'sequence_length': sequence_length,
                'prediction_horizon': prediction_horizon,
                'zero_ratio': standard_data.get_zero_ratio(),
                'normalized': normalize
            },
            'original_data': standard_data
        }


def load_csv_data(file_path: str,
                  value_column: str = 'value',
                  timestamp_column: Optional[str] = None,
                  feature_columns: Optional[List[str]] = None,
                  **kwargs) -> StandardTimeSeriesFormat:
    """
    Load zero-inflated time series data from CSV file.
    
    Args:
        file_path: Path to CSV file
        value_column: Name of the value column
        timestamp_column: Name of the timestamp column (optional)
        feature_columns: List of feature column names (optional)
        **kwargs: Additional arguments for convert_to_standard_format
        
    Returns:
        StandardTimeSeriesFormat object
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return convert_to_standard_format(
        data=file_path,
        value_column=value_column,
        timestamp_column=timestamp_column,
        feature_columns=feature_columns,
        **kwargs
    )


def load_numpy_data(data: np.ndarray, 
                   timestamps: Optional[np.ndarray] = None,
                   features: Optional[np.ndarray] = None,
                   **kwargs) -> StandardTimeSeriesFormat:
    """
    Load zero-inflated time series data from numpy arrays.
    
    Args:
        data: Time series values
        timestamps: Optional timestamps
        features: Optional features
        **kwargs: Additional arguments
        
    Returns:
        StandardTimeSeriesFormat object
    """
    return StandardTimeSeriesFormat(
        values=data,
        timestamps=timestamps,
        features=features,
        **kwargs
    )


def load_pandas_data(data: Union[pd.Series, pd.DataFrame],
                    value_column: str = 'value',
                    timestamp_column: Optional[str] = None,
                    feature_columns: Optional[List[str]] = None,
                    **kwargs) -> StandardTimeSeriesFormat:
    """
    Load zero-inflated time series data from pandas objects.
    
    Args:
        data: Pandas Series or DataFrame
        value_column: Name of the value column (for DataFrame)
        timestamp_column: Name of the timestamp column (optional)
        feature_columns: List of feature column names (optional)
        **kwargs: Additional arguments
        
    Returns:
        StandardTimeSeriesFormat object
    """
    return convert_to_standard_format(
        data=data,
        value_column=value_column,
        timestamp_column=timestamp_column,
        feature_columns=feature_columns,
        **kwargs
    )


def create_sample_dataset(n_samples: int = 1000,
                         zero_ratio: float = 0.3,
                         include_features: bool = True,
                         include_timestamps: bool = True,
                         random_state: int = 42) -> StandardTimeSeriesFormat:
    """
    Create a sample zero-inflated time series dataset for testing.
    
    Args:
        n_samples: Number of samples to generate
        zero_ratio: Ratio of zero values
        include_features: Whether to include additional features
        include_timestamps: Whether to include timestamps
        random_state: Random seed
        
    Returns:
        StandardTimeSeriesFormat object with sample data
    """
    np.random.seed(random_state)
    
    # Generate time series
    t = np.arange(n_samples)
    trend = 0.001 * t
    seasonal = 2 * np.sin(2 * np.pi * t / 365.25)  # Annual cycle
    noise = np.random.normal(0, 0.5, n_samples)
    base_values = 3 + trend + seasonal + noise
    base_values = np.maximum(base_values, 0)
    
    # Add zero-inflation
    zero_mask = np.random.random(n_samples) < zero_ratio
    values = base_values * ~zero_mask
    
    # Generate features
    features = None
    if include_features:
        feature1 = np.random.normal(10, 2, n_samples)  # Temperature-like
        feature2 = np.random.poisson(3, n_samples)     # Count-like
        features = np.column_stack([feature1, feature2])
    
    # Generate timestamps
    timestamps = None
    if include_timestamps:
        timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='D').values
    
    return StandardTimeSeriesFormat(
        values=values,
        timestamps=timestamps,
        features=features,
        zero_threshold=1e-6
    )