"""
Data preprocessing utilities for zero-inflated time series analysis.

This module provides preprocessing tools specifically designed for 
zero-inflated time series data, including scalers, sequence generators,
and train-test splitters.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings

from .formatters import StandardTimeSeriesFormat


class ZeroInflatedPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive preprocessor for zero-inflated time series data.
    
    This preprocessor handles various transformations while preserving
    the zero-inflated nature of the data.
    """
    
    def __init__(self,
                 method: str = 'log_plus_one',
                 handle_outliers: bool = True,
                 outlier_threshold: float = 3.0,
                 zero_threshold: float = 1e-6):
        """
        Initialize the preprocessor.
        
        Args:
            method: Transformation method ('log_plus_one', 'sqrt', 'none')
            handle_outliers: Whether to handle outliers
            outlier_threshold: Z-score threshold for outlier detection
            zero_threshold: Threshold for considering values as zero
        """
        self.method = method
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        self.zero_threshold = zero_threshold
        
        self.fitted = False
        self.outlier_bounds = None
    
    def fit(self, X: np.ndarray, y=None):
        """
        Fit the preprocessor to the data.
        
        Args:
            X: Input data
            y: Target data (ignored)
            
        Returns:
            self
        """
        X = np.asarray(X)
        
        if self.handle_outliers:
            # Calculate outlier bounds based on non-zero values
            non_zero_mask = np.abs(X) > self.zero_threshold
            if np.any(non_zero_mask):
                non_zero_values = X[non_zero_mask]
                mean_val = np.mean(non_zero_values)
                std_val = np.std(non_zero_values)
                
                self.outlier_bounds = {
                    'lower': mean_val - self.outlier_threshold * std_val,
                    'upper': mean_val + self.outlier_threshold * std_val
                }
            else:
                self.outlier_bounds = None
        
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = np.asarray(X)
        X_transformed = X.copy()
        
        # Handle outliers
        if self.handle_outliers and self.outlier_bounds is not None:
            non_zero_mask = np.abs(X_transformed) > self.zero_threshold
            
            # Cap outliers at bounds (preserve zeros)
            outlier_mask = (X_transformed > self.outlier_bounds['upper']) & non_zero_mask
            X_transformed[outlier_mask] = self.outlier_bounds['upper']
            
            outlier_mask = (X_transformed < self.outlier_bounds['lower']) & non_zero_mask
            X_transformed[outlier_mask] = self.outlier_bounds['lower']
        
        # Apply transformation
        if self.method == 'log_plus_one':
            X_transformed = np.log1p(X_transformed)
        elif self.method == 'sqrt':
            X_transformed = np.sqrt(np.maximum(X_transformed, 0))
        elif self.method == 'none':
            pass  # No transformation
        else:
            raise ValueError(f"Unknown transformation method: {self.method}")
        
        return X_transformed
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data.
        
        Args:
            X: Transformed data
            
        Returns:
            Original scale data
        """
        X = np.asarray(X)
        
        # Reverse transformation
        if self.method == 'log_plus_one':
            X_original = np.expm1(X)
        elif self.method == 'sqrt':
            X_original = X ** 2
        elif self.method == 'none':
            X_original = X
        else:
            raise ValueError(f"Unknown transformation method: {self.method}")
        
        # Ensure non-negative values
        X_original = np.maximum(X_original, 0)
        
        return X_original


class TimeSeriesScaler(BaseEstimator, TransformerMixin):
    """
    Scaler specifically designed for zero-inflated time series.
    
    This scaler preserves the zero-inflation structure while normalizing
    the non-zero values appropriately.
    """
    
    def __init__(self,
                 scaler_type: str = 'standard',
                 preserve_zeros: bool = True,
                 zero_threshold: float = 1e-6):
        """
        Initialize the scaler.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            preserve_zeros: Whether to preserve exact zero values
            zero_threshold: Threshold for considering values as zero
        """
        self.scaler_type = scaler_type
        self.preserve_zeros = preserve_zeros
        self.zero_threshold = zero_threshold
        
        # Initialize the underlying scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.fitted = False
    
    def fit(self, X: np.ndarray, y=None):
        """
        Fit the scaler to the data.
        
        Args:
            X: Input data
            y: Target data (ignored)
            
        Returns:
            self
        """
        X = np.asarray(X)
        
        if self.preserve_zeros:
            # Fit only on non-zero values
            non_zero_mask = np.abs(X) > self.zero_threshold
            if np.any(non_zero_mask):
                if X.ndim == 1:
                    non_zero_values = X[non_zero_mask].reshape(-1, 1)
                else:
                    # For 2D arrays, handle each column separately
                    non_zero_values = X[non_zero_mask.any(axis=1)]
                
                self.scaler.fit(non_zero_values)
            else:
                warnings.warn("No non-zero values found for fitting scaler")
        else:
            # Fit on all values
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.scaler.fit(X)
        
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data.
        
        Args:
            X: Input data
            
        Returns:
            Scaled data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        X = np.asarray(X)
        original_shape = X.shape
        is_1d = X.ndim == 1
        
        if is_1d:
            X = X.reshape(-1, 1)
        
        X_scaled = X.copy()
        
        if self.preserve_zeros:
            # Scale only non-zero values
            non_zero_mask = np.abs(X) > self.zero_threshold
            
            if np.any(non_zero_mask):
                if not is_1d and X.shape[1] > 1:
                    # Handle multivariate case
                    for i in range(X.shape[1]):
                        col_mask = non_zero_mask[:, i]
                        if np.any(col_mask):
                            X_scaled[col_mask, i] = self.scaler.transform(
                                X[col_mask, i].reshape(-1, 1)
                            ).flatten()
                else:
                    # Handle univariate case - work with 2D array throughout
                    flat_mask = non_zero_mask.flatten()
                    if np.any(flat_mask):
                        transformed = self.scaler.transform(X[flat_mask])
                        X_scaled[flat_mask] = transformed.reshape(-1, 1)
        else:
            # Scale all values
            X_scaled = self.scaler.transform(X)
        
        if is_1d:
            return X_scaled.flatten()
        else:
            return X_scaled
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data.
        
        Args:
            X: Scaled data
            
        Returns:
            Original scale data
        """
        X = np.asarray(X)
        original_shape = X.shape
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.preserve_zeros:
            # Identify which values were originally zero (approximately)
            # This is approximate since we don't store the original zero mask
            X_original = X.copy()
            
            # For preserved zeros, values should be close to zero after scaling
            likely_zero_mask = np.abs(X) < self.zero_threshold
            non_zero_mask = ~likely_zero_mask
            
            if np.any(non_zero_mask):
                if X.ndim == 2 and X.shape[1] > 1:
                    for i in range(X.shape[1]):
                        col_mask = non_zero_mask[:, i]
                        if np.any(col_mask):
                            X_original[col_mask, i] = self.scaler.inverse_transform(
                                X[col_mask, i].reshape(-1, 1)
                            ).flatten()
                else:
                    flat_mask = non_zero_mask.flatten()
                    if np.any(flat_mask):
                        X_original[flat_mask] = self.scaler.inverse_transform(
                            X[flat_mask].reshape(-1, 1)
                        ).flatten()
            
            # Ensure zeros remain zeros
            X_original[likely_zero_mask] = 0.0
        else:
            X_original = self.scaler.inverse_transform(X)
        
        return X_original.reshape(original_shape)


class SequenceGenerator:
    """
    Generate sequences for time series modeling.
    
    This class creates input-output sequence pairs for training
    time series forecasting models.
    """
    
    def __init__(self,
                 sequence_length: int = 96,
                 prediction_horizon: int = 24,
                 stride: int = 1):
        """
        Initialize the sequence generator.
        
        Args:
            sequence_length: Length of input sequences
            prediction_horizon: Length of prediction sequences
            stride: Step size between sequences
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
    
    def generate_sequences(self, data: np.ndarray,
                          features: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate sequences from time series data.
        
        Args:
            data: Time series data (1D or 2D)
            features: Optional additional features
            
        Returns:
            Tuple of (input_sequences, target_sequences, feature_sequences)
        """
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        total_length = self.sequence_length + self.prediction_horizon
        max_start = len(data) - total_length + 1
        
        if max_start <= 0:
            raise ValueError(f"Data length ({len(data)}) is too short for "
                           f"sequence length ({self.sequence_length}) + "
                           f"prediction horizon ({self.prediction_horizon})")
        
        n_sequences = (max_start - 1) // self.stride + 1
        
        # Initialize arrays
        input_sequences = np.zeros((n_sequences, self.sequence_length, data.shape[1]))
        target_sequences = np.zeros((n_sequences, self.prediction_horizon, data.shape[1]))
        
        feature_sequences = None
        if features is not None:
            features = np.asarray(features)
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            feature_sequences = np.zeros((n_sequences, self.sequence_length, features.shape[1]))
        
        # Generate sequences
        for i in range(n_sequences):
            start_idx = i * self.stride
            end_input = start_idx + self.sequence_length
            end_target = end_input + self.prediction_horizon
            
            input_sequences[i] = data[start_idx:end_input]
            target_sequences[i] = data[end_input:end_target]
            
            if features is not None:
                feature_sequences[i] = features[start_idx:end_input]
        
        return input_sequences, target_sequences, feature_sequences
    
    def get_sequence_info(self, data_length: int) -> Dict[str, int]:
        """
        Get information about sequences that would be generated.
        
        Args:
            data_length: Length of the time series data
            
        Returns:
            Dictionary with sequence information
        """
        total_length = self.sequence_length + self.prediction_horizon
        max_start = data_length - total_length + 1
        n_sequences = max(0, (max_start - 1) // self.stride + 1)
        
        return {
            'data_length': data_length,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'stride': self.stride,
            'total_sequence_length': total_length,
            'max_sequences': n_sequences,
            'coverage_ratio': (n_sequences * self.stride + total_length - 1) / data_length if data_length > 0 else 0
        }


class TrainTestSplitter:
    """
    Time series aware train-test splitter for zero-inflated data.
    
    This splitter respects the temporal order of time series data
    and provides options for handling zero-inflation in splits.
    """
    
    def __init__(self,
                 test_size: float = 0.2,
                 validation_size: float = 0.1,
                 shuffle: bool = False,
                 stratify_zeros: bool = False,
                 zero_threshold: float = 1e-6):
        """
        Initialize the splitter.
        
        Args:
            test_size: Fraction of data for testing
            validation_size: Fraction of training data for validation
            shuffle: Whether to shuffle data (not recommended for time series)
            stratify_zeros: Whether to stratify based on zero ratio
            zero_threshold: Threshold for considering values as zero
        """
        self.test_size = test_size
        self.validation_size = validation_size
        self.shuffle = shuffle
        self.stratify_zeros = stratify_zeros
        self.zero_threshold = zero_threshold
        
        if shuffle:
            warnings.warn("Shuffling is not recommended for time series data")
    
    def split(self, data: StandardTimeSeriesFormat
              ) -> Tuple[StandardTimeSeriesFormat, StandardTimeSeriesFormat, StandardTimeSeriesFormat]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: StandardTimeSeriesFormat object
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        n_total = len(data.values)
        
        if self.stratify_zeros:
            # Try to maintain similar zero ratios across splits
            return self._stratified_split(data, n_total)
        else:
            # Simple temporal split
            return self._temporal_split(data, n_total)
    
    def _temporal_split(self, data: StandardTimeSeriesFormat, n_total: int
                       ) -> Tuple[StandardTimeSeriesFormat, StandardTimeSeriesFormat, StandardTimeSeriesFormat]:
        """Perform temporal split (respecting time order)."""
        n_test = int(n_total * self.test_size)
        n_train_val = n_total - n_test
        n_val = int(n_train_val * self.validation_size)
        n_train = n_train_val - n_val
        
        # Create splits
        train_data = StandardTimeSeriesFormat(
            values=data.values[:n_train],
            timestamps=data.timestamps[:n_train] if data.timestamps is not None else None,
            features=data.features[:n_train] if data.features is not None else None,
            zero_threshold=self.zero_threshold,
            frequency=data.frequency
        )
        
        val_data = StandardTimeSeriesFormat(
            values=data.values[n_train:n_train+n_val],
            timestamps=data.timestamps[n_train:n_train+n_val] if data.timestamps is not None else None,
            features=data.features[n_train:n_train+n_val] if data.features is not None else None,
            zero_threshold=self.zero_threshold,
            frequency=data.frequency
        )
        
        test_data = StandardTimeSeriesFormat(
            values=data.values[n_train+n_val:],
            timestamps=data.timestamps[n_train+n_val:] if data.timestamps is not None else None,
            features=data.features[n_train+n_val:] if data.features is not None else None,
            zero_threshold=self.zero_threshold,
            frequency=data.frequency
        )
        
        return train_data, val_data, test_data
    
    def _stratified_split(self, data: StandardTimeSeriesFormat, n_total: int
                         ) -> Tuple[StandardTimeSeriesFormat, StandardTimeSeriesFormat, StandardTimeSeriesFormat]:
        """Perform stratified split to maintain zero ratios."""
        # This is a simplified stratified split that tries to maintain
        # similar zero ratios by selecting blocks with similar characteristics
        
        # Calculate local zero ratios in sliding windows
        window_size = max(50, n_total // 20)  # Adaptive window size
        local_zero_ratios = []
        
        for i in range(0, n_total - window_size + 1, window_size // 2):
            window_data = data.values[i:i+window_size]
            zero_ratio = np.mean(np.abs(window_data) <= self.zero_threshold)
            local_zero_ratios.append((i, zero_ratio))
        
        # Sort by zero ratio
        local_zero_ratios.sort(key=lambda x: x[1])
        
        # Distribute blocks to maintain similar ratios
        n_blocks = len(local_zero_ratios)
        n_test_blocks = max(1, int(n_blocks * self.test_size))
        n_val_blocks = max(1, int((n_blocks - n_test_blocks) * self.validation_size))
        
        # Select blocks for each split
        test_indices = set()
        val_indices = set()
        train_indices = set()
        
        # Distribute blocks evenly across splits
        for i, (start_idx, ratio) in enumerate(local_zero_ratios):
            end_idx = min(start_idx + window_size, n_total)
            block_indices = set(range(start_idx, end_idx))
            
            if i % 3 == 0 and len(test_indices) < n_total * self.test_size:
                test_indices.update(block_indices)
            elif i % 3 == 1 and len(val_indices) < n_total * self.validation_size:
                val_indices.update(block_indices)
            else:
                train_indices.update(block_indices)
        
        # Convert to sorted lists
        train_indices = sorted(list(train_indices))
        val_indices = sorted(list(val_indices))
        test_indices = sorted(list(test_indices))
        
        # Create splits
        train_data = StandardTimeSeriesFormat(
            values=data.values[train_indices],
            timestamps=data.timestamps[train_indices] if data.timestamps is not None else None,
            features=data.features[train_indices] if data.features is not None else None,
            zero_threshold=self.zero_threshold
        )
        
        val_data = StandardTimeSeriesFormat(
            values=data.values[val_indices],
            timestamps=data.timestamps[val_indices] if data.timestamps is not None else None,
            features=data.features[val_indices] if data.features is not None else None,
            zero_threshold=self.zero_threshold
        )
        
        test_data = StandardTimeSeriesFormat(
            values=data.values[test_indices],
            timestamps=data.timestamps[test_indices] if data.timestamps is not None else None,
            features=data.features[test_indices] if data.features is not None else None,
            zero_threshold=self.zero_threshold
        )
        
        return train_data, val_data, test_data
    
    def get_split_info(self, n_total: int) -> Dict[str, Any]:
        """
        Get information about the splits.
        
        Args:
            n_total: Total number of samples
            
        Returns:
            Dictionary with split information
        """
        n_test = int(n_total * self.test_size)
        n_train_val = n_total - n_test
        n_val = int(n_train_val * self.validation_size)
        n_train = n_train_val - n_val
        
        return {
            'total_samples': n_total,
            'train_samples': n_train,
            'val_samples': n_val,
            'test_samples': n_test,
            'train_ratio': n_train / n_total,
            'val_ratio': n_val / n_total,
            'test_ratio': n_test / n_total,
            'stratified': self.stratify_zeros,
            'shuffled': self.shuffle
        }