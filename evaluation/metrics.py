"""
Comprehensive metrics for evaluating zero-inflated time series models.

This module provides specialized metrics that are particularly relevant for
zero-inflated data, including standard forecasting metrics, distribution
metrics, and zero-inflation specific metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings


class ZeroInflatedMetrics:
    """
    Comprehensive metrics class for zero-inflated time series evaluation.
    
    This class computes various metrics that are specifically designed for
    evaluating models on zero-inflated data.
    """
    
    def __init__(self, zero_threshold: float = 1e-6):
        """
        Initialize the metrics calculator.
        
        Args:
            zero_threshold: Threshold below which values are considered zero
        """
        self.zero_threshold = zero_threshold
    
    def __call__(self, predictions: np.ndarray, targets: np.ndarray, 
                 return_components: bool = False) -> Dict[str, float]:
        """
        Compute all zero-inflated metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            return_components: Whether to return component-wise metrics
            
        Returns:
            Dictionary of computed metrics
        """
        # Flatten arrays if needed
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        # Compute all metric categories
        forecasting_metrics = self.compute_forecasting_metrics(predictions, targets)
        distribution_metrics = self.compute_distribution_metrics(predictions, targets)
        zero_metrics = self.compute_zero_inflation_metrics(predictions, targets)
        
        # Combine all metrics
        all_metrics = {**forecasting_metrics, **distribution_metrics, **zero_metrics}
        
        if return_components:
            all_metrics['components'] = {
                'forecasting': forecasting_metrics,
                'distribution': distribution_metrics,
                'zero_inflation': zero_metrics
            }
        
        return all_metrics
    
    def compute_forecasting_metrics(self, predictions: np.ndarray, 
                                   targets: np.ndarray) -> Dict[str, float]:
        """Compute standard forecasting metrics."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            metrics = {}
            
            # Basic error metrics
            metrics['mse'] = mean_squared_error(targets, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(targets, predictions)
            
            # Relative error metrics
            non_zero_mask = np.abs(targets) > self.zero_threshold
            if np.any(non_zero_mask):
                relative_errors = np.abs(predictions[non_zero_mask] - targets[non_zero_mask]) / (np.abs(targets[non_zero_mask]) + 1e-8)
                metrics['mape'] = np.mean(relative_errors) * 100
                metrics['median_ape'] = np.median(relative_errors) * 100
            else:
                metrics['mape'] = 0.0
                metrics['median_ape'] = 0.0
            
            # Coefficient of determination
            if np.var(targets) > 1e-10:
                metrics['r2'] = r2_score(targets, predictions)
            else:
                metrics['r2'] = 0.0
            
            # Symmetric Mean Absolute Percentage Error
            denominator = (np.abs(targets) + np.abs(predictions)) / 2 + 1e-8
            metrics['smape'] = np.mean(np.abs(predictions - targets) / denominator) * 100
            
            return metrics
    
    def compute_distribution_metrics(self, predictions: np.ndarray, 
                                   targets: np.ndarray) -> Dict[str, float]:
        """Compute distributional metrics."""
        metrics = {}
        
        # Moments comparison
        metrics['mean_error'] = np.mean(predictions) - np.mean(targets)
        metrics['variance_error'] = np.var(predictions) - np.var(targets)
        metrics['std_error'] = np.std(predictions) - np.std(targets)
        
        # Skewness and kurtosis
        try:
            pred_skew = stats.skew(predictions)
            target_skew = stats.skew(targets)
            metrics['skewness_error'] = pred_skew - target_skew
        except:
            metrics['skewness_error'] = 0.0
        
        try:
            pred_kurt = stats.kurtosis(predictions)
            target_kurt = stats.kurtosis(targets)
            metrics['kurtosis_error'] = pred_kurt - target_kurt
        except:
            metrics['kurtosis_error'] = 0.0
        
        # Quantile comparisons
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            pred_q = np.quantile(predictions, q)
            target_q = np.quantile(targets, q)
            metrics[f'quantile_{int(q*100)}_error'] = pred_q - target_q
        
        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(predictions, targets)
            metrics['ks_statistic'] = ks_stat
            metrics['ks_pvalue'] = ks_pvalue
        except:
            metrics['ks_statistic'] = 1.0
            metrics['ks_pvalue'] = 0.0
        
        return metrics
    
    def compute_zero_inflation_metrics(self, predictions: np.ndarray, 
                                     targets: np.ndarray) -> Dict[str, float]:
        """Compute zero-inflation specific metrics."""
        metrics = {}
        
        # Zero ratio comparison
        pred_zero_ratio = np.mean(np.abs(predictions) <= self.zero_threshold)
        target_zero_ratio = np.mean(np.abs(targets) <= self.zero_threshold)
        metrics['zero_ratio_error'] = pred_zero_ratio - target_zero_ratio
        metrics['predicted_zero_ratio'] = pred_zero_ratio
        metrics['actual_zero_ratio'] = target_zero_ratio
        
        # Binary classification metrics for zero vs non-zero
        pred_binary = (np.abs(predictions) > self.zero_threshold).astype(int)
        target_binary = (np.abs(targets) > self.zero_threshold).astype(int)
        
        metrics['zero_classification_accuracy'] = accuracy_score(target_binary, pred_binary)
        
        if len(np.unique(target_binary)) > 1:  # Check if we have both classes
            metrics['zero_precision'] = precision_score(target_binary, pred_binary, zero_division=0)
            metrics['zero_recall'] = recall_score(target_binary, pred_binary, zero_division=0)
            metrics['zero_f1'] = f1_score(target_binary, pred_binary, zero_division=0)
        else:
            metrics['zero_precision'] = 1.0 if target_binary[0] == pred_binary[0] else 0.0
            metrics['zero_recall'] = metrics['zero_precision']
            metrics['zero_f1'] = metrics['zero_precision']
        
        # Metrics on non-zero values only
        non_zero_mask = np.abs(targets) > self.zero_threshold
        if np.any(non_zero_mask):
            non_zero_pred = predictions[non_zero_mask]
            non_zero_target = targets[non_zero_mask]
            
            metrics['nonzero_mse'] = mean_squared_error(non_zero_target, non_zero_pred)
            metrics['nonzero_mae'] = mean_absolute_error(non_zero_target, non_zero_pred)
            
            if np.var(non_zero_target) > 1e-10:
                metrics['nonzero_r2'] = r2_score(non_zero_target, non_zero_pred)
            else:
                metrics['nonzero_r2'] = 0.0
        else:
            metrics['nonzero_mse'] = 0.0
            metrics['nonzero_mae'] = 0.0
            metrics['nonzero_r2'] = 0.0
        
        # Excess zeros analysis
        metrics['excess_zero_count'] = max(0, pred_zero_ratio - target_zero_ratio)
        
        # Zero-inflation model assessment
        if target_zero_ratio > 0.1:  # Only compute if significantly zero-inflated
            metrics['zero_inflation_severity'] = target_zero_ratio
            metrics['model_zero_handling_quality'] = 1 - abs(metrics['zero_ratio_error'])
        else:
            metrics['zero_inflation_severity'] = 0.0
            metrics['model_zero_handling_quality'] = 1.0
        
        return metrics


def compute_zero_inflation_metrics(predictions: np.ndarray, targets: np.ndarray,
                                 zero_threshold: float = 1e-6) -> Dict[str, float]:
    """
    Convenience function to compute zero-inflation metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        zero_threshold: Threshold for considering values as zero
        
    Returns:
        Dictionary of zero-inflation metrics
    """
    calculator = ZeroInflatedMetrics(zero_threshold=zero_threshold)
    return calculator.compute_zero_inflation_metrics(predictions, targets)


def compute_distribution_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Convenience function to compute distribution metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary of distribution metrics
    """
    calculator = ZeroInflatedMetrics()
    return calculator.compute_distribution_metrics(predictions, targets)


def compute_forecasting_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Convenience function to compute standard forecasting metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary of forecasting metrics
    """
    calculator = ZeroInflatedMetrics()
    return calculator.compute_forecasting_metrics(predictions, targets)


def compute_torch_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                         zero_threshold: float = 1e-6) -> Dict[str, float]:
    """
    Compute metrics for PyTorch tensors.
    
    Args:
        predictions: Model predictions as torch tensor
        targets: Ground truth targets as torch tensor
        zero_threshold: Threshold for considering values as zero
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    # Compute metrics
    calculator = ZeroInflatedMetrics(zero_threshold=zero_threshold)
    return calculator(pred_np, target_np)


class MetricsTracker:
    """
    Track metrics over multiple evaluation runs.
    
    Useful for computing statistics over multiple folds, experiments, or epochs.
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def add_metrics(self, metrics: Dict[str, float]):
        """Add a set of metrics to the tracker."""
        self.metrics_history.append(metrics)
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics (mean, std, min, max) for all tracked metrics.
        
        Returns:
            Dictionary with statistics for each metric
        """
        if not self.metrics_history:
            return {}
        
        # Get all metric names
        all_metric_names = set()
        for metrics in self.metrics_history:
            all_metric_names.update(metrics.keys())
        
        summary = {}
        for metric_name in all_metric_names:
            values = [metrics.get(metric_name, np.nan) for metrics in self.metrics_history]
            values = [v for v in values if not np.isnan(v)]  # Remove NaN values
            
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values)
                }
            else:
                summary[metric_name] = {
                    'mean': np.nan, 'std': np.nan, 'min': np.nan,
                    'max': np.nan, 'median': np.nan, 'count': 0
                }
        
        return summary
    
    def clear(self):
        """Clear all tracked metrics."""
        self.metrics_history = []


def format_metrics_report(metrics: Dict[str, float], title: str = "Metrics Report") -> str:
    """
    Format metrics into a readable report.
    
    Args:
        metrics: Dictionary of metrics
        title: Report title
        
    Returns:
        Formatted string report
    """
    report = [f"\n{title}", "=" * len(title)]
    
    # Group metrics by category (exclude components)
    metrics_only = {k: v for k, v in metrics.items() if k != 'components' and isinstance(v, (int, float))}
    forecasting_metrics = {k: v for k, v in metrics_only.items() if k in ['mse', 'rmse', 'mae', 'mape', 'r2', 'smape']}
    distribution_metrics = {k: v for k, v in metrics_only.items() if 'error' in k or 'quantile' in k or 'ks_' in k}
    zero_metrics = {k: v for k, v in metrics_only.items() if 'zero' in k or 'nonzero' in k}
    other_metrics = {k: v for k, v in metrics_only.items() if k not in forecasting_metrics and k not in distribution_metrics and k not in zero_metrics}
    
    # Format each category
    if forecasting_metrics:
        report.append("\nForecasting Metrics:")
        for k, v in forecasting_metrics.items():
            report.append(f"  {k}: {v:.6f}")
    
    if zero_metrics:
        report.append("\nZero-Inflation Metrics:")
        for k, v in zero_metrics.items():
            report.append(f"  {k}: {v:.6f}")
    
    if distribution_metrics:
        report.append("\nDistribution Metrics:")
        for k, v in distribution_metrics.items():
            report.append(f"  {k}: {v:.6f}")
    
    if other_metrics:
        report.append("\nOther Metrics:")
        for k, v in other_metrics.items():
            report.append(f"  {k}: {v:.6f}")
    
    return "\n".join(report)


def compare_models_metrics(model_metrics: Dict[str, Dict[str, float]], 
                          metric_names: Optional[list] = None) -> str:
    """
    Compare metrics across multiple models.
    
    Args:
        model_metrics: Dictionary mapping model names to their metrics
        metric_names: Specific metrics to compare (if None, compare all)
        
    Returns:
        Formatted comparison report
    """
    if not model_metrics:
        return "No models to compare."
    
    # Get all metric names if not specified
    if metric_names is None:
        metric_names = set()
        for metrics in model_metrics.values():
            metric_names.update(metrics.keys())
        metric_names = sorted(list(metric_names))
    
    # Create comparison table
    report = ["\nModel Comparison", "=" * 50]
    
    # Header
    header = f"{'Metric':<25}"
    for model_name in model_metrics.keys():
        header += f"{model_name:<15}"
    report.append(header)
    report.append("-" * len(header))
    
    # Metrics rows
    for metric_name in metric_names:
        row = f"{metric_name:<25}"
        for model_name, metrics in model_metrics.items():
            value = metrics.get(metric_name, np.nan)
            if not np.isnan(value):
                row += f"{value:<15.6f}"
            else:
                row += f"{'N/A':<15}"
        report.append(row)
    
    # Find best model for each metric
    report.append("\nBest Model per Metric:")
    report.append("-" * 30)
    
    for metric_name in metric_names:
        values = {}
        for model_name, metrics in model_metrics.items():
            value = metrics.get(metric_name, np.nan)
            if not np.isnan(value):
                values[model_name] = value
        
        if values:
            # Determine if lower or higher is better
            lower_is_better = any(x in metric_name.lower() for x in ['mse', 'mae', 'error', 'ks_stat'])
            
            if lower_is_better:
                best_model = min(values.items(), key=lambda x: x[1])
            else:
                best_model = max(values.items(), key=lambda x: x[1])
            
            report.append(f"{metric_name:<25}: {best_model[0]} ({best_model[1]:.6f})")
    
    return "\n".join(report)