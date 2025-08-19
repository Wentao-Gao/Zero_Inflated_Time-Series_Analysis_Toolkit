"""
Benchmark suite for zero-inflated time series models.

This module provides standardized benchmarks and datasets for evaluating
zero-inflated time series models in a fair and consistent manner.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Callable
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from dataclasses import dataclass
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from generation.inject_zeros import inject_zeros
from generation.zero_mechanisms import ThresholdZeroInflation, MixtureZeroInflation
from .evaluator import ModelEvaluator, ComprehensiveEvaluation
from .metrics import ZeroInflatedMetrics


@dataclass
class BenchmarkDataset:
    """Container for benchmark dataset information."""
    name: str
    data: np.ndarray
    description: str
    zero_ratio: float
    mechanism: str
    n_samples: int
    metadata: Dict[str, Any]


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for zero-inflated time series models.
    
    This class provides standardized datasets and evaluation procedures
    for fair comparison of different modeling approaches.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize benchmark suite.
        
        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        self.datasets = {}
        self.results = {}
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    def create_synthetic_datasets(self) -> Dict[str, BenchmarkDataset]:
        """
        Create a suite of synthetic benchmark datasets with different characteristics.
        
        Returns:
            Dictionary of benchmark datasets
        """
        datasets = {}
        
        # Dataset 1: Low zero-inflation with trend
        trend_data = self._generate_trend_series(n_samples=2000, trend_strength=0.001)
        trend_zi = inject_zeros(trend_data, mechanism='threshold', zero_ratio=0.15, 
                               random_state=self.random_state)
        datasets['low_zi_trend'] = BenchmarkDataset(
            name='Low Zero-Inflation with Trend',
            data=trend_zi,
            description='Time series with upward trend and 15% zero-inflation',
            zero_ratio=np.mean(trend_zi == 0),
            mechanism='threshold',
            n_samples=len(trend_zi),
            metadata={'trend_strength': 0.001, 'seasonality': False}
        )
        
        # Dataset 2: High zero-inflation seasonal
        seasonal_data = self._generate_seasonal_series(n_samples=2000, seasonal_strength=2.0)
        seasonal_zi = inject_zeros(seasonal_data, mechanism='mixture', zero_ratio=0.45, 
                                  random_state=self.random_state)
        datasets['high_zi_seasonal'] = BenchmarkDataset(
            name='High Zero-Inflation with Seasonality',
            data=seasonal_zi,
            description='Seasonal time series with 45% zero-inflation',
            zero_ratio=np.mean(seasonal_zi == 0),
            mechanism='mixture',
            n_samples=len(seasonal_zi),
            metadata={'seasonal_period': 365.25, 'trend_strength': 0.0}
        )
        
        # Dataset 3: Moderate zero-inflation with noise
        noise_data = self._generate_noise_series(n_samples=1500, noise_strength=1.0)
        noise_zi = inject_zeros(noise_data, mechanism='tweedie', zero_ratio=0.30, 
                               random_state=self.random_state)
        datasets['moderate_zi_noise'] = BenchmarkDataset(
            name='Moderate Zero-Inflation with Noise',
            data=noise_zi,
            description='Noisy time series with 30% zero-inflation',
            zero_ratio=np.mean(noise_zi == 0),
            mechanism='tweedie',
            n_samples=len(noise_zi),
            metadata={'noise_strength': 1.0, 'base_mean': 3.0}
        )
        
        # Dataset 4: Mixed patterns
        mixed_data = self._generate_mixed_series(n_samples=1800)
        mixed_zi = inject_zeros(mixed_data, mechanism='hurdle', zero_ratio=0.25, 
                               random_state=self.random_state)
        datasets['mixed_patterns'] = BenchmarkDataset(
            name='Mixed Patterns',
            data=mixed_zi,
            description='Complex time series with multiple patterns and 25% zero-inflation',
            zero_ratio=np.mean(mixed_zi == 0),
            mechanism='hurdle',
            n_samples=len(mixed_zi),
            metadata={'pattern_types': ['trend', 'seasonal', 'cyclic']}
        )
        
        # Dataset 5: Extreme zero-inflation
        extreme_data = self._generate_baseline_series(n_samples=1200, base_value=2.0)
        extreme_zi = inject_zeros(extreme_data, mechanism='mixture', zero_ratio=0.70, 
                                 random_state=self.random_state)
        datasets['extreme_zi'] = BenchmarkDataset(
            name='Extreme Zero-Inflation',
            data=extreme_zi,
            description='Simple time series with extreme 70% zero-inflation',
            zero_ratio=np.mean(extreme_zi == 0),
            mechanism='mixture',
            n_samples=len(extreme_zi),
            metadata={'base_value': 2.0, 'pattern': 'constant'}
        )
        
        self.datasets = datasets
        return datasets
    
    def _generate_trend_series(self, n_samples: int, trend_strength: float) -> np.ndarray:
        """Generate time series with trend component."""
        t = np.arange(n_samples)
        trend = trend_strength * t
        noise = np.random.normal(0, 0.5, n_samples)
        base = 3.0
        series = base + trend + noise
        return np.maximum(series, 0)
    
    def _generate_seasonal_series(self, n_samples: int, seasonal_strength: float) -> np.ndarray:
        """Generate time series with seasonal component."""
        t = np.arange(n_samples)
        # Annual and weekly seasonality
        annual_season = seasonal_strength * np.sin(2 * np.pi * t / 365.25)
        weekly_season = 0.5 * seasonal_strength * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 0.3, n_samples)
        base = 4.0
        series = base + annual_season + weekly_season + noise
        return np.maximum(series, 0)
    
    def _generate_noise_series(self, n_samples: int, noise_strength: float) -> np.ndarray:
        """Generate time series with high noise."""
        noise = np.random.normal(0, noise_strength, n_samples)
        base = 3.0
        series = base + noise
        return np.maximum(series, 0)
    
    def _generate_mixed_series(self, n_samples: int) -> np.ndarray:
        """Generate time series with mixed patterns."""
        t = np.arange(n_samples)
        
        # Multiple components
        trend = 0.0005 * t
        seasonal = np.sin(2 * np.pi * t / 100) + 0.5 * np.sin(2 * np.pi * t / 20)
        cyclic = 0.3 * np.sin(2 * np.pi * t / 300)
        noise = np.random.normal(0, 0.4, n_samples)
        
        base = 3.5
        series = base + trend + seasonal + cyclic + noise
        return np.maximum(series, 0)
    
    def _generate_baseline_series(self, n_samples: int, base_value: float) -> np.ndarray:
        """Generate simple baseline time series."""
        noise = np.random.normal(0, 0.2, n_samples)
        series = base_value + noise
        return np.maximum(series, 0)
    
    def run_benchmark(self, models: Dict[str, Any], 
                     dataset_names: Optional[List[str]] = None,
                     test_split: float = 0.3,
                     sequence_length: int = 48,
                     prediction_horizon: int = 12) -> Dict[str, Any]:
        """
        Run comprehensive benchmark on specified models and datasets.
        
        Args:
            models: Dictionary of models to benchmark
            dataset_names: List of dataset names to use (all if None)
            test_split: Fraction of data for testing
            sequence_length: Length of input sequences
            prediction_horizon: Length of prediction horizon
            
        Returns:
            Comprehensive benchmark results
        """
        if not self.datasets:
            self.create_synthetic_datasets()
        
        if dataset_names is None:
            dataset_names = list(self.datasets.keys())
        
        results = {
            'benchmark_config': {
                'test_split': test_split,
                'sequence_length': sequence_length,
                'prediction_horizon': prediction_horizon,
                'random_state': self.random_state
            },
            'datasets': {},
            'models': list(models.keys()),
            'summary': {}
        }
        
        evaluator = ModelEvaluator()
        
        for dataset_name in dataset_names:
            if dataset_name not in self.datasets:
                print(f"Warning: Dataset '{dataset_name}' not found, skipping...")
                continue
            
            dataset = self.datasets[dataset_name]
            print(f"\nBenchmarking on dataset: {dataset.name}")
            print(f"Zero ratio: {dataset.zero_ratio:.3f}, Samples: {dataset.n_samples}")
            
            # Prepare time series data
            X, y = self._prepare_sequences(
                dataset.data, sequence_length, prediction_horizon
            )
            
            # Split data
            split_idx = int(len(X) * (1 - test_split))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            dataset_results = {
                'dataset_info': {
                    'name': dataset.name,
                    'zero_ratio': dataset.zero_ratio,
                    'mechanism': dataset.mechanism,
                    'n_samples': dataset.n_samples,
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                },
                'model_results': {}
            }
            
            # Evaluate each model
            for model_name, model in models.items():
                print(f"  Evaluating {model_name}...")
                
                try:
                    start_time = time.time()
                    
                    # Train model if it has a fit method
                    if hasattr(model, 'fit'):
                        model.fit(X_train, y_train)
                    
                    # Evaluate
                    result = evaluator.evaluate_model(
                        model, X_test, y_test, model_name=model_name
                    )
                    
                    dataset_results['model_results'][model_name] = {
                        'metrics': result.metrics,
                        'evaluation_time': result.evaluation_time,
                        'success': True
                    }
                    
                    print(f"    MSE: {result.metrics.get('mse', 'N/A'):.6f}, "
                          f"Zero Acc: {result.metrics.get('zero_classification_accuracy', 'N/A'):.3f}")
                    
                except Exception as e:
                    print(f"    Failed: {str(e)}")
                    dataset_results['model_results'][model_name] = {
                        'error': str(e),
                        'success': False
                    }
            
            results['datasets'][dataset_name] = dataset_results
        
        # Generate summary
        results['summary'] = self._generate_benchmark_summary(results)
        self.results = results
        
        return results
    
    def _prepare_sequences(self, data: np.ndarray, seq_len: int, 
                          pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for time series modeling.
        
        Args:
            data: Time series data
            seq_len: Input sequence length
            pred_len: Prediction sequence length
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len:i+seq_len+pred_len])
        
        return np.array(X), np.array(y)
    
    def _generate_benchmark_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for benchmark results."""
        summary = {
            'total_datasets': len(results['datasets']),
            'total_models': len(results['models']),
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'best_models_by_dataset': {},
            'overall_rankings': {}
        }
        
        # Count successes and failures
        for dataset_name, dataset_results in results['datasets'].items():
            for model_name, model_result in dataset_results['model_results'].items():
                if model_result.get('success', False):
                    summary['successful_evaluations'] += 1
                else:
                    summary['failed_evaluations'] += 1
        
        # Find best models for each dataset
        for dataset_name, dataset_results in results['datasets'].items():
            successful_models = {
                name: result['metrics'] 
                for name, result in dataset_results['model_results'].items()
                if result.get('success', False)
            }
            
            if successful_models:
                # Find best model by MSE (lower is better)
                best_model = min(successful_models.items(), 
                               key=lambda x: x[1].get('mse', float('inf')))
                summary['best_models_by_dataset'][dataset_name] = {
                    'model': best_model[0],
                    'mse': best_model[1].get('mse', 'N/A')
                }
        
        # Overall model rankings
        model_scores = {}
        for model_name in results['models']:
            scores = []
            for dataset_results in results['datasets'].values():
                if model_name in dataset_results['model_results']:
                    result = dataset_results['model_results'][model_name]
                    if result.get('success', False):
                        scores.append(result['metrics'].get('mse', float('inf')))
            
            if scores:
                model_scores[model_name] = np.mean(scores)
        
        if model_scores:
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
            summary['overall_rankings'] = {
                f'rank_{i+1}': {'model': model, 'avg_mse': score}
                for i, (model, score) in enumerate(sorted_models)
            }
        
        return summary
    
    def generate_benchmark_report(self, results: Optional[Dict[str, Any]] = None,
                                 save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            results: Benchmark results to report on
            save_path: Path to save the report
            
        Returns:
            Formatted report string
        """
        if results is None:
            results = self.results
        
        if not results:
            return "No benchmark results available."
        
        report = []
        report.append("ZERO-INFLATED TIME SERIES MODEL BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Configuration
        config = results.get('benchmark_config', {})
        report.append("BENCHMARK CONFIGURATION")
        report.append("-" * 30)
        report.append(f"Test Split: {config.get('test_split', 'Unknown')}")
        report.append(f"Sequence Length: {config.get('sequence_length', 'Unknown')}")
        report.append(f"Prediction Horizon: {config.get('prediction_horizon', 'Unknown')}")
        report.append(f"Random State: {config.get('random_state', 'Unknown')}")
        report.append("")
        
        # Summary
        summary = results.get('summary', {})
        report.append("BENCHMARK SUMMARY")
        report.append("-" * 25)
        report.append(f"Datasets: {summary.get('total_datasets', 'Unknown')}")
        report.append(f"Models: {summary.get('total_models', 'Unknown')}")
        report.append(f"Successful Evaluations: {summary.get('successful_evaluations', 0)}")
        report.append(f"Failed Evaluations: {summary.get('failed_evaluations', 0)}")
        report.append("")
        
        # Overall rankings
        if 'overall_rankings' in summary and summary['overall_rankings']:
            report.append("OVERALL MODEL RANKINGS (by average MSE)")
            report.append("-" * 45)
            
            for rank, info in summary['overall_rankings'].items():
                rank_num = rank.split('_')[1]
                report.append(f"{rank_num}. {info['model']}: {info['avg_mse']:.6f}")
            
            report.append("")
        
        # Dataset-specific results
        report.append("DATASET-SPECIFIC RESULTS")
        report.append("-" * 35)
        
        for dataset_name, dataset_results in results.get('datasets', {}).items():
            dataset_info = dataset_results.get('dataset_info', {})
            report.append(f"\n{dataset_info.get('name', dataset_name)}:")
            report.append(f"  Zero Ratio: {dataset_info.get('zero_ratio', 'Unknown'):.3f}")
            report.append(f"  Mechanism: {dataset_info.get('mechanism', 'Unknown')}")
            report.append(f"  Test Samples: {dataset_info.get('test_samples', 'Unknown')}")
            
            # Model results for this dataset
            successful_results = {}
            for model_name, model_result in dataset_results.get('model_results', {}).items():
                if model_result.get('success', False):
                    metrics = model_result['metrics']
                    successful_results[model_name] = metrics.get('mse', float('inf'))
                    report.append(f"    {model_name}: MSE={metrics.get('mse', 'N/A'):.6f}, "
                                f"Zero_Acc={metrics.get('zero_classification_accuracy', 'N/A'):.3f}")
                else:
                    report.append(f"    {model_name}: FAILED")
            
            # Best model for this dataset
            if successful_results:
                best_model = min(successful_results.items(), key=lambda x: x[1])
                report.append(f"  → Best Model: {best_model[0]} (MSE: {best_model[1]:.6f})")
        
        report_text = "\n".join(report)
        
        # Save if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


class StandardBenchmarks:
    """
    Standard benchmark configurations for common use cases.
    
    This class provides pre-configured benchmark suites for different
    types of zero-inflated time series analysis scenarios.
    """
    
    @staticmethod
    def quick_benchmark(models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quick benchmark with basic datasets.
        
        Args:
            models: Dictionary of models to benchmark
            
        Returns:
            Benchmark results
        """
        suite = BenchmarkSuite(random_state=42)
        
        # Create subset of datasets for quick testing
        suite.create_synthetic_datasets()
        quick_datasets = ['low_zi_trend', 'moderate_zi_noise', 'high_zi_seasonal']
        
        return suite.run_benchmark(
            models=models,
            dataset_names=quick_datasets,
            test_split=0.3,
            sequence_length=24,
            prediction_horizon=6
        )
    
    @staticmethod
    def comprehensive_benchmark(models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive benchmark with all datasets and longer sequences.
        
        Args:
            models: Dictionary of models to benchmark
            
        Returns:
            Benchmark results
        """
        suite = BenchmarkSuite(random_state=42)
        suite.create_synthetic_datasets()
        
        return suite.run_benchmark(
            models=models,
            test_split=0.2,
            sequence_length=96,
            prediction_horizon=24
        )
    
    @staticmethod
    def zero_inflation_focused_benchmark(models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Benchmark focused on different levels of zero-inflation.
        
        Args:
            models: Dictionary of models to benchmark
            
        Returns:
            Benchmark results
        """
        suite = BenchmarkSuite(random_state=42)
        suite.create_synthetic_datasets()
        
        # Focus on datasets with varying zero-inflation levels
        zi_datasets = ['low_zi_trend', 'moderate_zi_noise', 'high_zi_seasonal', 'extreme_zi']
        
        return suite.run_benchmark(
            models=models,
            dataset_names=zi_datasets,
            test_split=0.25,
            sequence_length=48,
            prediction_horizon=12
        )


def create_benchmark_report(benchmark_results: Dict[str, Any], 
                           report_path: str) -> str:
    """
    Create and save a benchmark report.
    
    Args:
        benchmark_results: Results from benchmark suite
        report_path: Path to save the report
        
    Returns:
        Report content as string
    """
    suite = BenchmarkSuite()
    report = suite.generate_benchmark_report(benchmark_results, save_path=report_path)
    return report