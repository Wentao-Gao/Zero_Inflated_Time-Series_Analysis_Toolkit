"""
Model evaluation framework for zero-inflated time series models.

This module provides comprehensive evaluation tools including cross-validation,
holdout validation, and comparative analysis of multiple models.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from sklearn.model_selection import TimeSeriesSplit, KFold
import warnings
from dataclasses import dataclass
import time
import json

from .metrics import ZeroInflatedMetrics, MetricsTracker, compute_torch_metrics


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metrics: Dict[str, float]
    predictions: np.ndarray
    targets: np.ndarray
    model_name: str
    evaluation_time: float
    metadata: Dict[str, Any]


class ModelEvaluator:
    """
    Comprehensive model evaluation framework for zero-inflated time series.
    
    This class provides methods for evaluating models using various validation
    strategies and comprehensive metrics.
    """
    
    def __init__(self, metrics_calculator: Optional[ZeroInflatedMetrics] = None,
                 device: str = 'cpu'):
        """
        Initialize the evaluator.
        
        Args:
            metrics_calculator: Custom metrics calculator
            device: Device to run evaluations on
        """
        self.metrics_calculator = metrics_calculator or ZeroInflatedMetrics()
        self.device = device
        self.results_history = []
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                      model_name: str = "Unknown",
                      prediction_method: str = "predict",
                      **kwargs) -> EvaluationResult:
        """
        Evaluate a single model on given data.
        
        Args:
            model: The model to evaluate
            X: Input features
            y: Target values
            model_name: Name of the model for reporting
            prediction_method: Method name to call for predictions
            **kwargs: Additional arguments for prediction method
            
        Returns:
            EvaluationResult object
        """
        start_time = time.time()
        
        try:
            # Make predictions
            if hasattr(model, prediction_method):
                predictions = getattr(model, prediction_method)(X, **kwargs)
            elif callable(model):
                predictions = model(X, **kwargs)
            else:
                raise ValueError(f"Model does not have method '{prediction_method}' and is not callable")
            
            # Convert to numpy if needed
            if torch.is_tensor(predictions):
                predictions = predictions.detach().cpu().numpy()
            if torch.is_tensor(y):
                y = y.detach().cpu().numpy()
            
            # Ensure correct shapes
            predictions = np.array(predictions).flatten()
            y = np.array(y).flatten()
            
            # Compute metrics
            metrics = self.metrics_calculator(predictions, y)
            
            evaluation_time = time.time() - start_time
            
            result = EvaluationResult(
                metrics=metrics,
                predictions=predictions,
                targets=y,
                model_name=model_name,
                evaluation_time=evaluation_time,
                metadata={
                    'prediction_method': prediction_method,
                    'kwargs': kwargs,
                    'n_samples': len(y),
                    'zero_ratio': np.mean(np.abs(y) <= self.metrics_calculator.zero_threshold)
                }
            )
            
            self.results_history.append(result)
            return result
            
        except Exception as e:
            evaluation_time = time.time() - start_time
            return EvaluationResult(
                metrics={'error': 1.0},
                predictions=np.array([]),
                targets=y,
                model_name=model_name,
                evaluation_time=evaluation_time,
                metadata={'error': str(e), 'failed': True}
            )
    
    def evaluate_pytorch_model(self, model: nn.Module, dataloader,
                              model_name: str = "PyTorch Model",
                              device: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate a PyTorch model using a DataLoader.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader with evaluation data
            model_name: Name of the model
            device: Device to use (defaults to self.device)
            
        Returns:
            EvaluationResult object
        """
        if device is None:
            device = self.device
        
        model.eval()
        model.to(device)
        
        start_time = time.time()
        all_predictions = []
        all_targets = []
        
        try:
            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch[0].to(device), batch[1].to(device)
                    else:
                        # Assume batch is just inputs, targets are the next values
                        inputs = batch.to(device)
                        targets = inputs[:, 1:, :]  # Simple assumption
                        inputs = inputs[:, :-1, :]
                    
                    # Make predictions
                    if hasattr(model, 'forward'):
                        predictions = model(inputs)
                    else:
                        predictions = model(inputs)
                    
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
            
            # Concatenate all predictions and targets
            predictions = torch.cat(all_predictions, dim=0).numpy().flatten()
            targets = torch.cat(all_targets, dim=0).numpy().flatten()
            
            # Compute metrics
            metrics = self.metrics_calculator(predictions, targets)
            
            evaluation_time = time.time() - start_time
            
            result = EvaluationResult(
                metrics=metrics,
                predictions=predictions,
                targets=targets,
                model_name=model_name,
                evaluation_time=evaluation_time,
                metadata={
                    'model_type': 'pytorch',
                    'n_batches': len(dataloader),
                    'n_samples': len(targets),
                    'zero_ratio': np.mean(np.abs(targets) <= self.metrics_calculator.zero_threshold)
                }
            )
            
            self.results_history.append(result)
            return result
            
        except Exception as e:
            evaluation_time = time.time() - start_time
            return EvaluationResult(
                metrics={'error': 1.0},
                predictions=np.array([]),
                targets=np.array([]),
                model_name=model_name,
                evaluation_time=evaluation_time,
                metadata={'error': str(e), 'failed': True}
            )
    
    def compare_models(self, results: List[EvaluationResult],
                      key_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.
        
        Args:
            results: List of evaluation results
            key_metrics: Specific metrics to focus on for comparison
            
        Returns:
            Dictionary with comparison results
        """
        if not results:
            return {}
        
        if key_metrics is None:
            key_metrics = ['mse', 'mae', 'r2', 'zero_classification_accuracy', 'nonzero_r2']
        
        comparison = {
            'model_names': [r.model_name for r in results],
            'metrics_comparison': {},
            'rankings': {},
            'summary': {}
        }
        
        # Extract metrics for each model
        model_metrics = {}
        for result in results:
            if 'error' not in result.metrics:  # Skip failed evaluations
                model_metrics[result.model_name] = result.metrics
        
        if not model_metrics:
            return comparison
        
        # Compare metrics
        comparison['metrics_comparison'] = model_metrics
        
        # Rank models for each key metric
        for metric in key_metrics:
            if metric in model_metrics[list(model_metrics.keys())[0]]:
                metric_values = {}
                for model_name, metrics in model_metrics.items():
                    if metric in metrics:
                        metric_values[model_name] = metrics[metric]
                
                if metric_values:
                    # Determine ranking order (lower is better for error metrics)
                    lower_is_better = any(x in metric.lower() for x in ['mse', 'mae', 'error'])
                    
                    sorted_models = sorted(metric_values.items(), 
                                         key=lambda x: x[1],
                                         reverse=not lower_is_better)
                    
                    comparison['rankings'][metric] = [
                        {'model': model, 'value': value, 'rank': i + 1}
                        for i, (model, value) in enumerate(sorted_models)
                    ]
        
        # Overall summary
        comparison['summary'] = {
            'total_models': len(model_metrics),
            'evaluation_times': {r.model_name: r.evaluation_time for r in results},
            'best_models': {}
        }
        
        # Find best model for each key metric
        for metric, rankings in comparison['rankings'].items():
            if rankings:
                comparison['summary']['best_models'][metric] = rankings[0]['model']
        
        return comparison
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed."""
        if not self.results_history:
            return {'message': 'No evaluations performed yet'}
        
        successful_results = [r for r in self.results_history if 'error' not in r.metrics]
        failed_results = [r for r in self.results_history if 'error' in r.metrics]
        
        return {
            'total_evaluations': len(self.results_history),
            'successful_evaluations': len(successful_results),
            'failed_evaluations': len(failed_results),
            'average_evaluation_time': np.mean([r.evaluation_time for r in self.results_history]),
            'model_names': list(set(r.model_name for r in self.results_history)),
            'failed_models': [r.model_name for r in failed_results] if failed_results else []
        }
    
    def clear_history(self):
        """Clear evaluation history."""
        self.results_history = []


class CrossValidationEvaluator:
    """
    Cross-validation evaluator for time series models.
    
    Provides time series-aware cross-validation strategies.
    """
    
    def __init__(self, cv_strategy: str = 'time_series', n_splits: int = 5,
                 metrics_calculator: Optional[ZeroInflatedMetrics] = None):
        """
        Initialize cross-validation evaluator.
        
        Args:
            cv_strategy: Cross-validation strategy ('time_series' or 'kfold')
            n_splits: Number of CV splits
            metrics_calculator: Custom metrics calculator
        """
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.metrics_calculator = metrics_calculator or ZeroInflatedMetrics()
        self.tracker = MetricsTracker()
    
    def cross_validate(self, model_factory: Callable, X: np.ndarray, y: np.ndarray,
                      model_name: str = "Model",
                      fit_method: str = "fit",
                      predict_method: str = "predict",
                      **model_kwargs) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model_factory: Function that creates a new model instance
            X: Input features
            y: Target values
            model_name: Name of the model
            fit_method: Method name for training
            predict_method: Method name for prediction
            **model_kwargs: Additional arguments for model factory
            
        Returns:
            Dictionary with cross-validation results
        """
        self.tracker.clear()
        
        # Choose CV strategy
        if self.cv_strategy == 'time_series':
            cv_splitter = TimeSeriesSplit(n_splits=self.n_splits)
        else:
            cv_splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try:
                # Create and train model
                model = model_factory(**model_kwargs)
                
                # Fit the model
                if hasattr(model, fit_method):
                    getattr(model, fit_method)(X_train, y_train)
                else:
                    raise ValueError(f"Model does not have method '{fit_method}'")
                
                # Make predictions
                if hasattr(model, predict_method):
                    predictions = getattr(model, predict_method)(X_test)
                else:
                    raise ValueError(f"Model does not have method '{predict_method}'")
                
                # Convert to numpy if needed
                if torch.is_tensor(predictions):
                    predictions = predictions.detach().cpu().numpy()
                
                predictions = np.array(predictions).flatten()
                y_test = np.array(y_test).flatten()
                
                # Compute metrics
                metrics = self.metrics_calculator(predictions, y_test)
                metrics['fold'] = fold
                
                self.tracker.add_metrics(metrics)
                fold_results.append({
                    'fold': fold,
                    'metrics': metrics,
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                })
                
            except Exception as e:
                fold_results.append({
                    'fold': fold,
                    'metrics': {'error': str(e)},
                    'train_size': len(X_train) if 'X_train' in locals() else 0,
                    'test_size': len(X_test) if 'X_test' in locals() else 0
                })
        
        # Get summary statistics
        summary_stats = self.tracker.get_summary_statistics()
        
        return {
            'model_name': model_name,
            'cv_strategy': self.cv_strategy,
            'n_splits': self.n_splits,
            'fold_results': fold_results,
            'summary_statistics': summary_stats,
            'successful_folds': len([r for r in fold_results if 'error' not in r['metrics']])
        }


class ComprehensiveEvaluation:
    """
    Comprehensive evaluation suite combining multiple evaluation strategies.
    
    This class orchestrates various evaluation approaches to provide a complete
    assessment of zero-inflated time series models.
    """
    
    def __init__(self):
        self.model_evaluator = ModelEvaluator()
        self.cv_evaluator = CrossValidationEvaluator()
        self.results = {}
    
    def full_evaluation(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                       test_split: float = 0.2, 
                       perform_cv: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of multiple models.
        
        Args:
            models: Dictionary mapping model names to model instances/factories
            X: Input features
            y: Target values
            test_split: Fraction of data to use for testing
            perform_cv: Whether to perform cross-validation
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            'holdout_results': {},
            'cv_results': {},
            'comparison': {},
            'summary': {}
        }
        
        # Split data for holdout validation
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        holdout_results = []
        
        # Holdout evaluation
        for model_name, model in models.items():
            # Assume model needs to be fitted first
            try:
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                
                result = self.model_evaluator.evaluate_model(
                    model, X_test, y_test, model_name=model_name
                )
                results['holdout_results'][model_name] = result
                holdout_results.append(result)
                
            except Exception as e:
                print(f"Failed to evaluate {model_name} in holdout: {e}")
        
        # Cross-validation (if requested)
        if perform_cv:
            for model_name, model_factory in models.items():
                if callable(model_factory):  # Assume it's a factory function
                    try:
                        cv_result = self.cv_evaluator.cross_validate(
                            model_factory, X_train, y_train, model_name=model_name
                        )
                        results['cv_results'][model_name] = cv_result
                    except Exception as e:
                        print(f"Failed to cross-validate {model_name}: {e}")
        
        # Model comparison
        if holdout_results:
            results['comparison'] = self.model_evaluator.compare_models(holdout_results)
        
        # Overall summary
        results['summary'] = {
            'n_models_evaluated': len(models),
            'data_split': {'train': len(X_train), 'test': len(X_test)},
            'holdout_successful': len(results['holdout_results']),
            'cv_successful': len(results['cv_results']),
            'evaluation_timestamp': time.time()
        }
        
        self.results = results
        return results
    
    def generate_report(self, results: Optional[Dict[str, Any]] = None,
                       save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Results to generate report for (uses self.results if None)
            save_path: Path to save the report
            
        Returns:
            Formatted report string
        """
        if results is None:
            results = self.results
        
        if not results:
            return "No evaluation results available."
        
        report = []
        report.append("COMPREHENSIVE ZERO-INFLATED TIME SERIES MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        if 'summary' in results:
            summary = results['summary']
            report.append("EVALUATION SUMMARY")
            report.append("-" * 30)
            report.append(f"Models Evaluated: {summary.get('n_models_evaluated', 'Unknown')}")
            report.append(f"Training Samples: {summary.get('data_split', {}).get('train', 'Unknown')}")
            report.append(f"Test Samples: {summary.get('data_split', {}).get('test', 'Unknown')}")
            report.append(f"Successful Holdout Evaluations: {summary.get('holdout_successful', 0)}")
            report.append(f"Successful CV Evaluations: {summary.get('cv_successful', 0)}")
            report.append("")
        
        # Holdout results
        if 'holdout_results' in results and results['holdout_results']:
            report.append("HOLDOUT VALIDATION RESULTS")
            report.append("-" * 40)
            
            for model_name, result in results['holdout_results'].items():
                report.append(f"\n{model_name}:")
                report.append(f"  Evaluation Time: {result.evaluation_time:.3f}s")
                report.append(f"  Samples: {len(result.targets)}")
                
                # Key metrics
                key_metrics = ['mse', 'mae', 'r2', 'zero_classification_accuracy', 'actual_zero_ratio']
                for metric in key_metrics:
                    if metric in result.metrics:
                        report.append(f"  {metric}: {result.metrics[metric]:.6f}")
            
            report.append("")
        
        # Model comparison
        if 'comparison' in results and results['comparison']:
            comp = results['comparison']
            if 'best_models' in comp.get('summary', {}):
                report.append("BEST MODELS BY METRIC")
                report.append("-" * 30)
                
                for metric, best_model in comp['summary']['best_models'].items():
                    report.append(f"{metric}: {best_model}")
                
                report.append("")
        
        # Cross-validation results summary
        if 'cv_results' in results and results['cv_results']:
            report.append("CROSS-VALIDATION SUMMARY")
            report.append("-" * 35)
            
            for model_name, cv_result in results['cv_results'].items():
                summary_stats = cv_result.get('summary_statistics', {})
                successful_folds = cv_result.get('successful_folds', 0)
                
                report.append(f"\n{model_name}:")
                report.append(f"  Successful Folds: {successful_folds}/{cv_result.get('n_splits', 'Unknown')}")
                
                # Key metrics means and stds
                key_metrics = ['mse', 'mae', 'r2', 'zero_classification_accuracy']
                for metric in key_metrics:
                    if metric in summary_stats:
                        stats = summary_stats[metric]
                        report.append(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f}")
        
        report_text = "\n".join(report)
        
        # Save if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def save_results(self, file_path: str, results: Optional[Dict[str, Any]] = None):
        """
        Save evaluation results to JSON file.
        
        Args:
            file_path: Path to save results
            results: Results to save (uses self.results if None)
        """
        if results is None:
            results = self.results
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif hasattr(obj, '__dict__'):
                return convert_for_json(obj.__dict__)
            else:
                return obj
        
        json_results = convert_for_json(results)
        
        with open(file_path, 'w') as f:
            json.dump(json_results, f, indent=2)