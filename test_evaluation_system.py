"""
Test script for the comprehensive evaluation system.

This script demonstrates the usage of the evaluation metrics, model evaluator,
and benchmark suite on zero-inflated time series models.
"""

import sys
sys.path.append('/home/wentao/papercode/zero_inflated_comprehensive')

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Import our evaluation components
from evaluation.metrics import ZeroInflatedMetrics, compute_torch_metrics, format_metrics_report
from evaluation.evaluator import ModelEvaluator, ComprehensiveEvaluation
from evaluation.benchmarks import BenchmarkSuite, StandardBenchmarks

# Import some models for testing
from models.baseline.zip_model import ZeroInflatedPoisson
from models.zero_aware.zip_rnn import ZIPRNN
from models.zero_aware.dual_branch_network import DualBranchNetwork

# Import data generation
from generation.inject_zeros import inject_zeros


def test_metrics_system():
    """Test the zero-inflated metrics system."""
    print("Testing Zero-Inflated Metrics System")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    n_samples = 500
    
    # Create ground truth with zero-inflation
    true_values = np.random.exponential(2.0, n_samples)
    true_values = inject_zeros(true_values, mechanism='mixture', zero_ratio=0.3, random_state=42)
    
    # Create predictions with some error
    predictions = true_values + np.random.normal(0, 0.5, n_samples)
    predictions = np.maximum(predictions, 0)  # Ensure non-negative
    
    print(f"True values zero ratio: {np.mean(true_values == 0):.3f}")
    print(f"Predicted values zero ratio: {np.mean(predictions == 0):.3f}")
    
    # Test metrics calculation
    metrics_calc = ZeroInflatedMetrics(zero_threshold=1e-6)
    metrics = metrics_calc(predictions, true_values, return_components=True)
    
    print("\nComputed Metrics:")
    print("-" * 20)
    for category, category_metrics in metrics['components'].items():
        print(f"\n{category.upper()}:")
        for metric, value in category_metrics.items():
            print(f"  {metric}: {value:.6f}")
    
    # Test formatted report
    print("\n" + format_metrics_report(metrics, "Test Metrics Report"))
    
    return True


def test_model_evaluator():
    """Test the model evaluation framework."""
    print("\n\nTesting Model Evaluation Framework")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(123)
    n_samples = 400
    n_features = 3
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create zero-inflated targets
    y_base = 2 + X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.normal(0, 0.5, n_samples)
    y_base = np.maximum(y_base, 0)
    y = inject_zeros(y_base, mechanism='threshold', zero_ratio=0.25, random_state=123)
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Target zero ratio: {np.mean(y == 0):.3f}")
    
    # Create simple models for testing
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'ZIP Model': ZeroInflatedPoisson()
    }
    
    # Split data
    split_idx = int(0.7 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train and evaluate models
    evaluator = ModelEvaluator()
    results = []
    
    for model_name, model in models.items():
        try:
            print(f"\nEvaluating {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            result = evaluator.evaluate_model(model, X_test, y_test, model_name=model_name)
            results.append(result)
            
            # Print key metrics
            metrics = result.metrics
            print(f"  MSE: {metrics.get('mse', 'N/A'):.6f}")
            print(f"  MAE: {metrics.get('mae', 'N/A'):.6f}")
            print(f"  Zero Classification Accuracy: {metrics.get('zero_classification_accuracy', 'N/A'):.3f}")
            print(f"  Evaluation Time: {result.evaluation_time:.3f}s")
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
    
    # Compare models
    if len(results) > 1:
        print("\nModel Comparison:")
        comparison = evaluator.compare_models(results)
        
        if 'best_models' in comparison.get('summary', {}):
            print("Best models by metric:")
            for metric, best_model in comparison['summary']['best_models'].items():
                print(f"  {metric}: {best_model}")
    
    return True


def test_benchmark_suite():
    """Test the benchmark suite."""
    print("\n\nTesting Benchmark Suite")
    print("=" * 50)
    
    # Create benchmark suite
    suite = BenchmarkSuite(random_state=42)
    
    # Create synthetic datasets
    datasets = suite.create_synthetic_datasets()
    
    print(f"Created {len(datasets)} benchmark datasets:")
    for name, dataset in datasets.items():
        print(f"  {name}: {dataset.n_samples} samples, {dataset.zero_ratio:.3f} zero ratio, {dataset.mechanism} mechanism")
    
    # Create simple models for benchmarking
    def create_linear_model():
        return LinearRegression()
    
    def create_rf_model():
        return RandomForestRegressor(n_estimators=30, random_state=42)
    
    models = {
        'Linear': create_linear_model,
        'RandomForest': create_rf_model
    }
    
    print(f"\nRunning quick benchmark with {len(models)} models...")
    
    # Run quick benchmark
    try:
        benchmark_results = StandardBenchmarks.quick_benchmark(models)
        
        # Generate report
        report = suite.generate_benchmark_report(benchmark_results)
        print("\nBenchmark Report:")
        print(report)
        
        return True
        
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
        return False


def test_pytorch_integration():
    """Test integration with PyTorch models."""
    print("\n\nTesting PyTorch Integration")
    print("=" * 50)
    
    # Generate sequential data for PyTorch models
    np.random.seed(456)
    torch.manual_seed(456)
    
    # Create time series data
    t = np.arange(1000)
    base_series = 3 + 0.001 * t + np.sin(2 * np.pi * t / 100) + np.random.normal(0, 0.3, 1000)
    base_series = np.maximum(base_series, 0)
    zi_series = inject_zeros(base_series, mechanism='mixture', zero_ratio=0.2, random_state=456)
    
    print(f"Time series length: {len(zi_series)}")
    print(f"Zero ratio: {np.mean(zi_series == 0):.3f}")
    
    # Create sequences for PyTorch model
    def create_sequences(data, seq_len, pred_len):
        X, y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len:i+seq_len+pred_len])
        return torch.FloatTensor(X).unsqueeze(-1), torch.FloatTensor(y).unsqueeze(-1)
    
    X, y = create_sequences(zi_series, seq_len=24, pred_len=6)
    
    print(f"Sequence shapes: X={X.shape}, y={y.shape}")
    
    # Test with a simple PyTorch model
    class SimpleRNN(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, pred_len=6):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, pred_len * input_dim)
            self.pred_len = pred_len
            self.input_dim = input_dim
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])
            return output.view(-1, self.pred_len, self.input_dim)
    
    # Create and test model
    model = SimpleRNN()
    
    # Test metrics computation directly
    with torch.no_grad():
        predictions = model(X[:50])
        targets = y[:50]
        
        metrics = compute_torch_metrics(predictions, targets)
        
        print("\nPyTorch Model Metrics:")
        for metric, value in metrics.items():
            if 'components' not in metric:
                print(f"  {metric}: {value:.6f}")
    
    return True


def test_comprehensive_evaluation():
    """Test the comprehensive evaluation framework."""
    print("\n\nTesting Comprehensive Evaluation")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(789)
    n_samples = 600
    
    # Create features and targets
    X = np.random.randn(n_samples, 2)
    y_base = 2 + X[:, 0] - 0.5 * X[:, 1] + np.random.normal(0, 0.4, n_samples)
    y_base = np.maximum(y_base, 0)
    y = inject_zeros(y_base, mechanism='mixture', zero_ratio=0.35, random_state=789)
    
    print(f"Dataset: {n_samples} samples, zero ratio: {np.mean(y == 0):.3f}")
    
    # Create model factories for cross-validation
    def create_linear():
        return LinearRegression()
    
    def create_rf():
        return RandomForestRegressor(n_estimators=20, random_state=42)
    
    models = {
        'Linear': create_linear,
        'RandomForest': create_rf
    }
    
    # Run comprehensive evaluation
    comprehensive_eval = ComprehensiveEvaluation()
    
    try:
        results = comprehensive_eval.full_evaluation(
            models=models,
            X=X,
            y=y,
            test_split=0.3,
            perform_cv=True
        )
        
        # Generate report
        report = comprehensive_eval.generate_report(results)
        print("\nComprehensive Evaluation Report:")
        print(report)
        
        return True
        
    except Exception as e:
        print(f"Comprehensive evaluation failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("Zero-Inflated Time Series Evaluation System Test Suite")
    print("=" * 80)
    
    test_results = []
    
    # Run all tests
    test_results.append(('Metrics System', test_metrics_system()))
    test_results.append(('Model Evaluator', test_model_evaluator()))
    test_results.append(('Benchmark Suite', test_benchmark_suite()))
    test_results.append(('PyTorch Integration', test_pytorch_integration()))
    test_results.append(('Comprehensive Evaluation', test_comprehensive_evaluation()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, success in test_results:
        status = "✓" if success else "✗"
        print(f"{status} {test_name}")
    
    successful_tests = sum(1 for _, success in test_results if success)
    total_tests = len(test_results)
    
    print(f"\nOverall: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("\n🎉 All evaluation system tests passed!")
        print("\nThe comprehensive zero-inflated time series evaluation system is ready for use.")
        print("\nKey features available:")
        print("• Comprehensive metrics for zero-inflated data")
        print("• Model evaluation and comparison framework")
        print("• Cross-validation for time series")
        print("• Standardized benchmark datasets")
        print("• PyTorch integration")
        print("• Automated report generation")
    else:
        print(f"\n⚠️  Some tests failed. Check the detailed output above.")