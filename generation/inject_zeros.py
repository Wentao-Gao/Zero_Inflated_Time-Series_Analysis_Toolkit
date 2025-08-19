"""
Convenient functions for injecting zeros into time series data.

This module provides high-level functions to apply various zero-inflation
mechanisms to time series data with minimal configuration.
"""

import numpy as np
from typing import Union, Optional, Dict, Any
from .zero_mechanisms import (
    ThresholdZeroInflation,
    MixtureZeroInflation,
    TweedieZeroInflation,
    HurdleZeroInflation,
    estimate_zero_inflation_parameters
)


def inject_zeros(data: np.ndarray,
                mechanism: str = 'threshold',
                zero_ratio: float = 0.3,
                random_state: Optional[int] = None,
                **kwargs) -> np.ndarray:
    """
    Inject zeros into time series data using specified mechanism.
    
    Args:
        data: Input time series data (1D or 2D array)
        mechanism: Zero-inflation mechanism ('threshold', 'mixture', 'tweedie', 'hurdle')  
        zero_ratio: Target proportion of zeros (0.0 to 1.0)
        random_state: Random seed for reproducibility
        **kwargs: Additional mechanism-specific parameters
        
    Returns:
        Zero-inflated time series
        
    Examples:
        >>> data = np.random.gamma(2, 1, 1000)  # Generate positive data
        >>> zi_data = inject_zeros(data, mechanism='threshold', zero_ratio=0.4)
        >>> zi_data = inject_zeros(data, mechanism='mixture', zero_ratio=0.3, distribution='gamma')
        >>> zi_data = inject_zeros(data, mechanism='tweedie', zero_ratio=0.25, power=1.6)
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Handle multidimensional data
    if data.ndim == 1:
        return _inject_zeros_1d(data, mechanism, zero_ratio, random_state, **kwargs)
    elif data.ndim == 2:
        # Apply to each column independently
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            result[:, i] = _inject_zeros_1d(data[:, i], mechanism, zero_ratio, 
                                          random_state, **kwargs)
        return result
    else:
        raise ValueError("Input data must be 1D or 2D array")


def _inject_zeros_1d(data: np.ndarray,
                     mechanism: str,
                     zero_ratio: float,
                     random_state: Optional[int] = None,
                     **kwargs) -> np.ndarray:
    """Apply zero-inflation to 1D array."""
    
    if mechanism == 'threshold':
        return inject_zeros_threshold(data, zero_ratio=zero_ratio, 
                                    random_state=random_state, **kwargs)
    elif mechanism == 'mixture':
        return inject_zeros_mixture(data, zero_probability=zero_ratio,
                                  random_state=random_state, **kwargs)
    elif mechanism == 'tweedie':
        return inject_zeros_tweedie(data, target_zero_ratio=zero_ratio,
                                  random_state=random_state, **kwargs)
    elif mechanism == 'hurdle':
        return inject_zeros_hurdle(data, zero_probability=zero_ratio,
                                 random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


def inject_zeros_threshold(data: np.ndarray,
                          zero_ratio: float = 0.3,
                          threshold_type: str = 'percentile',
                          random_state: Optional[int] = None,
                          **kwargs) -> np.ndarray:
    """
    Inject zeros using threshold-based method.
    
    Args:
        data: Input time series data
        zero_ratio: Target proportion of zeros
        threshold_type: Type of threshold ('percentile', 'absolute', 'dynamic', 'adaptive')
        random_state: Random seed
        **kwargs: Additional parameters for threshold calculation
        
    Returns:
        Zero-inflated time series
    """
    # Convert zero_ratio to percentile for threshold method
    percentile = zero_ratio * 100 if threshold_type == 'percentile' else kwargs.get('percentile', 30.0)
    
    inflator = ThresholdZeroInflation(threshold_type=threshold_type, 
                                    random_state=random_state)
    
    return inflator.apply_zero_inflation(data, percentile=percentile, **kwargs)


def inject_zeros_mixture(data: np.ndarray,
                        zero_probability: float = 0.3,
                        distribution: str = 'gamma',
                        preserve_distribution: bool = True,
                        random_state: Optional[int] = None,
                        **kwargs) -> np.ndarray:
    """
    Inject zeros using mixture model method.
    
    Args:
        data: Input time series data
        zero_probability: Probability of zero in mixture
        distribution: Distribution for non-zero component
        preserve_distribution: Whether to preserve original distribution shape
        random_state: Random seed
        **kwargs: Additional distribution parameters
        
    Returns:
        Zero-inflated time series
    """
    inflator = MixtureZeroInflation(distribution=distribution, 
                                  random_state=random_state)
    
    return inflator.apply_zero_inflation(data, 
                                       zero_probability=zero_probability,
                                       preserve_distribution=preserve_distribution,
                                       **kwargs)


def inject_zeros_tweedie(data: np.ndarray,
                        power: float = 1.5,
                        target_zero_ratio: Optional[float] = None,
                        random_state: Optional[int] = None,
                        **kwargs) -> np.ndarray:
    """
    Inject zeros using Tweedie distribution method.
    
    Args:
        data: Input time series data
        power: Tweedie power parameter (1 < power < 2)
        target_zero_ratio: Target proportion of zeros
        random_state: Random seed
        **kwargs: Additional Tweedie parameters (mu, phi)
        
    Returns:
        Zero-inflated time series following Tweedie distribution
    """
    inflator = TweedieZeroInflation(power=power, random_state=random_state)
    
    return inflator.apply_zero_inflation(data, 
                                       target_zero_ratio=target_zero_ratio,
                                       **kwargs)


def inject_zeros_hurdle(data: np.ndarray,
                       zero_probability: float = 0.3,
                       binary_model: str = 'logistic',
                       positive_distribution: str = 'gamma',
                       random_state: Optional[int] = None,
                       **kwargs) -> np.ndarray:
    """
    Inject zeros using hurdle model method.
    
    Args:
        data: Input time series data
        zero_probability: Probability of zero in binary model
        binary_model: Model for zero vs non-zero
        positive_distribution: Distribution for positive values
        random_state: Random seed
        **kwargs: Additional model parameters
        
    Returns:
        Zero-inflated time series following hurdle model
    """
    inflator = HurdleZeroInflation(binary_model=binary_model,
                                 positive_distribution=positive_distribution,
                                 random_state=random_state)
    
    return inflator.apply_zero_inflation(data, 
                                       zero_probability=zero_probability,
                                       **kwargs)


def auto_inject_zeros(data: np.ndarray,
                     target_zero_ratio: float = 0.3,
                     method_selection: str = 'auto',
                     random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Automatically select and apply the best zero-inflation method.
    
    Args:
        data: Input time series data
        target_zero_ratio: Target proportion of zeros
        method_selection: Selection strategy ('auto', 'cross_validation', 'information_criteria')
        random_state: Random seed
        
    Returns:
        Dictionary containing zero-inflated data and method information
    """
    methods = ['threshold', 'mixture', 'tweedie', 'hurdle']
    results = {}
    
    if method_selection == 'auto':
        # Simple heuristic-based selection
        data_stats = _compute_data_statistics(data)
        selected_method = _select_method_heuristic(data_stats, target_zero_ratio)
        
    elif method_selection == 'cross_validation':
        # Cross-validation based selection (simplified implementation)
        selected_method = _select_method_cv(data, target_zero_ratio, methods, random_state)
        
    elif method_selection == 'information_criteria':
        # Information criteria based selection
        selected_method = _select_method_ic(data, target_zero_ratio, methods, random_state)
        
    else:
        raise ValueError(f"Unknown method selection: {method_selection}")
    
    # Apply selected method
    zero_inflated_data = inject_zeros(data, mechanism=selected_method, 
                                    zero_ratio=target_zero_ratio,
                                    random_state=random_state)
    
    return {
        'data': zero_inflated_data,
        'method': selected_method,
        'target_zero_ratio': target_zero_ratio,
        'actual_zero_ratio': np.mean(zero_inflated_data == 0),
        'parameters': estimate_zero_inflation_parameters(zero_inflated_data, selected_method)
    }


def _compute_data_statistics(data: np.ndarray) -> Dict[str, float]:
    """Compute relevant statistics for method selection."""
    non_zero_data = data[data > 0]
    
    return {
        'mean': np.mean(data),
        'var': np.var(data),
        'skewness': float(stats.skew(data)) if len(data) > 2 else 0,
        'kurtosis': float(stats.kurtosis(data)) if len(data) > 3 else 0,
        'zero_ratio': np.mean(data == 0),
        'min_val': np.min(data),
        'max_val': np.max(data),
        'cv': np.std(data) / np.mean(data) if np.mean(data) > 0 else 0,
        'range_ratio': (np.max(data) - np.min(data)) / (np.mean(data) + 1e-8)
    }


def _select_method_heuristic(stats: Dict[str, float], target_zero_ratio: float) -> str:
    """Select method based on data characteristics heuristics."""
    
    # If data is already zero-inflated, prefer mixture or hurdle
    if stats['zero_ratio'] > 0.1:
        if stats['cv'] > 2:  # High variability suggests hurdle model
            return 'hurdle'
        else:
            return 'mixture'
    
    # If target zero ratio is high, prefer Tweedie or threshold  
    if target_zero_ratio > 0.5:
        if stats['skewness'] > 1:  # Right-skewed data fits Tweedie well
            return 'tweedie'
        else:
            return 'threshold'
    
    # For moderate zero ratios, prefer mixture for distributional flexibility
    if 0.2 <= target_zero_ratio <= 0.5:
        return 'mixture'
    
    # For low zero ratios, threshold is often sufficient
    return 'threshold'


def _select_method_cv(data: np.ndarray, target_zero_ratio: float, 
                     methods: list, random_state: Optional[int]) -> str:
    """Select method using cross-validation (simplified implementation)."""
    from sklearn.metrics import mean_squared_error
    
    n_folds = 5
    fold_size = len(data) // n_folds
    method_scores = {method: [] for method in methods}
    
    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(data)
        
        test_data = data[start_idx:end_idx]
        train_data = np.concatenate([data[:start_idx], data[end_idx:]])
        
        for method in methods:
            try:
                # Generate zero-inflated version
                zi_train = inject_zeros(train_data, mechanism=method, 
                                      zero_ratio=target_zero_ratio, 
                                      random_state=random_state)
                
                # Simple reconstruction score (how well it preserves distribution)
                zi_test = inject_zeros(test_data, mechanism=method,
                                     zero_ratio=target_zero_ratio,
                                     random_state=random_state)
                
                # Score based on distributional similarity
                score = _compute_distribution_similarity(test_data, zi_test)
                method_scores[method].append(score)
                
            except Exception:
                # If method fails, assign worst score
                method_scores[method].append(float('inf'))
    
    # Select method with best average score
    avg_scores = {method: np.mean(scores) for method, scores in method_scores.items()}
    return min(avg_scores, key=avg_scores.get)


def _select_method_ic(data: np.ndarray, target_zero_ratio: float,
                     methods: list, random_state: Optional[int]) -> str:
    """Select method using information criteria."""
    # Simplified implementation - would need proper likelihood calculations
    method_scores = {}
    
    for method in methods:
        try:
            zi_data = inject_zeros(data, mechanism=method, 
                                 zero_ratio=target_zero_ratio,
                                 random_state=random_state)
            
            # Approximate AIC based on distribution fit
            score = _compute_approximate_aic(data, zi_data, method)
            method_scores[method] = score
            
        except Exception:
            method_scores[method] = float('inf')
    
    return min(method_scores, key=method_scores.get)


def _compute_distribution_similarity(original: np.ndarray, 
                                   zero_inflated: np.ndarray) -> float:
    """Compute similarity between original and zero-inflated distributions."""
    from scipy.stats import ks_2samp
    
    # Kolmogorov-Smirnov test statistic as dissimilarity measure
    ks_stat, _ = ks_2samp(original, zero_inflated)
    return ks_stat


def _compute_approximate_aic(original: np.ndarray, zero_inflated: np.ndarray, 
                           method: str) -> float:
    """Compute approximate AIC for method selection."""
    # Simplified AIC approximation
    n = len(zero_inflated)
    
    # Number of parameters (rough estimates)
    k_params = {'threshold': 1, 'mixture': 3, 'tweedie': 3, 'hurdle': 4}
    k = k_params.get(method, 2)
    
    # Approximate log-likelihood based on distribution similarity
    similarity = _compute_distribution_similarity(original, zero_inflated)
    log_likelihood = -n * similarity  # Negative because similarity is distance-like
    
    # AIC = 2k - 2ln(L)
    aic = 2 * k - 2 * log_likelihood
    
    return aic


def compare_zero_inflation_methods(data: np.ndarray,
                                 target_zero_ratio: float = 0.3,
                                 methods: Optional[list] = None,
                                 random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Compare different zero-inflation methods on the same data.
    
    Args:
        data: Input time series data
        target_zero_ratio: Target proportion of zeros
        methods: List of methods to compare (if None, uses all available)
        random_state: Random seed
        
    Returns:
        Dictionary with comparison results for each method
    """
    if methods is None:
        methods = ['threshold', 'mixture', 'tweedie', 'hurdle']
    
    results = {}
    
    for method in methods:
        try:
            zi_data = inject_zeros(data, mechanism=method, 
                                 zero_ratio=target_zero_ratio,
                                 random_state=random_state)
            
            actual_zero_ratio = np.mean(zi_data == 0)
            distribution_sim = _compute_distribution_similarity(data, zi_data)
            
            results[method] = {
                'zero_inflated_data': zi_data,
                'target_zero_ratio': target_zero_ratio,
                'actual_zero_ratio': actual_zero_ratio,
                'zero_ratio_error': abs(target_zero_ratio - actual_zero_ratio),
                'distribution_similarity': distribution_sim,
                'parameters': estimate_zero_inflation_parameters(zi_data, method)
            }
            
        except Exception as e:
            results[method] = {
                'error': str(e),
                'zero_inflated_data': None
            }
    
    return results


# Import scipy.stats for statistical functions
from scipy import stats