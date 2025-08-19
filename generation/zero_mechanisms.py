"""
Mathematical zero-inflation mechanisms for time series data.

This module implements various theoretical approaches to generate zero-inflated
time series based on established statistical and probabilistic methods.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, Tuple
from scipy import stats
from scipy.special import gamma, digamma
import warnings

class ZeroInflationMechanism(ABC):
    """
    Abstract base class for zero-inflation mechanisms.
    
    All zero-inflation mechanisms should inherit from this class and implement
    the apply_zero_inflation method.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the zero-inflation mechanism.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def apply_zero_inflation(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply zero-inflation to the input data.
        
        Args:
            data: Input time series data
            **kwargs: Additional mechanism-specific parameters
            
        Returns:
            Zero-inflated time series
        """
        pass
    
    def get_zero_ratio(self, data: np.ndarray) -> float:
        """Calculate the zero ratio of the data."""
        return np.mean(data == 0)
    
    def validate_input(self, data: np.ndarray) -> None:
        """Validate input data."""
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        if data.ndim > 2:
            raise ValueError("Input data must be 1D or 2D")
        if np.any(np.isnan(data)):
            raise ValueError("Input data contains NaN values")


class ThresholdZeroInflation(ZeroInflationMechanism):
    """
    Threshold-based zero-inflation mechanism.
    
    Sets values below a certain threshold to zero. The threshold can be
    absolute, percentile-based, or dynamic.
    
    Mathematical basis:
    Y_zi = { 0           if Y <= threshold
           { Y           if Y > threshold
    """
    
    def __init__(self, threshold_type: str = 'percentile', **kwargs):
        """
        Initialize threshold-based zero inflation.
        
        Args:
            threshold_type: Type of threshold ('absolute', 'percentile', 'dynamic', 'adaptive')
            **kwargs: Additional parameters passed to parent
        """
        super().__init__(**kwargs)
        self.threshold_type = threshold_type
        
    def apply_zero_inflation(self, data: np.ndarray, 
                           threshold_value: Optional[float] = None,
                           percentile: float = 30.0,
                           window_size: int = 50,
                           adaptive_factor: float = 1.0) -> np.ndarray:
        """
        Apply threshold-based zero inflation.
        
        Args:
            data: Input time series data
            threshold_value: Absolute threshold value (for 'absolute' type)
            percentile: Percentile threshold (for 'percentile' type)
            window_size: Window size for dynamic threshold
            adaptive_factor: Adaptive factor for threshold adjustment
            
        Returns:
            Zero-inflated time series
        """
        self.validate_input(data)
        zero_inflated = data.copy()
        
        if self.threshold_type == 'absolute':
            if threshold_value is None:
                threshold_value = 0.0
            threshold = threshold_value
            
        elif self.threshold_type == 'percentile':
            threshold = np.percentile(data, percentile)
            
        elif self.threshold_type == 'dynamic':
            # Dynamic threshold based on moving statistics
            thresholds = self._compute_dynamic_threshold(data, window_size)
            for i in range(len(data)):
                if data[i] <= thresholds[i]:
                    zero_inflated[i] = 0
            return zero_inflated
            
        elif self.threshold_type == 'adaptive':
            # Adaptive threshold based on local statistics
            threshold = self._compute_adaptive_threshold(data, adaptive_factor)
            
        else:
            raise ValueError(f"Unknown threshold type: {self.threshold_type}")
        
        # Apply threshold
        zero_inflated[data <= threshold] = 0
        
        return zero_inflated
    
    def _compute_dynamic_threshold(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Compute dynamic threshold based on moving window statistics."""
        n = len(data)
        thresholds = np.zeros(n)
        
        for i in range(n):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(n, i + window_size // 2 + 1)
            window_data = data[start_idx:end_idx]
            
            # Use 25th percentile of local window as threshold
            thresholds[i] = np.percentile(window_data, 25)
        
        return thresholds
    
    def _compute_adaptive_threshold(self, data: np.ndarray, adaptive_factor: float) -> float:
        """Compute adaptive threshold based on data distribution."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Threshold = mean - adaptive_factor * std
        threshold = mean_val - adaptive_factor * std_val
        return max(0, threshold)  # Ensure non-negative


class MixtureZeroInflation(ZeroInflationMechanism):
    """
    Mixture model-based zero-inflation mechanism.
    
    Models the data as a mixture of a point mass at zero and a continuous distribution.
    
    Mathematical basis:
    Y_zi ~ (1-π)δ₀ + π * F(μ, σ)
    where δ₀ is point mass at zero, F is the continuous distribution
    """
    
    def __init__(self, distribution: str = 'normal', **kwargs):
        """
        Initialize mixture-based zero inflation.
        
        Args:
            distribution: Continuous distribution type ('normal', 'gamma', 'exponential', 'lognormal')
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.distribution = distribution
        
    def apply_zero_inflation(self, data: np.ndarray, 
                           zero_probability: float = 0.3,
                           preserve_distribution: bool = True,
                           fit_params: Optional[Dict] = None) -> np.ndarray:
        """
        Apply mixture-based zero inflation.
        
        Args:
            data: Input time series data
            zero_probability: Probability of zero (π parameter)
            preserve_distribution: Whether to preserve the original distribution shape
            fit_params: Pre-fitted distribution parameters
            
        Returns:
            Zero-inflated time series
        """
        self.validate_input(data)
        
        if not (0 <= zero_probability <= 1):
            raise ValueError("zero_probability must be between 0 and 1")
        
        n = len(data)
        zero_inflated = data.copy()
        
        # Generate binary mask for zeros
        zero_mask = np.random.binomial(1, zero_probability, n).astype(bool)
        
        if preserve_distribution:
            # Fit distribution to non-zero values and resample
            non_zero_data = data[data > 0]
            if len(non_zero_data) > 0:
                
                if fit_params is None:
                    fit_params = self._fit_distribution(non_zero_data)
                
                # Generate new values from fitted distribution
                new_values = self._sample_from_distribution(n, fit_params)
                
                # Apply mixture: zeros where mask is True, new values otherwise  
                zero_inflated = np.where(zero_mask, 0, new_values)
            else:
                # If no non-zero values, just apply zero mask
                zero_inflated[zero_mask] = 0
        else:
            # Simple masking approach
            zero_inflated[zero_mask] = 0
            
        return zero_inflated
    
    def _fit_distribution(self, data: np.ndarray) -> Dict[str, float]:
        """Fit specified distribution to the data."""
        if self.distribution == 'normal':
            mu, sigma = stats.norm.fit(data)
            return {'mu': mu, 'sigma': sigma}
            
        elif self.distribution == 'gamma':
            # Fit gamma distribution using method of moments
            alpha, loc, beta = stats.gamma.fit(data)
            return {'alpha': alpha, 'beta': beta, 'loc': loc}
            
        elif self.distribution == 'exponential':
            loc, scale = stats.expon.fit(data)
            return {'loc': loc, 'scale': scale}
            
        elif self.distribution == 'lognormal':
            sigma, loc, scale = stats.lognorm.fit(data)
            return {'sigma': sigma, 'loc': loc, 'scale': scale}
            
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
    
    def _sample_from_distribution(self, n: int, params: Dict[str, float]) -> np.ndarray:
        """Sample from the fitted distribution."""
        if self.distribution == 'normal':
            return stats.norm.rvs(loc=params['mu'], scale=params['sigma'], size=n)
            
        elif self.distribution == 'gamma':
            return stats.gamma.rvs(a=params['alpha'], scale=params['beta'], 
                                 loc=params['loc'], size=n)
            
        elif self.distribution == 'exponential':
            return stats.expon.rvs(loc=params['loc'], scale=params['scale'], size=n)
            
        elif self.distribution == 'lognormal':
            return stats.lognorm.rvs(s=params['sigma'], loc=params['loc'], 
                                   scale=params['scale'], size=n)


class TweedieZeroInflation(ZeroInflationMechanism):
    """
    Tweedie distribution-based zero-inflation mechanism.
    
    The Tweedie distribution naturally handles zero-inflation for 1 < power < 2,
    creating a compound Poisson-Gamma distribution with point mass at zero.
    
    Mathematical basis:
    Y ~ Tweedie(μ, φ, p) where 1 < p < 2
    
    The distribution has the form:
    f(y; μ, φ, p) = a(y, φ, p) exp((yθ - κ(θ))/φ)
    """
    
    def __init__(self, power: float = 1.5, **kwargs):
        """
        Initialize Tweedie-based zero inflation.
        
        Args:
            power: Tweedie power parameter (1 < power < 2)
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        if not (1 < power < 2):
            raise ValueError("Tweedie power parameter must be between 1 and 2 for zero-inflation")
        
        self.power = power
        
    def apply_zero_inflation(self, data: np.ndarray,
                           mu: Optional[float] = None,
                           phi: float = 1.0,
                           target_zero_ratio: Optional[float] = None) -> np.ndarray:
        """
        Apply Tweedie-based zero inflation.
        
        Args:
            data: Input time series data
            mu: Mean parameter (if None, estimated from data)
            phi: Dispersion parameter
            target_zero_ratio: Target proportion of zeros
            
        Returns:
            Zero-inflated time series following Tweedie distribution
        """
        self.validate_input(data)
        
        # Estimate parameters if not provided
        if mu is None:
            mu = np.mean(data)
        
        # Generate Tweedie random variables
        # For computational efficiency, we use the compound Poisson representation
        n = len(data)
        zero_inflated = np.zeros(n)
        
        # Lambda parameter for Poisson component
        lambda_param = mu ** (2 - self.power) / (phi * (2 - self.power))
        
        # Alpha and beta for Gamma component
        alpha = (2 - self.power) / (self.power - 1)
        beta = phi * (self.power - 1) * mu ** (self.power - 1)
        
        for i in range(n):
            # Generate number of jumps from Poisson
            N = np.random.poisson(lambda_param)
            
            if N == 0:
                zero_inflated[i] = 0
            else:
                # Sum of N gamma random variables
                gamma_sum = np.sum(np.random.gamma(alpha, scale=beta, size=N))
                zero_inflated[i] = gamma_sum
        
        # Adjust to match target zero ratio if specified
        if target_zero_ratio is not None:
            zero_inflated = self._adjust_zero_ratio(zero_inflated, target_zero_ratio)
        
        return zero_inflated
    
    def _adjust_zero_ratio(self, data: np.ndarray, target_ratio: float) -> np.ndarray:
        """Adjust the zero ratio to match target."""
        current_ratio = self.get_zero_ratio(data)
        
        if current_ratio < target_ratio:
            # Need more zeros
            non_zero_indices = np.where(data > 0)[0]
            n_to_zero = int((target_ratio - current_ratio) * len(data))
            n_to_zero = min(n_to_zero, len(non_zero_indices))
            
            if n_to_zero > 0:
                zero_indices = np.random.choice(non_zero_indices, n_to_zero, replace=False)
                data[zero_indices] = 0
                
        elif current_ratio > target_ratio:
            # Need fewer zeros - sample from fitted distribution
            zero_indices = np.where(data == 0)[0]
            n_to_unzero = int((current_ratio - target_ratio) * len(data))
            n_to_unzero = min(n_to_unzero, len(zero_indices))
            
            if n_to_unzero > 0:
                # Sample replacement values from non-zero data
                non_zero_data = data[data > 0]
                if len(non_zero_data) > 0:
                    replacement_values = np.random.choice(non_zero_data, n_to_unzero)
                    unzero_indices = np.random.choice(zero_indices, n_to_unzero, replace=False)
                    data[unzero_indices] = replacement_values
        
        return data


class HurdleZeroInflation(ZeroInflationMechanism):
    """
    Hurdle model-based zero-inflation mechanism.
    
    Separates the zero and non-zero processes using a hurdle approach:
    1. Binary model for zero vs non-zero
    2. Truncated distribution for positive values
    
    Mathematical basis:
    P(Y = 0) = f₁(0)
    P(Y = y | Y > 0) = f₂(y) / (1 - F₂(0)) for y > 0
    """
    
    def __init__(self, binary_model: str = 'logistic', 
                 positive_distribution: str = 'gamma', **kwargs):
        """
        Initialize hurdle-based zero inflation.
        
        Args:
            binary_model: Model for zero vs non-zero ('logistic', 'probit')
            positive_distribution: Distribution for positive values ('gamma', 'lognormal', 'exponential')
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.binary_model = binary_model
        self.positive_distribution = positive_distribution
        
    def apply_zero_inflation(self, data: np.ndarray,
                           covariates: Optional[np.ndarray] = None,
                           zero_probability: float = 0.3) -> np.ndarray:
        """
        Apply hurdle-based zero inflation.
        
        Args:
            data: Input time series data
            covariates: Covariate matrix for modeling zero probability
            zero_probability: Base probability of zero (if no covariates)
            
        Returns:
            Zero-inflated time series following hurdle model
        """
        self.validate_input(data)
        
        n = len(data)
        zero_inflated = np.zeros(n)
        
        if covariates is not None:
            # Use covariates to model zero probability
            zero_probs = self._compute_zero_probabilities(covariates)
        else:
            # Constant zero probability
            zero_probs = np.full(n, zero_probability)
        
        # Generate binary outcomes (zero vs non-zero)
        zero_indicators = np.random.binomial(1, zero_probs, n)
        
        # For non-zero outcomes, sample from truncated positive distribution
        non_zero_indices = np.where(zero_indicators == 0)[0]
        
        if len(non_zero_indices) > 0:
            # Fit distribution to original positive data
            positive_data = data[data > 0]
            if len(positive_data) > 0:
                positive_params = self._fit_positive_distribution(positive_data)
                positive_values = self._sample_positive_distribution(len(non_zero_indices), 
                                                                   positive_params)
                zero_inflated[non_zero_indices] = positive_values
        
        return zero_inflated
    
    def _compute_zero_probabilities(self, covariates: np.ndarray) -> np.ndarray:
        """Compute zero probabilities using binary model."""
        n, p = covariates.shape
        
        # Simple logistic model with random coefficients for demonstration
        # In practice, these would be estimated from data
        coefficients = np.random.normal(0, 0.1, p + 1)  # Including intercept
        
        # Add intercept column
        X = np.column_stack([np.ones(n), covariates])
        
        if self.binary_model == 'logistic':
            linear_pred = X @ coefficients
            zero_probs = 1 / (1 + np.exp(-linear_pred))
            
        elif self.binary_model == 'probit':
            linear_pred = X @ coefficients
            zero_probs = stats.norm.cdf(linear_pred)
            
        else:
            raise ValueError(f"Unknown binary model: {self.binary_model}")
        
        return np.clip(zero_probs, 1e-6, 1 - 1e-6)
    
    def _fit_positive_distribution(self, data: np.ndarray) -> Dict[str, float]:
        """Fit distribution to positive data."""
        if self.positive_distribution == 'gamma':
            alpha, loc, beta = stats.gamma.fit(data)
            return {'alpha': alpha, 'beta': beta, 'loc': loc}
            
        elif self.positive_distribution == 'lognormal':
            sigma, loc, scale = stats.lognorm.fit(data)
            return {'sigma': sigma, 'loc': loc, 'scale': scale}
            
        elif self.positive_distribution == 'exponential':
            loc, scale = stats.expon.fit(data)
            return {'loc': loc, 'scale': scale}
            
        else:
            raise ValueError(f"Unknown positive distribution: {self.positive_distribution}")
    
    def _sample_positive_distribution(self, n: int, params: Dict[str, float]) -> np.ndarray:
        """Sample from fitted positive distribution."""
        if self.positive_distribution == 'gamma':
            samples = stats.gamma.rvs(a=params['alpha'], scale=params['beta'], 
                                    loc=params['loc'], size=n)
            
        elif self.positive_distribution == 'lognormal':
            samples = stats.lognorm.rvs(s=params['sigma'], loc=params['loc'], 
                                      scale=params['scale'], size=n)
            
        elif self.positive_distribution == 'exponential':
            samples = stats.expon.rvs(loc=params['loc'], scale=params['scale'], size=n)
            
        else:
            raise ValueError(f"Unknown positive distribution: {self.positive_distribution}")
        
        # Ensure positive values (truncation)
        return np.maximum(samples, 1e-8)


def estimate_zero_inflation_parameters(data: np.ndarray, 
                                     mechanism: str = 'mixture') -> Dict[str, Any]:
    """
    Estimate parameters for zero-inflation mechanisms from observed data.
    
    Args:
        data: Observed zero-inflated time series
        mechanism: Type of mechanism ('threshold', 'mixture', 'tweedie', 'hurdle')
        
    Returns:
        Dictionary of estimated parameters
    """
    zero_ratio = np.mean(data == 0)
    non_zero_data = data[data > 0]
    
    if mechanism == 'threshold':
        # Estimate threshold that would produce similar zero ratio
        if zero_ratio > 0:
            threshold = np.percentile(data, zero_ratio * 100)
        else:
            threshold = np.min(data)
        return {'threshold_value': threshold, 'threshold_type': 'absolute'}
        
    elif mechanism == 'mixture':
        # Estimate mixture parameters
        if len(non_zero_data) > 0:
            mu = np.mean(non_zero_data)
            sigma = np.std(non_zero_data)
        else:
            mu, sigma = 0, 1
        return {'zero_probability': zero_ratio, 'mu': mu, 'sigma': sigma}
        
    elif mechanism == 'tweedie':
        # Estimate Tweedie parameters using method of moments
        if len(non_zero_data) > 0:
            mu = np.mean(data)  # Include zeros
            var = np.var(data)
            
            # Estimate power parameter using variance-mean relationship
            # Var = φ * μᵖ for Tweedie
            if mu > 0 and var > mu:
                power = np.log(var / mu) / np.log(mu) + 1
                power = np.clip(power, 1.1, 1.9)  # Ensure valid range
            else:
                power = 1.5
                
            phi = var / (mu ** power) if mu > 0 else 1.0
        else:
            mu, phi, power = 0, 1, 1.5
            
        return {'mu': mu, 'phi': phi, 'power': power}
        
    elif mechanism == 'hurdle':
        # Simple hurdle parameter estimation
        if len(non_zero_data) > 0:
            # Fit gamma to positive values
            alpha, loc, beta = stats.gamma.fit(non_zero_data)
        else:
            alpha, loc, beta = 1, 0, 1
        return {
            'zero_probability': zero_ratio,
            'alpha': alpha, 
            'beta': beta, 
            'loc': loc
        }
        
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")