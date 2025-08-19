"""
Zero-Inflated Poisson (ZIP) Model for time series forecasting.

Mathematical formulation:
P(Y = y) = { π + (1-π)e^(-λ)           if y = 0
           { (1-π) * λ^y * e^(-λ) / y!  if y > 0

where π is the zero-inflation parameter and λ is the Poisson rate parameter.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import loggamma
from typing import Optional, Dict, Any, Union, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
import warnings


class ZeroInflatedPoisson(BaseEstimator, RegressorMixin):
    """
    Zero-Inflated Poisson (ZIP) regression model.
    
    This model combines a logistic regression for modeling zero vs non-zero
    outcomes with a Poisson regression for positive counts.
    
    Parameters:
    -----------
    pi_formula : str, default='intercept'
        Formula for zero-inflation probability π
        - 'intercept': constant π across all observations
        - 'linear': linear function of covariates
        - 'logistic': logistic function of covariates
    
    lambda_formula : str, default='log_linear'
        Formula for Poisson rate parameter λ
        - 'constant': constant λ
        - 'linear': linear function of covariates
        - 'log_linear': log-linear function (ensures λ > 0)
    
    fit_intercept : bool, default=True
        Whether to fit intercept terms
        
    max_iter : int, default=1000
        Maximum number of iterations for optimization
        
    tol : float, default=1e-6
        Tolerance for convergence
        
    solver : str, default='BFGS'
        Optimization solver ('BFGS', 'L-BFGS-B', 'Newton-CG')
    """
    
    def __init__(self, 
                 pi_formula: str = 'intercept',
                 lambda_formula: str = 'log_linear',
                 fit_intercept: bool = True,
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 solver: str = 'BFGS'):
        
        self.pi_formula = pi_formula
        self.lambda_formula = lambda_formula
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        
        # Model parameters (set after fitting)
        self.pi_params_ = None
        self.lambda_params_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ZeroInflatedPoisson':
        """
        Fit the Zero-Inflated Poisson model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (non-negative integers)
            
        Returns:
        --------
        self : object
            Returns the fitted model
        """
        # Input validation
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        
        # Check that y contains non-negative integers
        if not np.all((y >= 0) & (y == np.asarray(y, dtype=int))):
            raise ValueError("Target values must be non-negative integers")
        
        self.n_features_in_ = X.shape[1]
        self.n_samples_, _ = X.shape
        
        # Add intercept if requested
        if self.fit_intercept:
            X_pi = np.column_stack([np.ones(self.n_samples_), X])
            X_lambda = np.column_stack([np.ones(self.n_samples_), X])
        else:
            X_pi = X.copy()
            X_lambda = X.copy()
        
        # Initialize parameters
        initial_params = self._initialize_parameters(X_pi, X_lambda, y)
        
        # Fit model using maximum likelihood estimation
        try:
            result = optimize.minimize(
                fun=self._negative_log_likelihood,
                x0=initial_params,
                args=(X_pi, X_lambda, y),
                method=self.solver,
                options={'maxiter': self.max_iter, 'gtol': self.tol}
            )
            
            self.converged_ = result.success
            self.n_iter_ = result.nit
            
            # Extract parameters
            n_pi_params = X_pi.shape[1]
            self.pi_params_ = result.x[:n_pi_params]
            self.lambda_params_ = result.x[n_pi_params:]
            
            # Compute final log-likelihood
            self.log_likelihood_ = -result.fun
            
            if not self.converged_:
                warnings.warn("Model did not converge. Consider increasing max_iter or "
                            "changing the solver.", UserWarning)
                
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}")
        
        return self
    
    def _initialize_parameters(self, X_pi: np.ndarray, X_lambda: np.ndarray, 
                             y: np.ndarray) -> np.ndarray:
        """Initialize model parameters using simple heuristics."""
        
        # Initialize π parameters
        zero_ratio = np.mean(y == 0)
        zero_ratio = np.clip(zero_ratio, 0.01, 0.99)  # Avoid extreme values
        
        # Always initialize based on actual design matrix size
        pi_init = np.random.normal(0, 0.1, X_pi.shape[1])
        if self.fit_intercept:
            pi_init[0] = np.log(zero_ratio / (1 - zero_ratio))
        
        # Initialize λ parameters  
        nonzero_y = y[y > 0]
        if len(nonzero_y) > 0:
            mean_nonzero = np.mean(nonzero_y)
        else:
            mean_nonzero = 1.0
            
        if self.lambda_formula in ['log_linear', 'constant']:
            # Initialize with log of mean
            lambda_init = np.random.normal(0, 0.1, X_lambda.shape[1])
            if self.fit_intercept:
                lambda_init[0] = np.log(mean_nonzero)
        else:
            # Linear case
            lambda_init = np.random.normal(0, 0.1, X_lambda.shape[1])
            if self.fit_intercept:
                lambda_init[0] = mean_nonzero
        
        return np.concatenate([pi_init, lambda_init])
    
    def _negative_log_likelihood(self, params: np.ndarray, X_pi: np.ndarray,
                               X_lambda: np.ndarray, y: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization."""
        
        n_pi_params = X_pi.shape[1]
        pi_params = params[:n_pi_params]
        lambda_params = params[n_pi_params:]
        
        # Compute π (zero-inflation probability)
        if self.pi_formula == 'intercept':
            pi = np.full(len(y), 1 / (1 + np.exp(-pi_params[0])))
        elif self.pi_formula in ['linear', 'logistic']:
            linear_pi = X_pi @ pi_params
            pi = 1 / (1 + np.exp(-linear_pi))  # Logistic transformation
        else:
            raise ValueError(f"Unknown pi_formula: {self.pi_formula}")
        
        # Compute λ (Poisson rate parameter)
        if self.lambda_formula == 'constant':
            if self.fit_intercept:
                lambda_val = np.full(len(y), np.exp(lambda_params[0]))
            else:
                lambda_val = np.full(len(y), lambda_params[0])
        elif self.lambda_formula == 'log_linear':
            linear_lambda = X_lambda @ lambda_params
            lambda_val = np.exp(linear_lambda)
        elif self.lambda_formula == 'linear':
            lambda_val = X_lambda @ lambda_params
            lambda_val = np.maximum(lambda_val, 1e-8)  # Ensure positivity
        else:
            raise ValueError(f"Unknown lambda_formula: {self.lambda_formula}")
        
        # Ensure parameters are in valid range
        pi = np.clip(pi, 1e-8, 1 - 1e-8)
        lambda_val = np.maximum(lambda_val, 1e-8)
        
        # Compute log-likelihood
        log_likelihood = 0.0
        
        # For zero observations
        zero_mask = (y == 0)
        if np.any(zero_mask):
            log_like_zeros = np.log(pi[zero_mask] + (1 - pi[zero_mask]) * np.exp(-lambda_val[zero_mask]))
            log_likelihood += np.sum(log_like_zeros)
        
        # For non-zero observations
        nonzero_mask = (y > 0)
        if np.any(nonzero_mask):
            y_nonzero = y[nonzero_mask]
            pi_nonzero = pi[nonzero_mask]
            lambda_nonzero = lambda_val[nonzero_mask]
            
            # Poisson log-probability: y*log(λ) - λ - log(y!)
            log_poisson = (y_nonzero * np.log(lambda_nonzero) - 
                          lambda_nonzero - loggamma(y_nonzero + 1))
            
            log_like_nonzeros = (np.log(1 - pi_nonzero) + log_poisson)
            log_likelihood += np.sum(log_like_nonzeros)
        
        return -log_likelihood  # Return negative for minimization
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        proba_dict : dict
            Dictionary containing:
            - 'pi': Zero-inflation probabilities
            - 'lambda': Poisson rate parameters
            - 'zero_prob': Probability of zero outcome
        """
        self._check_fitted()
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        
        if self.fit_intercept:
            X_pi = np.column_stack([np.ones(X.shape[0]), X])
            X_lambda = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_pi = X.copy()
            X_lambda = X.copy()
        
        # Compute π
        if self.pi_formula == 'intercept':
            pi = np.full(X.shape[0], 1 / (1 + np.exp(-self.pi_params_[0])))
        else:
            linear_pi = X_pi @ self.pi_params_
            pi = 1 / (1 + np.exp(-linear_pi))
        
        # Compute λ
        if self.lambda_formula == 'constant':
            if self.fit_intercept:
                lambda_val = np.full(X.shape[0], np.exp(self.lambda_params_[0]))
            else:
                lambda_val = np.full(X.shape[0], self.lambda_params_[0])
        elif self.lambda_formula == 'log_linear':
            linear_lambda = X_lambda @ self.lambda_params_
            lambda_val = np.exp(linear_lambda)
        else:  # linear
            lambda_val = X_lambda @ self.lambda_params_
            lambda_val = np.maximum(lambda_val, 1e-8)
        
        # Compute probability of zero
        zero_prob = pi + (1 - pi) * np.exp(-lambda_val)
        
        return {
            'pi': pi,
            'lambda': lambda_val,
            'zero_prob': zero_prob
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the ZIP model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : ndarray, shape (n_samples,)
            Predicted values (expected values of ZIP distribution)
        """
        proba_dict = self.predict_proba(X)
        
        # Expected value of ZIP: E[Y] = (1 - π) * λ
        expected_values = (1 - proba_dict['pi']) * proba_dict['lambda']
        
        return expected_values
    
    def sample(self, X: np.ndarray, n_samples: int = 1, 
              random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample from the fitted ZIP distribution.
        
        Parameters:
        -----------
        X : array-like, shape (n_obs, n_features)
            Input features
        n_samples : int, default=1
            Number of samples to draw for each observation
        random_state : int, optional
            Random seed
            
        Returns:
        --------
        samples : ndarray, shape (n_obs, n_samples)
            Samples from ZIP distribution
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        proba_dict = self.predict_proba(X)
        n_obs = X.shape[0]
        
        samples = np.zeros((n_obs, n_samples))
        
        for i in range(n_obs):
            pi_i = proba_dict['pi'][i]
            lambda_i = proba_dict['lambda'][i]
            
            # Sample from mixture
            for j in range(n_samples):
                # First decide if zero from excess zeros
                if np.random.random() < pi_i:
                    samples[i, j] = 0
                else:
                    # Sample from Poisson
                    samples[i, j] = np.random.poisson(lambda_i)
        
        return samples.astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the log-likelihood score of the model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True values
            
        Returns:
        --------
        score : float
            Log-likelihood score
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        
        if self.fit_intercept:
            X_pi = np.column_stack([np.ones(X.shape[0]), X])
            X_lambda = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_pi = X.copy()
            X_lambda = X.copy()
        
        n_pi_params = X_pi.shape[1]
        params = np.concatenate([self.pi_params_, self.lambda_params_])
        
        return -self._negative_log_likelihood(params, X_pi, X_lambda, y)
    
    def _check_fitted(self) -> None:
        """Check if the model has been fitted."""
        if self.pi_params_ is None or self.lambda_params_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
    
    def get_params_summary(self) -> Dict[str, Any]:
        """Get summary of fitted parameters."""
        self._check_fitted()
        
        summary = {
            'pi_parameters': self.pi_params_.tolist(),
            'lambda_parameters': self.lambda_params_.tolist(),
            'log_likelihood': self.log_likelihood_,
            'converged': self.converged_,
            'n_iterations': self.n_iter_,
            'n_parameters': len(self.pi_params_) + len(self.lambda_params_)
        }
        
        # Compute AIC and BIC if possible
        n_params = summary['n_parameters']
        if hasattr(self, 'n_samples_'):
            summary['AIC'] = 2 * n_params - 2 * self.log_likelihood_
            summary['BIC'] = np.log(self.n_samples_) * n_params - 2 * self.log_likelihood_
        
        return summary
    
    def predict_quantiles(self, X: np.ndarray, quantiles: np.ndarray = None) -> np.ndarray:
        """
        Predict quantiles of the ZIP distribution.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        quantiles : array-like, default=[0.1, 0.5, 0.9]
            Quantiles to predict
            
        Returns:
        --------
        quantile_predictions : ndarray, shape (n_samples, n_quantiles)
            Predicted quantiles
        """
        if quantiles is None:
            quantiles = np.array([0.1, 0.5, 0.9])
        
        proba_dict = self.predict_proba(X)
        n_obs = X.shape[0]
        n_quantiles = len(quantiles)
        
        quantile_predictions = np.zeros((n_obs, n_quantiles))
        
        for i in range(n_obs):
            pi_i = proba_dict['pi'][i]
            lambda_i = proba_dict['lambda'][i]
            
            # Compute CDF values for a range of integers
            max_val = int(np.ceil(lambda_i + 5 * np.sqrt(lambda_i)) + 10)
            values = np.arange(0, max_val + 1)
            
            # Compute ZIP probabilities
            zero_prob = pi_i + (1 - pi_i) * np.exp(-lambda_i)
            
            probs = np.zeros(len(values))
            probs[0] = zero_prob
            
            if len(values) > 1:
                poisson_probs = stats.poisson.pmf(values[1:], lambda_i)
                probs[1:] = (1 - pi_i) * poisson_probs
            
            # Compute CDF
            cdf = np.cumsum(probs)
            
            # Find quantiles
            for j, q in enumerate(quantiles):
                quantile_predictions[i, j] = np.searchsorted(cdf, q, side='right')
        
        return quantile_predictions