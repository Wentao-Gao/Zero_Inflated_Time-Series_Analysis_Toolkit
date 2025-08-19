"""
Tweedie Generalized Linear Model (GLM) for zero-inflated time series.

The Tweedie distribution is a member of the exponential dispersion family
that naturally handles zero-inflation when 1 < power < 2. It is a compound
Poisson-Gamma distribution with point mass at zero.

Mathematical formulation:
f(y; μ, φ, p) = a(y, φ, p) * exp((y*θ - κ(θ))/φ)

where:
- θ = μ^(1-p) / (1-p)  (canonical parameter)
- κ(θ) = μ^(2-p) / (2-p)  (cumulant function)  
- φ is the dispersion parameter
- p is the power parameter (1 < p < 2 for zero-inflation)
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import gamma
from typing import Optional, Dict, Any, Union, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
import warnings


class TweedieGLM(BaseEstimator, RegressorMixin):
    """
    Tweedie Generalized Linear Model for zero-inflated data.
    
    The Tweedie distribution with 1 < power < 2 is particularly suitable for
    zero-inflated count and continuous data, as it naturally models both
    the point mass at zero and the continuous positive distribution.
    
    Parameters:
    -----------
    power : float, default=1.5
        Tweedie power parameter. Must be in (1, 2) for zero-inflation.
        - p = 1.5: Compound Poisson-Gamma (typical for insurance claims)
        - p → 1: Approaches Poisson distribution  
        - p → 2: Approaches Gamma distribution
    
    link : str, default='log'
        Link function for the mean
        - 'log': log(μ) = Xβ (ensures μ > 0)
        - 'identity': μ = Xβ (with positivity constraints)
        - 'power': μ^(1-power) = Xβ (canonical link)
    
    fit_intercept : bool, default=True
        Whether to fit an intercept term
        
    alpha : float, default=0.0
        Regularization strength (L2 penalty)
        
    max_iter : int, default=100
        Maximum number of IRLS iterations
        
    tol : float, default=1e-6
        Tolerance for convergence
        
    fit_dispersion : bool, default=True
        Whether to estimate dispersion parameter φ or fix it to 1.0
    """
    
    def __init__(self, 
                 power: float = 1.5,
                 link: str = 'log',
                 fit_intercept: bool = True,
                 alpha: float = 0.0,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 fit_dispersion: bool = True):
        
        if not (1 < power < 2):
            raise ValueError("Power parameter must be in (1, 2) for zero-inflation")
        
        self.power = power
        self.link = link
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_dispersion = fit_dispersion
        
        # Model parameters (set after fitting)
        self.coef_ = None
        self.intercept_ = None
        self.dispersion_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.deviance_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TweedieGLM':
        """
        Fit the Tweedie GLM using Iteratively Reweighted Least Squares (IRLS).
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (non-negative)
            
        Returns:
        --------
        self : object
            Returns the fitted model
        """
        # Input validation
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        
        # Check that y contains non-negative values
        if np.any(y < 0):
            raise ValueError("Target values must be non-negative for Tweedie distribution")
        
        self.n_features_in_ = X.shape[1]
        n_samples, n_features = X.shape
        
        # Add intercept if requested
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n_samples), X])
        else:
            X_design = X.copy()
        
        # Initialize parameters
        # For Tweedie, we need good starting values
        mu_init = self._initialize_mu(y)
        eta_init = self._link_function(mu_init)
        
        # Use simple linear regression as starting point
        try:
            beta_init = np.linalg.lstsq(X_design, eta_init, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta_init = np.zeros(X_design.shape[1])
            if self.fit_intercept:
                beta_init[0] = np.mean(eta_init)
        
        # IRLS algorithm
        beta = beta_init.copy()
        
        for iteration in range(self.max_iter):
            beta_old = beta.copy()
            
            # Linear predictor
            eta = X_design @ beta
            
            # Mean response
            mu = self._inverse_link_function(eta)
            mu = np.maximum(mu, 1e-10)  # Ensure positivity
            
            # Variance function: V(μ) = μ^p
            variance = np.power(mu, self.power)
            
            # Working weights: w = 1 / (V(μ) * g'(μ)^2)
            d_mu_d_eta = self._derivative_inverse_link(eta)
            weights = d_mu_d_eta**2 / variance
            weights = np.maximum(weights, 1e-10)  # Avoid zero weights
            
            # Working response: z = η + (y - μ) / g'(μ)
            working_response = eta + (y - mu) / d_mu_d_eta
            
            # Weighted least squares with regularization
            try:
                W = np.diag(weights)
                XtWX = X_design.T @ W @ X_design
                
                # Add L2 regularization
                if self.alpha > 0:
                    XtWX += self.alpha * np.eye(X_design.shape[1])
                
                XtWz = X_design.T @ (weights * working_response)
                beta = np.linalg.solve(XtWX, XtWz)
                
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudo-inverse
                W_sqrt = np.diag(np.sqrt(weights))
                X_weighted = W_sqrt @ X_design
                z_weighted = np.sqrt(weights) * working_response
                
                beta = np.linalg.lstsq(X_weighted, z_weighted, rcond=None)[0]
            
            # Check for convergence
            if np.allclose(beta, beta_old, atol=self.tol, rtol=self.tol):
                self.converged_ = True
                break
        
        self.n_iter_ = iteration + 1
        
        if not self.converged_:
            warnings.warn(f"IRLS did not converge after {self.max_iter} iterations", 
                        UserWarning)
        
        # Store fitted parameters
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        
        # Estimate dispersion parameter
        if self.fit_dispersion:
            self.dispersion_ = self._estimate_dispersion(X_design, y, beta)
        else:
            self.dispersion_ = 1.0
        
        # Compute deviance
        self.deviance_ = self._compute_deviance(y, mu)
        
        return self
    
    def _initialize_mu(self, y: np.ndarray) -> np.ndarray:
        """Initialize mean values avoiding zeros."""
        # For Tweedie, we need to handle zeros carefully
        mu_init = y.copy()
        
        # Replace zeros with small positive values
        zero_mask = (y == 0)
        if np.any(zero_mask):
            nonzero_vals = y[y > 0]
            if len(nonzero_vals) > 0:
                min_positive = np.min(nonzero_vals)
                mu_init[zero_mask] = min_positive / 10
            else:
                mu_init[zero_mask] = 0.1
        
        return mu_init
    
    def _link_function(self, mu: np.ndarray) -> np.ndarray:
        """Apply link function."""
        mu = np.maximum(mu, 1e-10)
        
        if self.link == 'log':
            return np.log(mu)
        elif self.link == 'identity':
            return mu
        elif self.link == 'power':
            # Canonical link: μ^(1-p)
            if self.power == 1.0:
                return np.log(mu)
            else:
                return np.power(mu, 1 - self.power)
        else:
            raise ValueError(f"Unknown link function: {self.link}")
    
    def _inverse_link_function(self, eta: np.ndarray) -> np.ndarray:
        """Apply inverse link function."""
        if self.link == 'log':
            return np.exp(eta)
        elif self.link == 'identity':
            return np.maximum(eta, 1e-10)
        elif self.link == 'power':
            # Inverse of canonical link
            if self.power == 1.0:
                return np.exp(eta)
            else:
                return np.power(np.maximum(eta, 1e-10), 1 / (1 - self.power))
        else:
            raise ValueError(f"Unknown link function: {self.link}")
    
    def _derivative_inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """Compute derivative of inverse link function."""
        if self.link == 'log':
            return np.exp(eta)
        elif self.link == 'identity':
            return np.ones_like(eta)
        elif self.link == 'power':
            if self.power == 1.0:
                return np.exp(eta)
            else:
                mu = self._inverse_link_function(eta)
                return mu**(self.power) / (1 - self.power)
        else:
            raise ValueError(f"Unknown link function: {self.link}")
    
    def _estimate_dispersion(self, X: np.ndarray, y: np.ndarray, 
                           beta: np.ndarray) -> float:
        """Estimate dispersion parameter using method of moments."""
        eta = X @ beta
        mu = self._inverse_link_function(eta)
        
        # Method of moments estimator for Tweedie dispersion
        # φ̂ = sum((y - μ)² / μ^p) / (n - p)
        variance_weights = np.power(mu, self.power)
        variance_weights = np.maximum(variance_weights, 1e-10)
        
        residuals_squared = (y - mu)**2
        dispersion_numerator = np.sum(residuals_squared / variance_weights)
        
        # Degrees of freedom
        df = len(y) - len(beta)
        df = max(df, 1)  # Avoid division by zero
        
        dispersion = dispersion_numerator / df
        
        # Ensure reasonable bounds
        return max(dispersion, 1e-3)
    
    def _compute_deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Compute Tweedie deviance."""
        # Tweedie deviance: D = 2 * sum(d(y, μ))
        # where d(y, μ) is the unit deviance
        
        deviance = 0.0
        
        for i in range(len(y)):
            y_i, mu_i = y[i], mu[i]
            mu_i = max(mu_i, 1e-10)
            
            if y_i == 0:
                # Unit deviance when y = 0
                unit_dev = 2 * mu_i**(2 - self.power) / (2 - self.power)
            else:
                # Unit deviance when y > 0
                term1 = y_i**(2 - self.power) / (2 - self.power)
                term2 = y_i * mu_i**(1 - self.power) / (1 - self.power)
                term3 = mu_i**(2 - self.power) / (2 - self.power)
                
                unit_dev = 2 * (term1 - term2 + term3)
            
            deviance += unit_dev
        
        return deviance
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the Tweedie GLM.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : ndarray, shape (n_samples,)
            Predicted mean values
        """
        self._check_fitted()
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        
        # Linear predictor
        eta = X @ self.coef_ + self.intercept_
        
        # Mean prediction
        mu = self._inverse_link_function(eta)
        
        return mu
    
    def predict_interval(self, X: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict confidence intervals.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
        alpha : float, default=0.05
            Significance level (for 1-alpha confidence interval)
            
        Returns:
        --------
        lower_bound, upper_bound : tuple of ndarray
            Lower and upper bounds of prediction intervals
        """
        self._check_fitted()
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        
        # Point predictions
        mu = self.predict(X)
        
        # Approximate standard errors using delta method
        # This is a rough approximation - exact intervals would require
        # more sophisticated methods
        
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_design = X.copy()
        
        # Approximate variance of linear predictor
        # Var(η) ≈ X (X'WX)^(-1) X' * φ
        # This is a simplification - exact calculation would need the weight matrix from fitting
        
        try:
            # Rough approximation using identity weights
            XtX_inv = np.linalg.inv(X_design.T @ X_design + self.alpha * np.eye(X_design.shape[1]))
            eta_var = np.diag(X_design @ XtX_inv @ X_design.T) * self.dispersion_
        except:
            # If inversion fails, use constant variance
            eta_var = np.full(len(mu), self.dispersion_)
        
        # Convert to mean scale using delta method
        d_mu_d_eta = self._derivative_inverse_link(X @ self.coef_ + self.intercept_)
        mu_var = eta_var * d_mu_d_eta**2
        mu_se = np.sqrt(mu_var)
        
        # Use normal approximation (rough)
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        lower_bound = np.maximum(mu - z_score * mu_se, 0)
        upper_bound = mu + z_score * mu_se
        
        return lower_bound, upper_bound
    
    def sample(self, X: np.ndarray, n_samples: int = 1,
              random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample from the fitted Tweedie distribution.
        
        Note: Exact sampling from Tweedie distribution is complex.
        This uses an approximation via compound Poisson-Gamma representation.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        mu = self.predict(X)
        n_obs = len(mu)
        
        samples = np.zeros((n_obs, n_samples))
        
        for i in range(n_obs):
            mu_i = mu[i]
            
            # Compound Poisson-Gamma representation
            # λ = μ^(2-p) / (φ(2-p))
            # α = (2-p)/(p-1), β = φ(p-1)μ^(p-1)
            
            lambda_param = mu_i**(2 - self.power) / (self.dispersion_ * (2 - self.power))
            alpha = (2 - self.power) / (self.power - 1)
            beta = self.dispersion_ * (self.power - 1) * mu_i**(self.power - 1)
            
            for j in range(n_samples):
                # Number of jumps from Poisson
                N = np.random.poisson(max(lambda_param, 1e-10))
                
                if N == 0:
                    samples[i, j] = 0
                else:
                    # Sum of N gamma random variables
                    gamma_sum = np.sum(np.random.gamma(alpha, beta, size=N))
                    samples[i, j] = gamma_sum
        
        return samples
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the Tweedie deviance score (negative, so higher is better).
        """
        y_pred = self.predict(X)
        deviance = self._compute_deviance(y, y_pred)
        return -deviance / len(y)  # Negative mean deviance
    
    def _check_fitted(self) -> None:
        """Check if the model has been fitted."""
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
    
    def get_params_summary(self) -> Dict[str, Any]:
        """Get summary of fitted parameters."""
        self._check_fitted()
        
        summary = {
            'coefficients': self.coef_.tolist(),
            'intercept': float(self.intercept_),
            'power_parameter': self.power,
            'dispersion_parameter': self.dispersion_,
            'deviance': self.deviance_,
            'converged': self.converged_,
            'n_iterations': self.n_iter_,
            'link_function': self.link
        }
        
        return summary