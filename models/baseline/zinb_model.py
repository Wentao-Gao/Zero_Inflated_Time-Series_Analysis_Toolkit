"""
Zero-Inflated Negative Binomial (ZINB) Model for time series forecasting.

Mathematical formulation:
P(Y = y) = { π + (1-π) * (r/(r+μ))^r                           if y = 0
           { (1-π) * Γ(y+r)/(Γ(r)Γ(y+1)) * (r/(r+μ))^r * (μ/(r+μ))^y  if y > 0

where π is the zero-inflation parameter, μ is the mean parameter, 
and r is the dispersion parameter of the Negative Binomial distribution.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import loggamma, digamma, polygamma
from typing import Optional, Dict, Any, Union, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
import warnings


class ZeroInflatedNegativeBinomial(BaseEstimator, RegressorMixin):
    """
    Zero-Inflated Negative Binomial (ZINB) regression model.
    
    This model combines a logistic regression for modeling zero vs non-zero
    outcomes with a Negative Binomial regression for positive counts.
    The Negative Binomial distribution allows for overdispersion relative to Poisson.
    
    Parameters:
    -----------
    pi_formula : str, default='intercept'
        Formula for zero-inflation probability π
        - 'intercept': constant π across all observations
        - 'logistic': logistic function of covariates
    
    mu_formula : str, default='log_linear'  
        Formula for NB mean parameter μ
        - 'log_linear': log-linear function (ensures μ > 0)
        - 'linear': linear function with positivity constraint
    
    dispersion_formula : str, default='constant'
        Formula for dispersion parameter r
        - 'constant': constant r across observations
        - 'log_linear': log-linear function of covariates
    
    fit_intercept : bool, default=True
        Whether to fit intercept terms
        
    max_iter : int, default=1000
        Maximum number of iterations for optimization
        
    tol : float, default=1e-6
        Tolerance for convergence
        
    solver : str, default='L-BFGS-B'
        Optimization solver
    """
    
    def __init__(self,
                 pi_formula: str = 'intercept',
                 mu_formula: str = 'log_linear',
                 dispersion_formula: str = 'constant',
                 fit_intercept: bool = True,
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 solver: str = 'L-BFGS-B'):
        
        self.pi_formula = pi_formula
        self.mu_formula = mu_formula
        self.dispersion_formula = dispersion_formula
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        
        # Model parameters (set after fitting)
        self.pi_params_ = None
        self.mu_params_ = None
        self.r_params_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ZeroInflatedNegativeBinomial':
        """
        Fit the Zero-Inflated Negative Binomial model.
        
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
        
        # Prepare design matrices
        if self.fit_intercept:
            X_pi = np.column_stack([np.ones(self.n_samples_), X])
            X_mu = np.column_stack([np.ones(self.n_samples_), X])
            X_r = np.column_stack([np.ones(self.n_samples_), X])
        else:
            X_pi = X.copy()
            X_mu = X.copy() 
            X_r = X.copy()
        
        # Handle dispersion parameter design matrix
        if self.dispersion_formula == 'constant':
            X_r = np.ones((self.n_samples_, 1))
        
        # Initialize parameters
        initial_params = self._initialize_parameters(X_pi, X_mu, X_r, y)
        
        # Set bounds for parameters
        bounds = self._get_parameter_bounds(X_pi, X_mu, X_r)
        
        # Fit model using maximum likelihood estimation
        try:
            result = optimize.minimize(
                fun=self._negative_log_likelihood,
                x0=initial_params,
                args=(X_pi, X_mu, X_r, y),
                method=self.solver,
                bounds=bounds,
                options={'maxiter': self.max_iter, 'gtol': self.tol}
            )
            
            self.converged_ = result.success
            self.n_iter_ = result.nit
            
            # Extract parameters
            n_pi_params = X_pi.shape[1]
            n_mu_params = X_mu.shape[1]
            n_r_params = X_r.shape[1]
            
            self.pi_params_ = result.x[:n_pi_params]
            self.mu_params_ = result.x[n_pi_params:n_pi_params + n_mu_params]
            self.r_params_ = result.x[n_pi_params + n_mu_params:]
            
            # Compute final log-likelihood
            self.log_likelihood_ = -result.fun
            
            if not self.converged_:
                warnings.warn("Model did not converge. Consider increasing max_iter or "
                            "changing the solver.", UserWarning)
                            
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}")
        
        return self
    
    def _initialize_parameters(self, X_pi: np.ndarray, X_mu: np.ndarray,
                             X_r: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Initialize model parameters using method of moments and simple heuristics."""
        
        # Initialize π parameters
        zero_ratio = np.mean(y == 0)
        zero_ratio = np.clip(zero_ratio, 0.01, 0.99)  # Avoid extreme values
        
        # Always initialize based on actual design matrix size
        pi_init = np.random.normal(0, 0.1, X_pi.shape[1])
        if self.fit_intercept:
            pi_init[0] = np.log(zero_ratio / (1 - zero_ratio))
        
        # Initialize μ parameters
        nonzero_y = y[y > 0]
        if len(nonzero_y) > 0:
            mean_nonzero = np.mean(nonzero_y)
            var_nonzero = np.var(nonzero_y)
        else:
            mean_nonzero = 1.0
            var_nonzero = 2.0
        
        if self.mu_formula == 'log_linear':
            mu_init = np.random.normal(0, 0.1, X_mu.shape[1])
            if self.fit_intercept:
                mu_init[0] = np.log(max(mean_nonzero, 1e-6))
        else:  # linear
            mu_init = np.random.normal(0, 0.1, X_mu.shape[1])
            if self.fit_intercept:
                mu_init[0] = mean_nonzero
        
        # Initialize r (dispersion) parameters using method of moments
        # For NB: Var = μ + μ²/r, so r = μ²/(Var - μ)
        if len(nonzero_y) > 1 and var_nonzero > mean_nonzero:
            r_estimate = mean_nonzero**2 / (var_nonzero - mean_nonzero)
            r_estimate = max(r_estimate, 0.1)  # Ensure reasonable value
        else:
            r_estimate = 1.0
        
        if self.dispersion_formula == 'constant':
            r_init = np.array([np.log(r_estimate)])  # Log-parameterize for positivity
        else:
            r_init = np.random.normal(0, 0.1, X_r.shape[1])
            if self.fit_intercept:
                r_init[0] = np.log(r_estimate)
        
        return np.concatenate([pi_init, mu_init, r_init])
    
    def _get_parameter_bounds(self, X_pi: np.ndarray, X_mu: np.ndarray,
                            X_r: np.ndarray) -> list:
        """Get parameter bounds for optimization."""
        bounds = []
        
        # Bounds for π parameters (logit scale, so unbounded)
        for _ in range(X_pi.shape[1]):
            bounds.append((-10, 10))
        
        # Bounds for μ parameters  
        if self.mu_formula == 'log_linear':
            # Log scale, so unbounded but practical limits
            for _ in range(X_mu.shape[1]):
                bounds.append((-10, 10))
        else:
            # Linear scale, must be positive
            for _ in range(X_mu.shape[1]):
                bounds.append((1e-6, None))
        
        # Bounds for r parameters (log scale for positivity)
        for _ in range(X_r.shape[1]):
            bounds.append((-5, 5))  # exp(-5) = 0.007, exp(5) = 148
        
        return bounds
    
    def _negative_log_likelihood(self, params: np.ndarray, X_pi: np.ndarray,
                               X_mu: np.ndarray, X_r: np.ndarray, 
                               y: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization."""
        
        n_pi_params = X_pi.shape[1]
        n_mu_params = X_mu.shape[1] 
        n_r_params = X_r.shape[1]
        
        pi_params = params[:n_pi_params]
        mu_params = params[n_pi_params:n_pi_params + n_mu_params]
        r_params = params[n_pi_params + n_mu_params:]
        
        # Compute π (zero-inflation probability)
        if self.pi_formula == 'intercept':
            linear_pi = np.full(len(y), pi_params[0])
        else:
            linear_pi = X_pi @ pi_params
        pi = 1 / (1 + np.exp(-linear_pi))
        pi = np.clip(pi, 1e-8, 1 - 1e-8)
        
        # Compute μ (mean parameter)
        if self.mu_formula == 'log_linear':
            linear_mu = X_mu @ mu_params
            mu = np.exp(linear_mu)
        else:
            mu = X_mu @ mu_params
        mu = np.maximum(mu, 1e-8)  # Ensure positivity
        
        # Compute r (dispersion parameter)
        if self.dispersion_formula == 'constant':
            r = np.full(len(y), np.exp(r_params[0]))
        else:
            linear_r = X_r @ r_params
            r = np.exp(linear_r)
        r = np.maximum(r, 1e-8)  # Ensure positivity
        
        # Compute log-likelihood
        log_likelihood = 0.0
        
        try:
            # For zero observations
            zero_mask = (y == 0)
            if np.any(zero_mask):
                # P(Y=0) = π + (1-π) * (r/(r+μ))^r
                r_zero = r[zero_mask]
                mu_zero = mu[zero_mask]
                pi_zero = pi[zero_mask]
                
                nb_zero_prob = (r_zero / (r_zero + mu_zero)) ** r_zero
                total_zero_prob = pi_zero + (1 - pi_zero) * nb_zero_prob
                total_zero_prob = np.maximum(total_zero_prob, 1e-10)
                
                log_likelihood += np.sum(np.log(total_zero_prob))
            
            # For non-zero observations
            nonzero_mask = (y > 0)
            if np.any(nonzero_mask):
                y_nonzero = y[nonzero_mask]
                pi_nonzero = pi[nonzero_mask]
                mu_nonzero = mu[nonzero_mask]
                r_nonzero = r[nonzero_mask]
                
                # Negative binomial log-probability
                log_nb_prob = (loggamma(y_nonzero + r_nonzero) - 
                              loggamma(r_nonzero) - loggamma(y_nonzero + 1) +
                              r_nonzero * np.log(r_nonzero / (r_nonzero + mu_nonzero)) +
                              y_nonzero * np.log(mu_nonzero / (r_nonzero + mu_nonzero)))
                
                log_total_prob = np.log(1 - pi_nonzero) + log_nb_prob
                log_likelihood += np.sum(log_total_prob)
            
        except (OverflowError, UnderflowError, ValueError):
            # If numerical issues, return large negative likelihood
            return 1e10
        
        if not np.isfinite(log_likelihood):
            return 1e10
        
        return -log_likelihood
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict distribution parameters.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        proba_dict : dict
            Dictionary containing:
            - 'pi': Zero-inflation probabilities
            - 'mu': NB mean parameters  
            - 'r': NB dispersion parameters
            - 'zero_prob': Probability of zero outcome
        """
        self._check_fitted()
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        
        # Prepare design matrices
        if self.fit_intercept:
            X_pi = np.column_stack([np.ones(X.shape[0]), X])
            X_mu = np.column_stack([np.ones(X.shape[0]), X])
            X_r = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_pi = X.copy()
            X_mu = X.copy()
            X_r = X.copy()
            
        if self.dispersion_formula == 'constant':
            X_r = np.ones((X.shape[0], 1))
        
        # Compute π
        if self.pi_formula == 'intercept':
            linear_pi = np.full(X.shape[0], self.pi_params_[0])
        else:
            linear_pi = X_pi @ self.pi_params_
        pi = 1 / (1 + np.exp(-linear_pi))
        
        # Compute μ
        if self.mu_formula == 'log_linear':
            linear_mu = X_mu @ self.mu_params_
            mu = np.exp(linear_mu)
        else:
            mu = X_mu @ self.mu_params_
        mu = np.maximum(mu, 1e-8)
        
        # Compute r
        if self.dispersion_formula == 'constant':
            r = np.full(X.shape[0], np.exp(self.r_params_[0]))
        else:
            linear_r = X_r @ self.r_params_
            r = np.exp(linear_r)
        r = np.maximum(r, 1e-8)
        
        # Compute probability of zero
        nb_zero_prob = (r / (r + mu)) ** r
        zero_prob = pi + (1 - pi) * nb_zero_prob
        
        return {
            'pi': pi,
            'mu': mu, 
            'r': r,
            'zero_prob': zero_prob
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the ZINB model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : ndarray, shape (n_samples,)
            Predicted values (expected values of ZINB distribution)
        """
        proba_dict = self.predict_proba(X)
        
        # Expected value of ZINB: E[Y] = (1 - π) * μ
        expected_values = (1 - proba_dict['pi']) * proba_dict['mu']
        
        return expected_values
    
    def sample(self, X: np.ndarray, n_samples: int = 1,
              random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample from the fitted ZINB distribution.
        
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
            Samples from ZINB distribution
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        proba_dict = self.predict_proba(X)
        n_obs = X.shape[0]
        
        samples = np.zeros((n_obs, n_samples))
        
        for i in range(n_obs):
            pi_i = proba_dict['pi'][i]
            mu_i = proba_dict['mu'][i]
            r_i = proba_dict['r'][i]
            
            # Convert to negative binomial parameterization used by numpy
            # numpy uses (n, p) where n=r and p = r/(r+μ)
            p_i = r_i / (r_i + mu_i)
            
            for j in range(n_samples):
                # First decide if zero from excess zeros
                if np.random.random() < pi_i:
                    samples[i, j] = 0
                else:
                    # Sample from negative binomial
                    samples[i, j] = np.random.negative_binomial(r_i, p_i)
        
        return samples.astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the log-likelihood score of the model.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Prepare design matrices
        if self.fit_intercept:
            X_pi = np.column_stack([np.ones(X.shape[0]), X])
            X_mu = np.column_stack([np.ones(X.shape[0]), X])
            X_r = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_pi = X.copy()
            X_mu = X.copy()
            X_r = X.copy()
            
        if self.dispersion_formula == 'constant':
            X_r = np.ones((X.shape[0], 1))
        
        params = np.concatenate([self.pi_params_, self.mu_params_, self.r_params_])
        
        return -self._negative_log_likelihood(params, X_pi, X_mu, X_r, y)
    
    def _check_fitted(self) -> None:
        """Check if the model has been fitted."""
        if (self.pi_params_ is None or self.mu_params_ is None or 
            self.r_params_ is None):
            raise ValueError("Model has not been fitted yet. Call fit() first.")
    
    def get_params_summary(self) -> Dict[str, Any]:
        """Get summary of fitted parameters."""
        self._check_fitted()
        
        summary = {
            'pi_parameters': self.pi_params_.tolist(),
            'mu_parameters': self.mu_params_.tolist(), 
            'r_parameters': self.r_params_.tolist(),
            'log_likelihood': self.log_likelihood_,
            'converged': self.converged_,
            'n_iterations': self.n_iter_,
            'n_parameters': len(self.pi_params_) + len(self.mu_params_) + len(self.r_params_)
        }
        
        # Compute AIC and BIC if possible
        n_params = summary['n_parameters']
        if hasattr(self, 'n_samples_'):
            summary['AIC'] = 2 * n_params - 2 * self.log_likelihood_
            summary['BIC'] = np.log(self.n_samples_) * n_params - 2 * self.log_likelihood_
        
        return summary