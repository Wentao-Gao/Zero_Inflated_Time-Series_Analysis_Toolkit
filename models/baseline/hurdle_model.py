"""
Hurdle Models for zero-inflated time series.

Hurdle models separate the modeling of zero and non-zero outcomes into two parts:
1. A binary model for zero vs. non-zero (hurdle equation)
2. A truncated count model for positive values (count equation)

Mathematical formulation:
P(Y = 0) = f₁(0)
P(Y = y | Y > 0) = f₂(y) / (1 - F₂(0))  for y > 0

where f₁ is the binary model and f₂ is the count model.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import loggamma
from typing import Optional, Dict, Any, Union, Tuple, Literal
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.linear_model import LogisticRegression
import warnings


class HurdleModel(BaseEstimator, RegressorMixin):
    """
    Hurdle regression model for zero-inflated count data.
    
    The hurdle model consists of two parts:
    1. Binary model: Models the probability of observing zero vs non-zero
    2. Count model: Models the distribution of positive counts
    
    Parameters:
    -----------
    binary_model : str, default='logistic'
        Model for zero vs non-zero hurdle
        - 'logistic': Logistic regression
        - 'probit': Probit regression
        - 'complementary_log_log': Complementary log-log link
    
    count_model : str, default='poisson'
        Model for positive counts
        - 'poisson': Truncated Poisson
        - 'negative_binomial': Truncated Negative Binomial
        - 'geometric': Truncated Geometric (special case of NB with r=1)
        
    fit_intercept : bool, default=True
        Whether to fit intercept terms
        
    separate_models : bool, default=False
        Whether to use separate design matrices for binary and count parts
        If False, uses the same covariates for both parts
        
    max_iter : int, default=1000
        Maximum number of iterations for optimization
        
    tol : float, default=1e-6
        Tolerance for convergence
    """
    
    def __init__(self,
                 binary_model: str = 'logistic',
                 count_model: str = 'poisson',
                 fit_intercept: bool = True,
                 separate_models: bool = False,
                 max_iter: int = 1000,
                 tol: float = 1e-6):
        
        self.binary_model = binary_model
        self.count_model = count_model
        self.fit_intercept = fit_intercept
        self.separate_models = separate_models
        self.max_iter = max_iter
        self.tol = tol
        
        # Model parameters (set after fitting)
        self.binary_params_ = None
        self.count_params_ = None
        self.dispersion_ = None  # For negative binomial
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = None
        
        # Sub-models
        self._binary_model = None
        self._count_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray,
           X_binary: Optional[np.ndarray] = None,
           X_count: Optional[np.ndarray] = None) -> 'HurdleModel':
        """
        Fit the Hurdle model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data (used for both parts if separate matrices not provided)
        y : array-like, shape (n_samples,)
            Target values (non-negative integers)
        X_binary : array-like, optional
            Design matrix for binary part (if separate_models=True)
        X_count : array-like, optional
            Design matrix for count part (if separate_models=True)
            
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
        n_samples, _ = X.shape
        
        # Prepare design matrices
        if self.separate_models:
            if X_binary is None:
                X_binary = X.copy()
            if X_count is None:
                X_count = X.copy()
        else:
            X_binary = X.copy()
            X_count = X.copy()
        
        # Create binary outcome (0 if zero, 1 if positive)
        y_binary = (y > 0).astype(int)
        
        # Extract positive counts for count model
        positive_mask = y > 0
        if not np.any(positive_mask):
            raise ValueError("No positive values in target - cannot fit count model")
        
        y_count = y[positive_mask]
        X_count_pos = X_count[positive_mask]
        
        # Fit binary model
        self._fit_binary_model(X_binary, y_binary)
        
        # Fit count model on positive values
        self._fit_count_model(X_count_pos, y_count)
        
        # Compute overall log-likelihood
        self.log_likelihood_ = self._compute_log_likelihood(X_binary, X_count, y)
        
        return self
    
    def _fit_binary_model(self, X: np.ndarray, y_binary: np.ndarray) -> None:
        """Fit the binary (hurdle) part of the model."""
        
        if self.binary_model == 'logistic':
            from sklearn.linear_model import LogisticRegression
            
            binary_model = LogisticRegression(
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol
            )
            
            binary_model.fit(X, y_binary)
            self.binary_params_ = np.concatenate([
                binary_model.intercept_ if self.fit_intercept else [0],
                binary_model.coef_.flatten()
            ])
            
        elif self.binary_model == 'probit':
            # Custom probit implementation
            self.binary_params_ = self._fit_probit_model(X, y_binary)
            
        elif self.binary_model == 'complementary_log_log':
            # Custom complementary log-log implementation
            self.binary_params_ = self._fit_cloglog_model(X, y_binary)
            
        else:
            raise ValueError(f"Unknown binary model: {self.binary_model}")
    
    def _fit_count_model(self, X: np.ndarray, y_count: np.ndarray) -> None:
        """Fit the count part of the model (truncated at zero)."""
        
        if self.count_model == 'poisson':
            self.count_params_ = self._fit_truncated_poisson(X, y_count)
            
        elif self.count_model == 'negative_binomial':
            self.count_params_, self.dispersion_ = self._fit_truncated_nb(X, y_count)
            
        elif self.count_model == 'geometric':
            # Geometric is NB with r=1
            self.count_params_, self.dispersion_ = self._fit_truncated_geometric(X, y_count)
            
        else:
            raise ValueError(f"Unknown count model: {self.count_model}")
    
    def _fit_probit_model(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit probit model using maximum likelihood."""
        
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(len(y)), X])
        else:
            X_design = X.copy()
        
        def probit_neg_log_likelihood(params):
            eta = X_design @ params
            proba = stats.norm.cdf(eta)
            proba = np.clip(proba, 1e-10, 1 - 1e-10)
            
            log_like = np.sum(y * np.log(proba) + (1 - y) * np.log(1 - proba))
            return -log_like
        
        # Initialize with zeros
        init_params = np.zeros(X_design.shape[1])
        
        result = optimize.minimize(
            probit_neg_log_likelihood,
            init_params,
            method='BFGS',
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )
        
        return result.x
    
    def _fit_cloglog_model(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit complementary log-log model."""
        
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(len(y)), X])
        else:
            X_design = X.copy()
        
        def cloglog_neg_log_likelihood(params):
            eta = X_design @ params
            proba = 1 - np.exp(-np.exp(eta))
            proba = np.clip(proba, 1e-10, 1 - 1e-10)
            
            log_like = np.sum(y * np.log(proba) + (1 - y) * np.log(1 - proba))
            return -log_like
        
        init_params = np.zeros(X_design.shape[1])
        
        result = optimize.minimize(
            cloglog_neg_log_likelihood,
            init_params,
            method='BFGS',
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )
        
        return result.x
    
    def _fit_truncated_poisson(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit truncated Poisson model (truncated at 0)."""
        
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(len(y)), X])
        else:
            X_design = X.copy()
        
        def truncated_poisson_neg_log_likelihood(params):
            eta = X_design @ params
            mu = np.exp(eta)  # Log link
            
            # Truncated Poisson: P(Y=y|Y>0) = P(Y=y) / (1 - P(Y=0))
            # = (μ^y * e^(-μ) / y!) / (1 - e^(-μ))
            
            log_prob_y = y * np.log(mu) - mu - loggamma(y + 1)
            log_truncation_factor = -np.log(1 - np.exp(-mu))
            
            log_like = np.sum(log_prob_y + log_truncation_factor)
            
            if not np.isfinite(log_like):
                return 1e10
            
            return -log_like
        
        # Initialize with log of mean
        init_params = np.zeros(X_design.shape[1])
        if self.fit_intercept:
            init_params[0] = np.log(np.mean(y))
        
        result = optimize.minimize(
            truncated_poisson_neg_log_likelihood,
            init_params,
            method='BFGS',
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )
        
        return result.x
    
    def _fit_truncated_nb(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit truncated negative binomial model."""
        
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(len(y)), X])
        else:
            X_design = X.copy()
        
        def truncated_nb_neg_log_likelihood(params):
            beta_params = params[:-1]
            log_r = params[-1]  # Log-parameterize r for positivity
            
            eta = X_design @ beta_params
            mu = np.exp(eta)
            r = np.exp(log_r)
            
            # Truncated NB log-probability
            # P(Y=y|Y>0) = P(Y=y) / (1 - P(Y=0))
            
            # NB probability: Γ(y+r)/(Γ(r)Γ(y+1)) * (r/(r+μ))^r * (μ/(r+μ))^y
            log_prob_y = (loggamma(y + r) - loggamma(r) - loggamma(y + 1) +
                         r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
            
            # Truncation factor: 1 - P(Y=0) = 1 - (r/(r+μ))^r
            log_truncation_factor = -np.log(1 - (r / (r + mu))**r)
            
            log_like = np.sum(log_prob_y + log_truncation_factor)
            
            if not np.isfinite(log_like):
                return 1e10
            
            return -log_like
        
        # Initialize parameters
        init_beta = np.zeros(X_design.shape[1])
        if self.fit_intercept:
            init_beta[0] = np.log(np.mean(y))
        
        # Initialize r using method of moments
        mean_y = np.mean(y)
        var_y = np.var(y)
        if var_y > mean_y:
            init_r = mean_y**2 / (var_y - mean_y)
        else:
            init_r = 1.0
        
        init_params = np.concatenate([init_beta, [np.log(max(init_r, 0.1))]])
        
        result = optimize.minimize(
            truncated_nb_neg_log_likelihood,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )
        
        beta_params = result.x[:-1]
        dispersion = np.exp(result.x[-1])
        
        return beta_params, dispersion
    
    def _fit_truncated_geometric(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit truncated geometric model (special case of NB with r=1)."""
        
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(len(y)), X])
        else:
            X_design = X.copy()
        
        def truncated_geometric_neg_log_likelihood(params):
            eta = X_design @ params
            mu = np.exp(eta)
            
            # Geometric distribution: p = 1/(1+μ)
            p = 1 / (1 + mu)
            
            # Truncated geometric: P(Y=y|Y>0) = (1-p)^(y-1) * p / (1 - p)
            # = (1-p)^(y-1)
            log_prob = (y - 1) * np.log(1 - p)
            
            log_like = np.sum(log_prob)
            
            if not np.isfinite(log_like):
                return 1e10
            
            return -log_like
        
        # Initialize
        init_params = np.zeros(X_design.shape[1])
        if self.fit_intercept:
            init_params[0] = np.log(np.mean(y))
        
        result = optimize.minimize(
            truncated_geometric_neg_log_likelihood,
            init_params,
            method='BFGS',
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )
        
        return result.x, 1.0  # Dispersion is 1 for geometric
    
    def _compute_log_likelihood(self, X_binary: np.ndarray, X_count: np.ndarray,
                              y: np.ndarray) -> float:
        """Compute overall log-likelihood of the hurdle model."""
        
        # Binary part probabilities
        hurdle_proba = self._predict_hurdle_proba(X_binary)
        
        # Count part probabilities (for positive values)
        positive_mask = y > 0
        
        log_likelihood = 0.0
        
        # Zero observations
        zero_mask = y == 0
        if np.any(zero_mask):
            log_likelihood += np.sum(np.log(1 - hurdle_proba[zero_mask]))
        
        # Positive observations
        if np.any(positive_mask):
            X_count_pos = X_count[positive_mask]
            y_pos = y[positive_mask]
            hurdle_proba_pos = hurdle_proba[positive_mask]
            
            count_log_proba = self._predict_count_log_proba(X_count_pos, y_pos)
            
            log_likelihood += np.sum(np.log(hurdle_proba_pos) + count_log_proba)
        
        return log_likelihood
    
    def _predict_hurdle_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of crossing the hurdle (non-zero outcome)."""
        
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_design = X.copy()
        
        eta = X_design @ self.binary_params_
        
        if self.binary_model == 'logistic':
            return 1 / (1 + np.exp(-eta))
        elif self.binary_model == 'probit':
            return stats.norm.cdf(eta)
        elif self.binary_model == 'complementary_log_log':
            return 1 - np.exp(-np.exp(eta))
        else:
            raise ValueError(f"Unknown binary model: {self.binary_model}")
    
    def _predict_count_log_proba(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Predict log-probability for count model (given positive)."""
        
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_design = X.copy()
        
        eta = X_design @ self.count_params_
        
        if self.count_model == 'poisson':
            mu = np.exp(eta)
            log_prob_y = y * np.log(mu) - mu - loggamma(y + 1)
            log_truncation_factor = -np.log(1 - np.exp(-mu))
            return log_prob_y + log_truncation_factor
            
        elif self.count_model == 'negative_binomial':
            mu = np.exp(eta)
            r = self.dispersion_
            
            log_prob_y = (loggamma(y + r) - loggamma(r) - loggamma(y + 1) +
                         r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
            log_truncation_factor = -np.log(1 - (r / (r + mu))**r)
            return log_prob_y + log_truncation_factor
            
        elif self.count_model == 'geometric':
            mu = np.exp(eta)
            p = 1 / (1 + mu)
            return (y - 1) * np.log(1 - p)
            
        else:
            raise ValueError(f"Unknown count model: {self.count_model}")
    
    def predict(self, X: np.ndarray, X_binary: Optional[np.ndarray] = None,
               X_count: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict using the Hurdle model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
        X_binary : array-like, optional
            Design matrix for binary part (if separate_models=True)
        X_count : array-like, optional
            Design matrix for count part (if separate_models=True)
            
        Returns:
        --------
        y_pred : ndarray, shape (n_samples,)
            Predicted values (expected values of Hurdle distribution)
        """
        self._check_fitted()
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        
        if self.separate_models:
            if X_binary is None:
                X_binary = X.copy()
            if X_count is None:
                X_count = X.copy()
        else:
            X_binary = X.copy()
            X_count = X.copy()
        
        # Probability of crossing hurdle
        hurdle_proba = self._predict_hurdle_proba(X_binary)
        
        # Expected count given positive
        count_expected = self._predict_count_expected(X_count)
        
        # Overall expected value: E[Y] = P(Y > 0) * E[Y | Y > 0]
        return hurdle_proba * count_expected
    
    def _predict_count_expected(self, X: np.ndarray) -> np.ndarray:
        """Predict expected count given positive outcome."""
        
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_design = X.copy()
        
        eta = X_design @ self.count_params_
        
        if self.count_model == 'poisson':
            mu = np.exp(eta)
            # E[Y | Y > 0] for truncated Poisson
            return mu / (1 - np.exp(-mu))
            
        elif self.count_model == 'negative_binomial':
            mu = np.exp(eta)
            r = self.dispersion_
            # E[Y | Y > 0] for truncated NB
            truncation_prob = (r / (r + mu))**r
            return mu / (1 - truncation_prob)
            
        elif self.count_model == 'geometric':
            mu = np.exp(eta)
            # E[Y | Y > 0] for truncated geometric = μ
            return mu
            
        else:
            raise ValueError(f"Unknown count model: {self.count_model}")
    
    def sample(self, X: np.ndarray, n_samples: int = 1,
              X_binary: Optional[np.ndarray] = None,
              X_count: Optional[np.ndarray] = None,
              random_state: Optional[int] = None) -> np.ndarray:
        """Sample from the fitted Hurdle distribution."""
        
        if random_state is not None:
            np.random.seed(random_state)
        
        if self.separate_models:
            if X_binary is None:
                X_binary = X.copy()
            if X_count is None:
                X_count = X.copy()
        else:
            X_binary = X.copy()
            X_count = X.copy()
        
        n_obs = X.shape[0]
        samples = np.zeros((n_obs, n_samples), dtype=int)
        
        # Get probabilities
        hurdle_proba = self._predict_hurdle_proba(X_binary)
        
        for i in range(n_obs):
            for j in range(n_samples):
                # First, decide if we cross the hurdle
                if np.random.random() < hurdle_proba[i]:
                    # Sample from count model
                    samples[i, j] = self._sample_count_model(X_count[i:i+1])[0]
                else:
                    # Zero outcome
                    samples[i, j] = 0
        
        return samples
    
    def _sample_count_model(self, X: np.ndarray) -> np.ndarray:
        """Sample from the count model (given positive)."""
        
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_design = X.copy()
        
        eta = X_design @ self.count_params_
        
        if self.count_model == 'poisson':
            mu = np.exp(eta)
            # Sample from truncated Poisson
            samples = []
            for mu_i in mu:
                while True:
                    sample = np.random.poisson(mu_i)
                    if sample > 0:
                        samples.append(sample)
                        break
            return np.array(samples)
            
        elif self.count_model in ['negative_binomial', 'geometric']:
            mu = np.exp(eta)
            if self.count_model == 'geometric':
                r = 1.0
            else:
                r = self.dispersion_
            
            # Convert to numpy parameterization: n=r, p=r/(r+μ)
            p = r / (r + mu)
            
            samples = []
            for i in range(len(mu)):
                while True:
                    sample = np.random.negative_binomial(r, p[i])
                    if sample > 0:
                        samples.append(sample)
                        break
            return np.array(samples)
        
        else:
            raise ValueError(f"Unknown count model: {self.count_model}")
    
    def score(self, X: np.ndarray, y: np.ndarray,
             X_binary: Optional[np.ndarray] = None,
             X_count: Optional[np.ndarray] = None) -> float:
        """Return the log-likelihood score."""
        
        if self.separate_models:
            if X_binary is None:
                X_binary = X.copy()
            if X_count is None:
                X_count = X.copy()
        else:
            X_binary = X.copy()
            X_count = X.copy()
        
        return self._compute_log_likelihood(X_binary, X_count, y)
    
    def _check_fitted(self) -> None:
        """Check if the model has been fitted."""
        if self.binary_params_ is None or self.count_params_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
    
    def get_params_summary(self) -> Dict[str, Any]:
        """Get summary of fitted parameters."""
        self._check_fitted()
        
        summary = {
            'binary_model': self.binary_model,
            'count_model': self.count_model,
            'binary_parameters': self.binary_params_.tolist(),
            'count_parameters': self.count_params_.tolist(),
            'log_likelihood': self.log_likelihood_,
            'separate_models': self.separate_models
        }
        
        if self.dispersion_ is not None:
            summary['dispersion_parameter'] = self.dispersion_
        
        return summary