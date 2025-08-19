"""
Test script for baseline statistical models.
"""

import sys
sys.path.append('/home/wentao/papercode/zero_inflated_comprehensive')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import our models
from models.baseline.zip_model import ZeroInflatedPoisson
from models.baseline.zinb_model import ZeroInflatedNegativeBinomial
from models.baseline.tweedie_glm import TweedieGLM
from models.baseline.hurdle_model import HurdleModel

# Import data generation
from generation.inject_zeros import inject_zeros


def generate_test_data(n_samples=1000, n_features=3, random_state=42):
    """Generate test data with covariates and zero-inflated counts."""
    np.random.seed(random_state)
    
    # Generate covariates
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Generate base counts (Poisson) - use appropriate number of coefficients
    if n_features == 2:
        coeffs = np.array([0.3, -0.2])
    elif n_features == 3:
        coeffs = np.array([0.3, -0.2, 0.4])
    else:
        coeffs = np.random.normal(0, 0.3, n_features)
    
    linear_pred = 0.5 + X @ coeffs
    mu = np.exp(linear_pred)
    
    # Sample from Poisson
    base_counts = np.random.poisson(mu)
    
    # Add zero-inflation
    zero_inflated_counts = inject_zeros(base_counts, mechanism='threshold', 
                                      zero_ratio=0.3, random_state=random_state)
    
    return X, zero_inflated_counts.astype(int)


def test_zip_model():
    """Test Zero-Inflated Poisson model."""
    print("Testing Zero-Inflated Poisson Model")
    print("=" * 50)
    
    # Generate data
    X, y = generate_test_data(random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")
    print(f"Zero ratio in training: {np.mean(y_train == 0):.3f}")
    print(f"Zero ratio in test: {np.mean(y_test == 0):.3f}")
    
    # Fit model
    try:
        zip_model = ZeroInflatedPoisson()
        zip_model.fit(X_train, y_train)
        
        print(f"Model converged: {zip_model.converged_}")
        print(f"Iterations: {zip_model.n_iter_}")
        
        # Make predictions
        y_pred = zip_model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        log_likelihood = zip_model.score(X_test, y_test)
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Log-likelihood: {log_likelihood:.4f}")
        
        # Get model summary
        summary = zip_model.get_params_summary()
        print(f"Number of parameters: {summary['n_parameters']}")
        if 'AIC' in summary:
            print(f"AIC: {summary['AIC']:.4f}")
        
        print("✓ ZIP model test successful")
        return True
        
    except Exception as e:
        print(f"✗ ZIP model test failed: {str(e)}")
        return False


def test_zinb_model():
    """Test Zero-Inflated Negative Binomial model."""
    print("\nTesting Zero-Inflated Negative Binomial Model")
    print("=" * 50)
    
    # Generate overdispersed data
    np.random.seed(456)
    X = np.random.normal(0, 1, (800, 2))
    
    # Generate NB data with higher variance
    linear_pred = 1.0 + X @ np.array([0.5, -0.3])
    mu = np.exp(linear_pred)
    
    # Sample from negative binomial (higher dispersion)
    r = 2.0  # dispersion parameter
    p = r / (r + mu)  # Convert to numpy parameterization
    base_counts = np.random.negative_binomial(r, p)
    
    # Add zeros
    y = inject_zeros(base_counts, mechanism='mixture', zero_ratio=0.25, random_state=456)
    y = y.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=456)
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Zero ratio in training: {np.mean(y_train == 0):.3f}")
    print(f"Mean count: {np.mean(y_train):.3f}")
    print(f"Variance: {np.var(y_train):.3f}")
    
    try:
        zinb_model = ZeroInflatedNegativeBinomial()
        zinb_model.fit(X_train, y_train)
        
        print(f"Model converged: {zinb_model.converged_}")
        print(f"Iterations: {zinb_model.n_iter_}")
        
        # Make predictions
        y_pred = zinb_model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        log_likelihood = zinb_model.score(X_test, y_test)
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Log-likelihood: {log_likelihood:.4f}")
        
        # Get model summary
        summary = zinb_model.get_params_summary()
        print(f"Estimated dispersion: {summary['r_parameters'][0]:.4f}")
        
        print("✓ ZINB model test successful")
        return True
        
    except Exception as e:
        print(f"✗ ZINB model test failed: {str(e)}")
        return False


def test_tweedie_glm():
    """Test Tweedie GLM model."""
    print("\nTesting Tweedie GLM Model")
    print("=" * 50)
    
    # Generate data more suitable for Tweedie
    np.random.seed(789)
    X = np.random.normal(0, 1, (600, 2))
    
    # Generate Tweedie-like data
    linear_pred = 0.8 + X @ np.array([0.4, -0.2])
    mu = np.exp(linear_pred)
    
    # Generate compound Poisson-Gamma (approximate Tweedie)
    power = 1.6
    lambda_param = mu**(2 - power) / (2 - power)
    alpha = (2 - power) / (power - 1)
    beta = (power - 1) * mu**(power - 1)
    
    y = np.zeros(len(mu))
    for i in range(len(mu)):
        # Number of jumps
        N = np.random.poisson(max(lambda_param[i], 1e-10))
        if N > 0:
            # Sum of gamma random variables
            y[i] = np.sum(np.random.gamma(alpha, beta[i], N))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=789)
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Zero ratio in training: {np.mean(y_train == 0):.3f}")
    print(f"Mean value: {np.mean(y_train):.3f}")
    
    try:
        tweedie_model = TweedieGLM(power=1.6)
        tweedie_model.fit(X_train, y_train)
        
        print(f"Model converged: {tweedie_model.converged_}")
        print(f"Iterations: {tweedie_model.n_iter_}")
        
        # Make predictions
        y_pred = tweedie_model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        deviance_score = tweedie_model.score(X_test, y_test)
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Deviance score: {deviance_score:.4f}")
        
        # Get model summary
        summary = tweedie_model.get_params_summary()
        print(f"Dispersion parameter: {summary['dispersion_parameter']:.4f}")
        
        print("✓ Tweedie GLM test successful")
        return True
        
    except Exception as e:
        print(f"✗ Tweedie GLM test failed: {str(e)}")
        return False


def test_hurdle_model():
    """Test Hurdle model."""
    print("\nTesting Hurdle Model")
    print("=" * 50)
    
    # Generate data for hurdle model
    np.random.seed(999)
    X = np.random.normal(0, 1, (700, 3))
    
    # Binary part: probability of non-zero
    linear_binary = -0.5 + X @ np.array([0.6, -0.4, 0.2])
    prob_nonzero = 1 / (1 + np.exp(-linear_binary))
    
    # Count part: Poisson for positive values
    linear_count = 0.3 + X @ np.array([0.3, 0.2, -0.1])
    mu_count = np.exp(linear_count)
    
    # Generate data
    y = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        if np.random.random() < prob_nonzero[i]:
            # Sample from truncated Poisson
            while True:
                count = np.random.poisson(mu_count[i])
                if count > 0:
                    y[i] = count
                    break
        # else y[i] remains 0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Zero ratio in training: {np.mean(y_train == 0):.3f}")
    print(f"Mean count (all): {np.mean(y_train):.3f}")
    print(f"Mean count (positive): {np.mean(y_train[y_train > 0]):.3f}")
    
    try:
        hurdle_model = HurdleModel(binary_model='logistic', count_model='poisson')
        hurdle_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = hurdle_model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        log_likelihood = hurdle_model.score(X_test, y_test)
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Log-likelihood: {log_likelihood:.4f}")
        
        # Get model summary
        summary = hurdle_model.get_params_summary()
        print(f"Binary parameters: {len(summary['binary_parameters'])}")
        print(f"Count parameters: {len(summary['count_parameters'])}")
        
        print("✓ Hurdle model test successful")
        return True
        
    except Exception as e:
        print(f"✗ Hurdle model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def compare_models():
    """Compare all models on the same dataset."""
    print("\nModel Comparison")
    print("=" * 50)
    
    # Generate common test data
    np.random.seed(1111)
    X, y = generate_test_data(n_samples=800, n_features=2, random_state=1111)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1111)
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Training zero ratio: {np.mean(y_train == 0):.3f}")
    print(f"Test zero ratio: {np.mean(y_test == 0):.3f}")
    
    models = {
        'ZIP': ZeroInflatedPoisson(),
        'ZINB': ZeroInflatedNegativeBinomial(),
        'Tweedie': TweedieGLM(power=1.5),
        'Hurdle': HurdleModel()
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            print(f"\nFitting {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            if hasattr(model, 'score'):
                score = model.score(X_test, y_test)
            else:
                score = None
            
            results[name] = {
                'MSE': mse,
                'MAE': mae,
                'Score': score,
                'Predictions': y_pred
            }
            
            print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}")
            if score is not None:
                print(f"  Log-likelihood: {score:.4f}")
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
            results[name] = None
    
    # Summary
    print("\nSUMMARY:")
    successful_models = {k: v for k, v in results.items() if v is not None}
    
    if successful_models:
        best_mse = min(successful_models.items(), key=lambda x: x[1]['MSE'])
        best_mae = min(successful_models.items(), key=lambda x: x[1]['MAE'])
        
        print(f"Best MSE: {best_mse[0]} ({best_mse[1]['MSE']:.4f})")
        print(f"Best MAE: {best_mae[0]} ({best_mae[1]['MAE']:.4f})")
    
    return results


if __name__ == "__main__":
    print("Baseline Statistical Models Test Suite")
    print("=" * 80)
    
    test_results = []
    
    # Run individual model tests
    test_results.append(('ZIP', test_zip_model()))
    test_results.append(('ZINB', test_zinb_model()))
    test_results.append(('Tweedie GLM', test_tweedie_glm()))
    test_results.append(('Hurdle', test_hurdle_model()))
    
    # Run comparison
    comparison_results = compare_models()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for model_name, success in test_results:
        status = "✓" if success else "✗"
        print(f"{status} {model_name}")
    
    successful_tests = sum(1 for _, success in test_results if success)
    total_tests = len(test_results)
    
    print(f"\nOverall: {successful_tests}/{total_tests} models working correctly")
    
    if successful_tests == total_tests:
        print("\n🎉 All baseline statistical models are working correctly!")
    else:
        print(f"\n⚠️  Some models failed. Check the detailed output above.")