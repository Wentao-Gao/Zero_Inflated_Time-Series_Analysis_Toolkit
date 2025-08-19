"""
Validation tools for zero-inflation mechanisms.

This module provides utilities to validate the correctness and quality
of zero-inflation implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, Optional, Tuple
import warnings


def validate_zero_inflation_quality(original_data: np.ndarray,
                                  zero_inflated_data: np.ndarray,
                                  mechanism: str,
                                  target_zero_ratio: float) -> Dict[str, Any]:
    """
    Validate the quality of zero-inflation results.
    
    Args:
        original_data: Original time series data
        zero_inflated_data: Zero-inflated time series data
        mechanism: Zero-inflation mechanism used
        target_zero_ratio: Target proportion of zeros
        
    Returns:
        Dictionary with validation metrics and results
    """
    results = {}
    
    # Basic statistics
    actual_zero_ratio = np.mean(zero_inflated_data == 0)
    zero_ratio_error = abs(target_zero_ratio - actual_zero_ratio)
    
    # Non-zero data comparison
    orig_nonzero = original_data[original_data > 0]
    zi_nonzero = zero_inflated_data[zero_inflated_data > 0]
    
    results.update({
        'target_zero_ratio': target_zero_ratio,
        'actual_zero_ratio': actual_zero_ratio,
        'zero_ratio_error': zero_ratio_error,
        'zero_ratio_accuracy': 1 - zero_ratio_error,
        'mechanism': mechanism
    })
    
    # Statistical tests
    if len(orig_nonzero) > 10 and len(zi_nonzero) > 10:
        # Kolmogorov-Smirnov test for distribution similarity
        ks_stat, ks_pval = stats.ks_2samp(orig_nonzero, zi_nonzero)
        results.update({
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'distribution_similar': ks_pval > 0.05
        })
        
        # Anderson-Darling test if samples are from same distribution
        try:
            ad_stat, ad_crit, ad_sig = stats.anderson_ksamp([orig_nonzero, zi_nonzero])
            results.update({
                'ad_statistic': ad_stat,
                'ad_critical_values': ad_crit,
                'ad_significance_level': ad_sig
            })
        except Exception:
            pass
    
    # Moment comparison
    if len(zi_nonzero) > 0:
        results.update({
            'original_mean': np.mean(orig_nonzero) if len(orig_nonzero) > 0 else 0,
            'zi_mean_nonzero': np.mean(zi_nonzero),
            'original_var': np.var(orig_nonzero) if len(orig_nonzero) > 0 else 0,
            'zi_var_nonzero': np.var(zi_nonzero),
            'overall_mean_preservation': abs(np.mean(original_data) - np.mean(zero_inflated_data)),
            'overall_var_preservation': abs(np.var(original_data) - np.var(zero_inflated_data))
        })
    
    # Quality score
    quality_score = _compute_quality_score(results)
    results['quality_score'] = quality_score
    
    return results


def _compute_quality_score(validation_results: Dict[str, Any]) -> float:
    """
    Compute an overall quality score for zero-inflation.
    
    Score ranges from 0 (poor) to 1 (excellent).
    """
    score_components = []
    
    # Zero ratio accuracy (weight: 0.4)
    zero_ratio_acc = validation_results.get('zero_ratio_accuracy', 0)
    score_components.append(0.4 * zero_ratio_acc)
    
    # Distribution preservation (weight: 0.3)
    if 'ks_pvalue' in validation_results:
        # Higher p-value means better preservation (up to p=1)
        dist_score = min(validation_results['ks_pvalue'], 1.0)
        score_components.append(0.3 * dist_score)
    
    # Moment preservation (weight: 0.3)
    if 'overall_mean_preservation' in validation_results:
        # Lower difference is better
        mean_diff = validation_results['overall_mean_preservation']
        orig_mean = validation_results.get('original_mean', 1)
        mean_score = 1 - min(mean_diff / (abs(orig_mean) + 1e-8), 1.0)
        score_components.append(0.15 * max(mean_score, 0))
    
    if 'overall_var_preservation' in validation_results:
        var_diff = validation_results['overall_var_preservation'] 
        orig_var = validation_results.get('original_var', 1)
        var_score = 1 - min(var_diff / (abs(orig_var) + 1e-8), 1.0)
        score_components.append(0.15 * max(var_score, 0))
    
    return sum(score_components)


def plot_zero_inflation_comparison(original_data: np.ndarray,
                                 zero_inflated_data: np.ndarray,
                                 mechanism: str,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comparison plots for original vs zero-inflated data.
    
    Args:
        original_data: Original time series data
        zero_inflated_data: Zero-inflated time series data
        mechanism: Zero-inflation mechanism used
        save_path: Path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Zero-Inflation Comparison: {mechanism.title()} Method', fontsize=16)
    
    # Time series plots
    axes[0, 0].plot(original_data, alpha=0.7, label='Original', linewidth=1)
    axes[0, 0].plot(zero_inflated_data, alpha=0.7, label='Zero-Inflated', linewidth=1)
    axes[0, 0].set_title('Time Series Comparison')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histograms
    axes[0, 1].hist(original_data, bins=50, alpha=0.6, label='Original', density=True)
    axes[0, 1].hist(zero_inflated_data, bins=50, alpha=0.6, label='Zero-Inflated', density=True)
    axes[0, 1].set_title('Distribution Comparison')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    orig_nonzero = original_data[original_data > 0]
    zi_nonzero = zero_inflated_data[zero_inflated_data > 0]
    
    if len(orig_nonzero) > 0 and len(zi_nonzero) > 0:
        quantiles_orig = np.percentile(orig_nonzero, np.linspace(1, 99, 50))
        quantiles_zi = np.percentile(zi_nonzero, np.linspace(1, 99, 50))
        
        axes[0, 2].scatter(quantiles_orig, quantiles_zi, alpha=0.6)
        min_val = min(quantiles_orig.min(), quantiles_zi.min())
        max_val = max(quantiles_orig.max(), quantiles_zi.max())
        axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 2].set_title('Q-Q Plot (Non-zero values)')
        axes[0, 2].set_xlabel('Original Quantiles')
        axes[0, 2].set_ylabel('Zero-Inflated Quantiles')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Zero pattern analysis
    zero_positions = np.where(zero_inflated_data == 0)[0]
    if len(zero_positions) > 1:
        zero_intervals = np.diff(zero_positions)
        axes[1, 0].hist(zero_intervals, bins=30, alpha=0.7)
        axes[1, 0].set_title('Zero Intervals Distribution')
        axes[1, 0].set_xlabel('Interval Length')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Box plots
    box_data = []
    box_labels = []
    if len(orig_nonzero) > 0:
        box_data.append(orig_nonzero)
        box_labels.append('Original\n(Non-zero)')
    if len(zi_nonzero) > 0:
        box_data.append(zi_nonzero)
        box_labels.append('Zero-Inflated\n(Non-zero)')
    
    if box_data:
        axes[1, 1].boxplot(box_data, labels=box_labels)
        axes[1, 1].set_title('Box Plot Comparison')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Statistics summary
    orig_zero_ratio = np.mean(original_data == 0)
    zi_zero_ratio = np.mean(zero_inflated_data == 0)
    
    stats_text = f"""
    Original Zero Ratio: {orig_zero_ratio:.3f}
    ZI Zero Ratio: {zi_zero_ratio:.3f}
    
    Original Mean: {np.mean(original_data):.3f}
    ZI Mean: {np.mean(zero_inflated_data):.3f}
    
    Original Std: {np.std(original_data):.3f}
    ZI Std: {np.std(zero_inflated_data):.3f}
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    axes[1, 2].set_title('Summary Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def run_mechanism_validation(data: np.ndarray,
                           mechanisms: Optional[list] = None,
                           target_zero_ratio: float = 0.3,
                           random_state: Optional[int] = 42) -> Dict[str, Dict[str, Any]]:
    """
    Run validation for multiple zero-inflation mechanisms.
    
    Args:
        data: Original time series data
        mechanisms: List of mechanisms to validate
        target_zero_ratio: Target proportion of zeros
        random_state: Random seed
        
    Returns:
        Dictionary with validation results for each mechanism
    """
    if mechanisms is None:
        mechanisms = ['threshold', 'mixture', 'tweedie', 'hurdle']
    
    # Import injection functions
    from .inject_zeros import inject_zeros
    
    results = {}
    
    for mechanism in mechanisms:
        try:
            # Apply zero-inflation
            zi_data = inject_zeros(data, mechanism=mechanism,
                                 zero_ratio=target_zero_ratio,
                                 random_state=random_state)
            
            # Validate results
            validation = validate_zero_inflation_quality(data, zi_data, mechanism, target_zero_ratio)
            results[mechanism] = validation
            
        except Exception as e:
            results[mechanism] = {
                'error': str(e),
                'quality_score': 0.0
            }
    
    # Rank mechanisms by quality score
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        ranked_mechanisms = sorted(valid_results.keys(), 
                                 key=lambda x: valid_results[x]['quality_score'], 
                                 reverse=True)
        
        # Add ranking information
        for i, mechanism in enumerate(ranked_mechanisms):
            results[mechanism]['rank'] = i + 1
            results[mechanism]['is_best'] = (i == 0)
    
    return results


def check_mathematical_properties(data: np.ndarray, 
                                 mechanism: str,
                                 **mechanism_params) -> Dict[str, bool]:
    """
    Check if zero-inflation preserves expected mathematical properties.
    
    Args:
        data: Zero-inflated time series data
        mechanism: Mechanism used for zero-inflation
        **mechanism_params: Parameters used for the mechanism
        
    Returns:
        Dictionary of boolean checks for mathematical properties
    """
    checks = {}
    
    # Basic validity checks
    checks['all_finite'] = np.all(np.isfinite(data))
    checks['non_negative'] = np.all(data >= 0)
    checks['has_zeros'] = np.any(data == 0)
    checks['has_positive'] = np.any(data > 0)
    
    # Distribution-specific checks
    if mechanism == 'tweedie':
        power = mechanism_params.get('power', 1.5)
        checks['valid_tweedie_power'] = 1 < power < 2
        
        # Tweedie should have compound Poisson-Gamma structure
        zero_ratio = np.mean(data == 0)
        checks['reasonable_zero_ratio'] = 0 < zero_ratio < 1
        
    elif mechanism == 'mixture':
        # Mixture should preserve non-zero distribution shape
        zero_prob = mechanism_params.get('zero_probability', 0.3)
        actual_zero_ratio = np.mean(data == 0)
        checks['mixture_zero_ratio_reasonable'] = abs(actual_zero_ratio - zero_prob) < 0.1
        
    elif mechanism == 'threshold':
        # Threshold should create clear separation
        zeros = data[data == 0]
        nonzeros = data[data > 0]
        if len(nonzeros) > 0:
            threshold_val = mechanism_params.get('threshold_value', 0)
            checks['threshold_separation'] = np.min(nonzeros) > threshold_val
        
    elif mechanism == 'hurdle':
        # Hurdle should have two-part structure
        checks['hurdle_two_part'] = np.any(data == 0) and np.any(data > 0)
    
    # Statistical moment checks
    if len(data) > 1:
        checks['finite_variance'] = np.isfinite(np.var(data))
        checks['finite_mean'] = np.isfinite(np.mean(data))
        
        # Check for extreme outliers (more than 5 std devs from mean)
        if np.std(data) > 0:
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            checks['no_extreme_outliers'] = np.all(z_scores < 5)
        else:
            checks['no_extreme_outliers'] = True
    
    return checks


def generate_validation_report(validation_results: Dict[str, Dict[str, Any]],
                             save_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive validation report.
    
    Args:
        validation_results: Results from run_mechanism_validation
        save_path: Path to save the report
        
    Returns:
        Report as string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ZERO-INFLATION VALIDATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary
    valid_mechanisms = [k for k, v in validation_results.items() if 'error' not in v]
    error_mechanisms = [k for k, v in validation_results.items() if 'error' in v]
    
    report_lines.append(f"Mechanisms tested: {len(validation_results)}")
    report_lines.append(f"Successful: {len(valid_mechanisms)}")
    report_lines.append(f"Failed: {len(error_mechanisms)}")
    report_lines.append("")
    
    if error_mechanisms:
        report_lines.append("FAILED MECHANISMS:")
        for mech in error_mechanisms:
            error_msg = validation_results[mech]['error']
            report_lines.append(f"  - {mech}: {error_msg}")
        report_lines.append("")
    
    # Detailed results for valid mechanisms
    if valid_mechanisms:
        report_lines.append("MECHANISM RANKINGS (by quality score):")
        report_lines.append("-" * 50)
        
        # Sort by quality score
        sorted_mechs = sorted(valid_mechanisms, 
                            key=lambda x: validation_results[x]['quality_score'], 
                            reverse=True)
        
        for i, mech in enumerate(sorted_mechs):
            results = validation_results[mech]
            report_lines.append(f"{i+1}. {mech.upper()}")
            report_lines.append(f"   Quality Score: {results['quality_score']:.3f}")
            report_lines.append(f"   Zero Ratio Error: {results['zero_ratio_error']:.3f}")
            report_lines.append(f"   Target: {results['target_zero_ratio']:.3f}, "
                               f"Actual: {results['actual_zero_ratio']:.3f}")
            
            if 'ks_pvalue' in results:
                report_lines.append(f"   Distribution Similarity (p-value): {results['ks_pvalue']:.3f}")
            
            report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text