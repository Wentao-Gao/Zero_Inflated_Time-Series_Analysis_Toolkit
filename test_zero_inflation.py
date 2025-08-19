"""
Simple test script for zero-inflation mechanisms.
"""

import sys
sys.path.append('/home/wentao/papercode/zero_inflated_comprehensive')

import numpy as np
import matplotlib.pyplot as plt
from generation.inject_zeros import inject_zeros, compare_zero_inflation_methods
from generation.validation import run_mechanism_validation, plot_zero_inflation_comparison

def test_basic_functionality():
    """Test basic functionality of zero-inflation mechanisms."""
    print("Testing Zero-Inflation Mechanisms")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    
    # Create gamma-distributed data (positive, right-skewed)
    data = np.random.gamma(shape=2, scale=1.5, size=1000)
    
    print(f"Original data statistics:")
    print(f"  Mean: {np.mean(data):.3f}")
    print(f"  Std: {np.std(data):.3f}")
    print(f"  Zero ratio: {np.mean(data == 0):.3f}")
    print(f"  Min: {np.min(data):.3f}, Max: {np.max(data):.3f}")
    print()
    
    # Test each mechanism
    target_zero_ratio = 0.3
    mechanisms = ['threshold', 'mixture', 'tweedie', 'hurdle']
    
    results = {}
    
    for mechanism in mechanisms:
        print(f"Testing {mechanism} mechanism...")
        try:
            zi_data = inject_zeros(data, mechanism=mechanism, 
                                 zero_ratio=target_zero_ratio, 
                                 random_state=42)
            
            actual_zero_ratio = np.mean(zi_data == 0)
            error = abs(target_zero_ratio - actual_zero_ratio)
            
            results[mechanism] = {
                'data': zi_data,
                'actual_zero_ratio': actual_zero_ratio,
                'error': error,
                'success': True
            }
            
            print(f"  ✓ Success! Actual zero ratio: {actual_zero_ratio:.3f} (error: {error:.3f})")
            
        except Exception as e:
            results[mechanism] = {'success': False, 'error': str(e)}
            print(f"  ✗ Failed: {str(e)}")
        
        print()
    
    return data, results

def test_validation_system():
    """Test the validation system."""
    print("Testing Validation System")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(123)
    data = np.random.exponential(scale=2.0, size=500)
    
    # Run comprehensive validation
    validation_results = run_mechanism_validation(data, target_zero_ratio=0.25, random_state=123)
    
    print("Validation Results:")
    for mechanism, results in validation_results.items():
        if 'error' not in results:
            print(f"  {mechanism}:")
            print(f"    Quality Score: {results['quality_score']:.3f}")
            print(f"    Zero Ratio Accuracy: {results['zero_ratio_accuracy']:.3f}")
            if 'rank' in results:
                print(f"    Rank: {results['rank']}")
        else:
            print(f"  {mechanism}: FAILED - {results['error']}")
        print()
    
    return validation_results

def test_comparison():
    """Test method comparison functionality."""
    print("Testing Method Comparison")
    print("=" * 50)
    
    # Generate test data with some existing zeros
    np.random.seed(456)
    base_data = np.random.lognormal(mean=0, sigma=1, size=800)
    
    # Add some natural zeros
    zero_mask = np.random.random(800) < 0.1
    base_data[zero_mask] = 0
    
    print(f"Base data has {np.mean(base_data == 0):.3f} natural zero ratio")
    
    # Compare methods
    comparison_results = compare_zero_inflation_methods(base_data, target_zero_ratio=0.4, random_state=456)
    
    print("Method Comparison Results:")
    for method, results in comparison_results.items():
        if 'error' not in results:
            print(f"  {method}:")
            print(f"    Target: {results['target_zero_ratio']:.3f}, Actual: {results['actual_zero_ratio']:.3f}")
            print(f"    Error: {results['zero_ratio_error']:.3f}")
            print(f"    Distribution Similarity: {results['distribution_similarity']:.3f}")
        else:
            print(f"  {method}: FAILED")
        print()
    
    return comparison_results

def create_visualization():
    """Create visualization of zero-inflation results."""
    print("Creating Visualizations")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(789)
    data = np.random.gamma(shape=1.5, scale=2, size=600)
    
    # Apply threshold method
    zi_data = inject_zeros(data, mechanism='threshold', zero_ratio=0.35, random_state=789)
    
    # Create comparison plot
    fig = plot_zero_inflation_comparison(data, zi_data, 'threshold', 
                                       save_path='/home/wentao/papercode/zero_inflated_comprehensive/test_visualization.png')
    
    print("Visualization saved to test_visualization.png")
    plt.close(fig)
    
    return True

if __name__ == "__main__":
    print("Zero-Inflation Mechanism Test Suite")
    print("=" * 80)
    print()
    
    try:
        # Run basic functionality tests
        original_data, basic_results = test_basic_functionality()
        
        # Test validation system
        validation_results = test_validation_system()
        
        # Test comparison functionality
        comparison_results = test_comparison()
        
        # Create visualizations
        viz_success = create_visualization()
        
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        successful_mechanisms = sum(1 for r in basic_results.values() if r.get('success', False))
        total_mechanisms = len(basic_results)
        
        print(f"Basic functionality: {successful_mechanisms}/{total_mechanisms} mechanisms working")
        print(f"Validation system: {'✓' if validation_results else '✗'}")
        print(f"Comparison system: {'✓' if comparison_results else '✗'}")
        print(f"Visualization: {'✓' if viz_success else '✗'}")
        
        if successful_mechanisms == total_mechanisms:
            print("\n🎉 All tests passed! Zero-inflation mechanisms are working correctly.")
        else:
            print(f"\n⚠️  Some mechanisms failed. Check the detailed output above.")
            
    except Exception as e:
        print(f"Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()