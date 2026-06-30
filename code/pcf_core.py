
"""
Point of Creation Framework (PCF) - Core Implementation

This module provides the fundamental tools for computing the PCF metric
across any time series data.

Author: A.M. Sterling
License: MIT
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, List


def compute_first_difference(X: np.ndarray) -> np.ndarray:
    """
    Compute first difference of a time series.
    
    Args:
        X: Input time series (1D array)
    
    Returns:
        First difference array (length = len(X) - 1)
    """
    return np.diff(X, n=1)


def compute_second_difference(X: np.ndarray) -> np.ndarray:
    """
    Compute second difference of a time series.
    
    Args:
        X: Input time series (1D array)
    
    Returns:
        Second difference array (length = len(X) - 2)
    """
    return np.diff(X, n=2)


def lag_1_autocorrelation(X: np.ndarray) -> float:
    """
    Compute lag-1 autocorrelation of a time series.
    
    Args:
        X: Input time series (1D array)
    
    Returns:
        Lag-1 autocorrelation coefficient
    """
    if len(X) < 2:
        return np.nan
    
    X_centered = X - np.mean(X)
    c0 = np.sum(X_centered ** 2) / len(X)
    c1 = np.sum(X_centered[:-1] * X_centered[1:]) / len(X)
    
    return c1 / c0 if c0 != 0 else np.nan


def compute_pcf(X: np.ndarray, ci: float = 0.95,
                n_bootstrap: int = 1000, block_method: str = 'stationary') -> Dict:
    """
    Compute the Point of Creation Framework metric.
    
    Formula: δ = ρ₁(Δ²X) + 2/3
    
    Where:
        - Δ²X is the second difference
        - ρ₁ is the lag-1 autocorrelation
        - Expected value under white noise: -2/3
        - δ > 0: structure above white-noise floor
        - δ ≈ 0: collapse to white-noise floor
    
    Args:
        X: Input time series (1D array, length >= 3)
        ci: Confidence interval (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples for CI estimation
        block_method: Bootstrap method ('stationary' or 'moving')
    
    Returns:
        Dictionary with keys:
            - 'delta': PCF metric value
            - 'rho1': Lag-1 autocorrelation of second differences
            - 'ci_lower': Lower confidence bound
            - 'ci_upper': Upper confidence bound
            - 'white_noise_floor': Reference floor (-2/3)
            - 'above_floor': Boolean (True if δ > 0)
            - 'second_differences': The Δ²X array used
    """
    
    if len(X) < 3:
        raise ValueError("Input series must have length >= 3")
    
    # Compute second differences
    d2 = compute_second_difference(X)
    
    # Compute lag-1 autocorrelation of second differences
    rho1 = lag_1_autocorrelation(d2)
    
    # Define white-noise floor
    WHITE_NOISE_FLOOR = -2/3
    
    # Compute PCF metric: δ = ρ₁(Δ²X) + 2/3
    # FIX: delta = rho1 - WHITE_NOISE_FLOOR (i.e. rho1 + 2/3)
    # White noise has ρ₁ ≈ -2/3, so δ ≈ 0 as expected
    delta = rho1 - WHITE_NOISE_FLOOR
    
    # Bootstrap confidence interval
    ci_lower, ci_upper = _bootstrap_ci(X, ci, n_bootstrap, block_method)
    
    return {
        'delta': delta,
        'rho1': rho1,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'white_noise_floor': WHITE_NOISE_FLOOR,
        'above_floor': delta > 0,
        'second_differences': d2,
        'n_samples': len(X),
        'n_second_diff': len(d2)
    }


def _bootstrap_ci(X: np.ndarray, ci: float, n_bootstrap: int,
                  block_method: str) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for PCF metric.
    
    Args:
        X: Input time series
        ci: Confidence level (e.g., 0.95)
        n_bootstrap: Number of bootstrap samples
        block_method: Resampling method
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    
    deltas = []
    n = len(X)
    block_size = int(np.sqrt(n))
    
    for _ in range(n_bootstrap):
        if block_method == 'stationary':
            X_boot = _stationary_block_bootstrap(X, block_size)
        else:
            X_boot = _moving_block_bootstrap(X, block_size)
        
        d2_boot = compute_second_difference(X_boot)
        rho1_boot = lag_1_autocorrelation(d2_boot)
        # FIX: delta = rho1 + 2/3 (consistent with compute_pcf)
        delta_boot = rho1_boot + 2/3
        deltas.append(delta_boot)
    
    deltas = np.array(deltas)
    alpha = 1 - ci
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(deltas, lower_percentile)
    ci_upper = np.percentile(deltas, upper_percentile)
    
    return ci_lower, ci_upper


def _stationary_block_bootstrap(X: np.ndarray, block_size: int) -> np.ndarray:
    """Stationary block bootstrap resampling."""
    n = len(X)
    n_blocks = int(np.ceil(n / block_size))
    indices = []
    
    for _ in range(n_blocks):
        start = np.random.randint(0, n - block_size + 1)
        indices.extend(range(start, start + block_size))
    
    return X[indices[:n]]


def _moving_block_bootstrap(X: np.ndarray, block_size: int) -> np.ndarray:
    """Moving block bootstrap resampling."""
    n = len(X)
    n_blocks = int(np.ceil(n / block_size))
    indices = np.random.choice(n - block_size + 1, n_blocks) + np.arange(block_size)[:, np.newaxis]
    indices = indices.ravel()[:n]
    
    return X[indices]


def test_against_white_noise(pcf_result: Dict, n_trials: int = 10000) -> Dict:
    """
    Test whether observed δ is significantly different from white noise.
    
    Args:
        pcf_result: Output dictionary from compute_pcf()
        n_trials: Number of white noise trials
    
    Returns:
        Dictionary with test statistics
    """
    
    WHITE_NOISE_FLOOR = -2/3
    observed_delta = pcf_result['delta']
    
    wn_deltas = []
    for _ in range(n_trials):
        wn = np.random.randn(pcf_result['n_samples'])
        d2_wn = compute_second_difference(wn)
        rho1_wn = lag_1_autocorrelation(d2_wn)
        delta_wn = rho1_wn - WHITE_NOISE_FLOOR
        wn_deltas.append(delta_wn)
    
    wn_deltas = np.array(wn_deltas)
    mean_wn = np.mean(wn_deltas)
    std_wn = np.std(wn_deltas)
    
    z_score = (observed_delta - mean_wn) / std_wn if std_wn > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
    
    return {
        'observed_delta': observed_delta,
        'white_noise_mean': mean_wn,
        'white_noise_std': std_wn,
        'z_score': z_score,
        'p_value': p_value,
        'significantly_above_floor': p_value < 0.05
    }


def detrend_series(X: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Detrend a time series.
    
    Args:
        X: Input time series
        method: 'linear' or 'polynomial'
    
    Returns:
        Detrended series
    """
    if method == 'linear':
        return stats.detrend(X)
    elif method == 'polynomial':
        coeffs = np.polyfit(np.arange(len(X)), X, 2)
        trend = np.polyval(coeffs, np.arange(len(X)))
        return X - trend
    else:
        raise ValueError("method must be 'linear' or 'polynomial'")


def windowed_pcf(X: np.ndarray, window_size: int,
                 step_size: int = None) -> List[Dict]:
    """
    Compute PCF over sliding windows.
    
    Args:
        X: Input time series
        window_size: Window length
        step_size: Step between windows (default: window_size / 2)
    
    Returns:
        List of PCF results for each window
    """
    
    if step_size is None:
        step_size = window_size // 2
    
    results = []
    for start in range(0, len(X) - window_size, step_size):
        X_window = X[start:start + window_size]
        result = compute_pcf(X_window)
        result['window_start'] = start
        result['window_end'] = start + window_size
        results.append(result)
    
    return results


# Example usage and testing
if __name__ == "__main__":
    
    print("Point of Creation Framework - Core Module")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Test 1: White noise (δ should be ≈ 0)
    print("\nTest 1: Pure White Noise")
    wn = np.random.randn(5000)
    result_wn = compute_pcf(wn)
    print(f"  δ = {result_wn['delta']:.4f}  (expect ≈ 0)")
    print(f"  95% CI: [{result_wn['ci_lower']:.4f}, {result_wn['ci_upper']:.4f}]")
    print(f"  Above floor: {result_wn['above_floor']}")
    
    # Test 2: AR(1) process (δ should be > 0)
    print("\nTest 2: AR(1) Process (ρ = 0.7)")
    ar1 = np.zeros(5000)
    ar1[0] = np.random.randn()
    for t in range(1, len(ar1)):
        ar1[t] = 0.7 * ar1[t-1] + np.random.randn()
    result_ar1 = compute_pcf(ar1)
    print(f"  δ = {result_ar1['delta']:.4f}  (expect > 0)")
    print(f"  95% CI: [{result_ar1['ci_lower']:.4f}, {result_ar1['ci_upper']:.4f}]")
    print(f"  Above floor: {result_ar1['above_floor']}")
    
    # Test 3: Significance test
    print("\nTest 3: Significance Test (AR(1) vs White Noise)")
    sig_test = test_against_white_noise(result_ar1)
    print(f"  Z-score: {sig_test['z_score']:.3f}")
    print(f"  p-value: {sig_test['p_value']:.4f}")
    print(f"  Significantly above floor: {sig_test['significantly_above_floor']}")
    
    print("\n" + "=" * 50)
    print("Tests complete.")
