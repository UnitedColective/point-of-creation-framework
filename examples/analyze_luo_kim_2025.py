"""
Point of Creation Framework Analysis: Luo & Kim (2025) Decision Commitment Data

This script analyzes commitment times from rodent perceptual decision-making experiments.
The core question: Do commitment decisions show evidence of generative self-governance
(δ > 0) or collapse to white-noise floor (δ ≈ 0)?

Author: Adapted for PCF analysis
License: MIT
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))

from pcf_core import (
    compute_pcf, 
    test_against_white_noise,
    windowed_pcf,
    detrend_series
)


def load_decision_data(filepath):
    """
    Load the Luo & Kim (2025) decision commitment data.
    
    Args:
        filepath: Path to CSV with columns:
            - recording_id
            - index_in_Trials
            - stereoclick_time_s
            - movementtime_s
            - commitment_timestep
            - commitment_time_s
            - cpoke_in
            - gamma
    
    Returns:
        DataFrame with data
    """
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} trials from {len(df['recording_id'].unique())} recording sessions")
    return df


def analyze_full_timeseries(df):
    """
    Analyze commitment times as a unified time series across all trials.
    
    This treats the sequence of commitment times as a single dynamical process,
    asking: does the animal's commitment timing show generative structure?
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: Full Time Series (All Trials)")
    print("="*70)
    
    X = df['commitment_time_s'].values
    
    print(f"\nTime series properties:")
    print(f"  - N trials: {len(X)}")
    print(f"  - Mean commitment time: {np.mean(X):.3f} s")
    print(f"  - Std deviation: {np.std(X):.3f} s")
    print(f"  - Min/Max: [{np.min(X):.3f}, {np.max(X):.3f}] s")
    
    # Compute PCF
    result = compute_pcf(X, ci=0.95, n_bootstrap=1000)
    
    print(f"\n🔬 PCF Results:")
    print(f"  - δ (delta) = {result['delta']:.4f}")
    print(f"  - 95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"  - ρ₁ (lag-1 autocorrelation) = {result['rho1']:.4f}")
    print(f"  - Above white-noise floor: {result['above_floor']}")
    
    # Statistical test
    sig_test = test_against_white_noise(result, n_trials=10000)
    print(f"\n📊 Significance Test (vs. White Noise):")
    print(f"  - Z-score: {sig_test['z_score']:.3f}")
    print(f"  - p-value: {sig_test['p_value']:.6f}")
    print(f"  - Significantly above floor (p<0.05): {sig_test['significantly_above_floor']}")
    
    if result['above_floor']:
        print(f"\n✅ EVIDENCE OF STRUCTURE: Commitment timing shows generative dynamics")
        print(f"   (δ > 0 indicates temporal coherence beyond white noise)")
    else:
        print(f"\n⚠️  COLLAPSE TO WHITE-NOISE FLOOR: Commitment timing is random/unpatterned")
    
    return result


def analyze_by_stimulus_strength(df):
    """
    Analyze commitment times stratified by stimulus strength (gamma).
    
    High |γ| = easy decision (strong evidence for one option)
    Low |γ| = hard decision (weak evidence, near indifference)
    
    Question: Does stimulus difficulty affect decision structure?
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: By Stimulus Strength (γ)")
    print("="*70)
    
    # Create difficulty bins
    df['difficulty'] = pd.cut(np.abs(df['gamma']), 
                              bins=3, 
                              labels=['Easy', 'Medium', 'Hard'])
    
    results_by_difficulty = {}
    
    for difficulty in ['Hard', 'Medium', 'Easy']:
        X = df[df['difficulty'] == difficulty]['commitment_time_s'].values
        
        if len(X) < 10:
            print(f"\n⚠️  {difficulty} decisions: Only {len(X)} trials, skipping")
            continue
        
        result = compute_pcf(X, ci=0.95, n_bootstrap=500)
        sig_test = test_against_white_noise(result, n_trials=5000)
        
        print(f"\n{difficulty} decisions (|γ| bins): N = {len(X)}")
        print(f"  - δ = {result['delta']:.4f}  [95% CI: {result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        print(f"  - Above floor: {result['above_floor']}")
        print(f"  - p-value: {sig_test['p_value']:.4f}")
        
        results_by_difficulty[difficulty] = result
    
    return results_by_difficulty


def analyze_by_recording_session(df):
    """
    Analyze commitment times separately for each recording session.
    
    This reveals whether structure is consistent across animals/days or
    if certain sessions show different dynamics.
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: By Recording Session")
    print("="*70)
    
    results_by_session = {}
    sessions = df['recording_id'].unique()
    
    print(f"\nAnalyzing {len(sessions)} recording sessions...\n")
    
    for session in sorted(sessions):
        X = df[df['recording_id'] == session]['commitment_time_s'].values
        
        if len(X) < 10:
            continue
        
        result = compute_pcf(X, ci=0.95, n_bootstrap=500)
        sig_test = test_against_white_noise(result, n_trials=5000)
        
        print(f"{session}: N={len(X):3d} trials | δ={result['delta']:+.4f} | "
              f"above_floor={result['above_floor']} | p={sig_test['p_value']:.4f}")
        
        results_by_session[session] = result
    
    # Summary statistics
    deltas = [r['delta'] for r in results_by_session.values()]
    print(f"\nSummary across sessions:")
    print(f"  - Mean δ: {np.mean(deltas):.4f}")
    print(f"  - Std δ: {np.std(deltas):.4f}")
    print(f"  - Sessions with δ > 0: {sum(1 for r in results_by_session.values() if r['above_floor'])}/{len(results_by_session)}")
    
    return results_by_session


def analyze_windowed_dynamics(df):
    """
    Compute PCF over sliding windows to track how commitment structure
    evolves across the experimental session.
    
    Question: Does the animal's decision process show stable or drifting dynamics?
    """
    print("\n" + "="*70)
    print("ANALYSIS 4: Windowed Dynamics (Temporal Evolution)")
    print("="*70)
    
    X = df['commitment_time_s'].values
    window_size = 50  # 50-trial windows
    step_size = 25    # 25-trial steps (50% overlap)
    
    windows = windowed_pcf(X, window_size=window_size, step_size=step_size)
    
    print(f"\nWindow analysis: size={window_size}, step={step_size}")
    print(f"Total windows: {len(windows)}\n")
    
    for i, w in enumerate(windows):
        trial_range = f"Trials {w['window_start']}-{w['window_end']}"
        above = "✓" if w['above_floor'] else "✗"
        print(f"  {above} Window {i:2d} ({trial_range:25s}): δ = {w['delta']:+.4f}")
    
    return windows


def compare_detrended(df):
    """
    Compare PCF before and after removing linear trend.
    
    This tests whether results are robust to detrending or if there's
    an underlying linear drift in commitment times.
    """
    print("\n" + "="*70)
    print("ANALYSIS 5: Robustness - Detrending Check")
    print("="*70)
    
    X = df['commitment_time_s'].values
    
    # Original
    result_orig = compute_pcf(X, ci=0.95, n_bootstrap=1000)
    
    # Detrended
    X_detrended = detrend_series(X, method='linear')
    result_detrended = compute_pcf(X_detrended, ci=0.95, n_bootstrap=1000)
    
    print(f"\nOriginal time series:")
    print(f"  - δ = {result_orig['delta']:.4f}")
    print(f"  - Above floor: {result_orig['above_floor']}")
    
    print(f"\nAfter linear detrending:")
    print(f"  - δ = {result_detrended['delta']:.4f}")
    print(f"  - Above floor: {result_detrended['above_floor']}")
    
    print(f"\nChange in δ: {abs(result_detrended['delta'] - result_orig['delta']):.4f}")
    print(f"  (δ < 0.01 change indicates robustness)")
    
    return result_orig, result_detrended


def main(data_path):
    """
    Run complete PCF analysis pipeline on decision commitment data.
    
    Args:
        data_path: Path to CSV file with Luo & Kim (2025) data
    """
    print("\n" + "="*70)
    print("POINT OF CREATION FRAMEWORK: Decision Commitment Analysis")
    print("Luo & Kim (2025) - Rodent Perceptual Decision-Making")
    print("="*70)
    
    # Load data
    df = load_decision_data(data_path)
    
    # Run analyses
    result_full = analyze_full_timeseries(df)
    results_by_difficulty = analyze_by_stimulus_strength(df)
    results_by_session = analyze_by_recording_session(df)
    windows = analyze_windowed_dynamics(df)
    result_orig, result_detrended = compare_detrended(df)
    
    # Summary
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
δ > 0 and p < 0.05:
  ✓ Commitment timing shows GENERATIVE SELF-GOVERNANCE
  ✓ Decisions exhibit temporal coherence (memory/structure)
  ✓ Evidence against purely stochastic decision-making

δ ≈ 0 or δ > 0 but p > 0.05:
  ⚠ Commitment timing COLLAPSES TO WHITE-NOISE FLOOR
  ⚠ Each trial's commitment time is essentially random/independent
  ⚠ No evidence of generative structure

Interpretation in context:
  • If δ > 0 across sessions: Robust decision structure
  • If δ varies by stimulus strength: Difficulty modulates structure
  • If δ drifts in windowed analysis: Learning or fatigue effects
  • Robustness to detrending: Rules out simple temporal drift
    """)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_luo_kim_2025.py <path_to_data.csv>")
        print("\nExample:")
        print("  python analyze_luo_kim_2025.py /data/decision_dynamics_commitment.csv")
        sys.exit(1)
    
    data_path = sys.argv[1]
    main(data_path)
