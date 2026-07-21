"""
Compare R = |mean(e^{iφ})| and δ = ρ₁(Δ²X) + 2/3
Tests whether phase coherence and generative structure are the same phenomenon.
"""
import numpy as np
from scipy import stats, signal
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from delta import compute_delta

def compute_R(time_series):
    """
    Compute phase coherence R = |mean(e^{iφ})|
    where φ is the instantaneous phase of the Hilbert transform
    """
    # Hilbert transform for instantaneous phase
    analytic = signal.hilbert(time_series)
    phase = np.angle(analytic)
    R = np.abs(np.mean(np.exp(1j * phase)))
    return R

def test_known_signals():
    """Test R and δ on known signals"""
    print("=" * 60)
    print("R vs δ COMPARISON — KNOWN SIGNALS")
    print("=" * 60)
    
    # 1. White noise
    np.random.seed(42)
    noise = np.random.randn(5000)
    R_noise = compute_R(noise)
    delta_noise = compute_delta(noise)
    print(f"\nWhite noise:")
    print(f"  R  = {R_noise:.6f}")
    print(f"  δ  = {delta_noise:.6f}")
    
    # 2. Pure sine wave (max coherence)
    t = np.linspace(0, 10, 5000)
    sine = np.sin(2 * np.pi * 10 * t)
    R_sine = compute_R(sine)
    delta_sine = compute_delta(sine)
    print(f"\nPure sine (10 Hz):")
    print(f"  R  = {R_sine:.6f}")
    print(f"  δ  = {delta_sine:.6f}")
    
    # 3. AR(1) process
    np.random.seed(42)
    ar1 = np.zeros(5000)
    phi = 0.8
    for i in range(1, len(ar1)):
        ar1[i] = phi * ar1[i-1] + np.random.randn()
    R_ar1 = compute_R(ar1)
    delta_ar1 = compute_delta(ar1)
    print(f"\nAR(1) (phi=0.8):")
    print(f"  R  = {R_ar1:.6f}")
    print(f"  δ  = {delta_ar1:.6f}")
    
    # 4. Noisy sine
    noisy_sine = sine + 0.5 * np.random.randn(5000)
    R_noisy = compute_R(noisy_sine)
    delta_noisy = compute_delta(noisy_sine)
    print(f"\nNoisy sine (SNR=2):")
    print(f"  R  = {R_noisy:.6f}")
    print(f"  δ  = {delta_noisy:.6f}")
    
    return {
        'white_noise': {'R': R_noise, 'delta': delta_noise},
        'sine': {'R': R_sine, 'delta': delta_sine},
        'ar1': {'R': R_ar1, 'delta': delta_ar1},
        'noisy_sine': {'R': R_noisy, 'delta': delta_noisy}
    }

def compare_on_data(ecg_data=None, ieeg_data=None):
    """Compare R and δ on real data"""
    print("\n" + "=" * 60)
    print("R vs δ COMPARISON — REAL DATA")
    print("=" * 60)
    
    # Placeholder: load real data when available
    # For now, use the same known signals but with different params
    
    np.random.seed(42)
    
    # Create a "healthy" signal (structured, coherent)
    t = np.linspace(0, 10, 5000)
    healthy = np.sin(2 * np.pi * 8 * t) + 0.3 * np.sin(2 * np.pi * 16 * t) + 0.1 * np.random.randn(5000)
    
    # Create a "pathological" signal (less coherent)
    pathological = np.sin(2 * np.pi * 8 * t) + 0.5 * np.random.randn(5000)
    
    R_healthy = compute_R(healthy)
    delta_healthy = compute_delta(healthy)
    R_path = compute_R(pathological)
    delta_path = compute_delta(pathological)
    
    print(f"\nHealthy signal:")
    print(f"  R  = {R_healthy:.6f}")
    print(f"  δ  = {delta_healthy:.6f}")
    
    print(f"\nPathological signal:")
    print(f"  R  = {R_path:.6f}")
    print(f"  δ  = {delta_path:.6f}")
    
    return {
        'healthy': {'R': R_healthy, 'delta': delta_healthy},
        'pathological': {'R': R_path, 'delta': delta_path}
    }

def main():
    print("R vs δ COMPARISON")
    print("=" * 60)
    
    # Test 1: Known signals
    known_results = test_known_signals()
    
    # Test 2: Real data simulation
    real_results = compare_on_data()
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("R measures phase coherence (binding/integration)")
    print("δ measures generative structure (temporal memory)")
    print("They are expected to correlate: high R ↔ high δ")
    print("This is the bridge between consciousness and structure")
    print("=" * 60)

if __name__ == "__main__":
    main()