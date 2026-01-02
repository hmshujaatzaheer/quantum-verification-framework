"""
Example: Complete Verification Workflow

Demonstrates the unified verification framework for fault-tolerant
quantum random sampling, implementing Algorithm 1 from the research proposal.

This example:
1. Creates cluster states with various k-uniformity
2. Performs standard DFE verification
3. Performs adaptive verification exploiting k-uniform structure
4. Compares with XEB benchmarking
5. Analyzes noise effects

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from stabilizer_tableau import StabilizerTableau, KUniformityCalculator
from fidelity_estimation import (
    DirectFidelityEstimator,
    AdaptiveStabilizerSampler,
    HybridVerificationProtocol,
    NoiseParameters,
    CrossEntropyBenchmark
)
from circuits import (
    create_cluster_state_circuit,
    create_surface_code_circuit,
    VerificationVisualizer
)


def example_1_basic_verification():
    """
    Example 1: Basic Direct Fidelity Estimation
    
    Demonstrates standard DFE on a cluster state.
    """
    print("=" * 70)
    print("Example 1: Basic Direct Fidelity Estimation")
    print("=" * 70)
    
    # Create 3x3 cluster state
    tableau = StabilizerTableau.cluster_state(3, 3)
    print(f"\nCreated 3x3 cluster state ({tableau.n_qubits} qubits)")
    
    # Analyze k-uniformity
    calculator = KUniformityCalculator(tableau)
    print("\nk-Uniformity Analysis:")
    for k in range(1, 5):
        delta = calculator.compute_delta(k)
        status = "✓ k-uniform" if delta < 1e-10 else f"Δ = {delta:.4f}"
        print(f"  k={k}: {status}")
    
    # Perform DFE with different noise levels
    print("\nDirect Fidelity Estimation:")
    noise_levels = [0, 0.001, 0.005, 0.01, 0.02]
    
    for p in noise_levels:
        noise = NoiseParameters(p3=p)
        dfe = DirectFidelityEstimator(tableau, noise)
        result = dfe.estimate_fidelity(n_measurements=500)
        print(f"  p_meas={p:.3f}: F={result.fidelity:.4f} ± {np.sqrt(result.variance):.4f}, "
              f"TVD≤{result.tvd_bound:.4f}")
    
    return tableau


def example_2_adaptive_sampling():
    """
    Example 2: Adaptive Stabilizer Sampling (Algorithm 1)
    
    Demonstrates improved verification using k-uniform structure.
    """
    print("\n" + "=" * 70)
    print("Example 2: Adaptive Stabilizer Sampling")
    print("=" * 70)
    
    # Create 4x4 cluster state
    tableau = StabilizerTableau.cluster_state(4, 4)
    print(f"\nCreated 4x4 cluster state ({tableau.n_qubits} qubits)")
    
    # Initialize adaptive sampler
    adaptive = AdaptiveStabilizerSampler(tableau)
    
    # Identify k-uniform regions
    print("\nIdentifying k-uniform regions...")
    regions = adaptive.identify_k_uniform_regions(k_max=3)
    for k, subsystems in regions.items():
        print(f"  k={k}: {len(subsystems)} k-uniform subsystems found")
    
    # Compute priority stabilizers
    priority = adaptive.compute_priority_stabilizers()
    print(f"\nPriority stabilizers identified: {len(priority)}")
    
    # Compare standard vs adaptive estimation
    noise = NoiseParameters(p3=0.01)
    n_trials = 10
    
    print("\nComparison (average over 10 trials, p_meas=0.01):")
    
    # Standard DFE
    dfe = DirectFidelityEstimator(tableau, noise)
    dfe_fidelities = []
    for _ in range(n_trials):
        result = dfe.estimate_fidelity(n_measurements=500)
        dfe_fidelities.append(result.fidelity)
    
    # Adaptive sampling
    adaptive_fidelities = []
    for _ in range(n_trials):
        result = adaptive.adaptive_estimate(epsilon=0.05, delta=0.05, noise=noise)
        adaptive_fidelities.append(result.fidelity)
    
    print(f"  Standard DFE: F = {np.mean(dfe_fidelities):.4f} ± {np.std(dfe_fidelities):.4f}")
    print(f"  Adaptive:     F = {np.mean(adaptive_fidelities):.4f} ± {np.std(adaptive_fidelities):.4f}")
    print(f"  Variance reduction: {(np.var(dfe_fidelities) / np.var(adaptive_fidelities)):.2f}x")
    
    return adaptive


def example_3_hybrid_protocol():
    """
    Example 3: Hybrid Physical-Logical Verification
    
    Demonstrates the hybrid scheme from Majidy et al. (2025).
    """
    print("\n" + "=" * 70)
    print("Example 3: Hybrid Physical-Logical Verification")
    print("=" * 70)
    
    # Compare different code distances
    print("\nLogical error rates vs code distance:")
    for d in [3, 5, 7]:
        hybrid = HybridVerificationProtocol(code_distance=d, n_logical=10)
        
        print(f"\n  Surface code d={d}:")
        for p_phys in [1e-3, 5e-3, 1e-2]:
            p_log = hybrid.logical_error_rate(p_phys)
            print(f"    p_phys={p_phys:.4f} → p_log={p_log:.2e}")
    
    # Verification fidelity bounds
    print("\nVerification fidelity bounds (d=5, n_logical=10):")
    hybrid = HybridVerificationProtocol(code_distance=5, n_logical=10)
    
    for p2 in [1e-4, 1e-3, 5e-3]:
        noise = NoiseParameters(p1=p2/10, p2=p2, p3=p2)
        bound = hybrid.verification_fidelity_bound(noise)
        print(f"  p_2Q={p2:.4f}: F_bound ≥ {bound:.4f}")


def example_4_noise_analysis():
    """
    Example 4: Noise Model Analysis
    
    Studies how different noise sources affect verification.
    """
    print("\n" + "=" * 70)
    print("Example 4: Noise Analysis")
    print("=" * 70)
    
    tableau = StabilizerTableau.cluster_state(3, 3)
    
    # Vary measurement noise
    print("\nEffect of measurement noise (1000 measurements):")
    p_values = np.linspace(0, 0.1, 11)
    fidelities = []
    
    for p in p_values:
        noise = NoiseParameters(p3=p)
        dfe = DirectFidelityEstimator(tableau, noise)
        result = dfe.estimate_fidelity(n_measurements=1000)
        fidelities.append(result.fidelity)
        
    # Print results
    for p, f in zip(p_values, fidelities):
        print(f"  p={p:.2f}: F={f:.4f}")
    
    # Expected: F ≈ 1 - 2*p for small p (single-qubit errors flip outcomes)
    print("\nTheoretical comparison:")
    print("  For depolarizing measurement noise: F ≈ 1 - 2p")
    print(f"  At p=0.05: F_theory ≈ {1 - 2*0.05:.4f}, F_measured ≈ {fidelities[5]:.4f}")
    
    return p_values, fidelities


def example_5_sample_complexity():
    """
    Example 5: Sample Complexity Analysis
    
    Compares sample requirements for DFE vs XEB.
    """
    print("\n" + "=" * 70)
    print("Example 5: Sample Complexity Comparison")
    print("=" * 70)
    
    tableau = StabilizerTableau.cluster_state(4, 4)
    dfe = DirectFidelityEstimator(tableau)
    
    print("\nSample complexity for (ε, δ)-accuracy:")
    print("\n  DFE (polynomial in 1/ε):")
    for eps in [0.1, 0.05, 0.01, 0.005]:
        M = dfe.required_measurements(eps, delta=0.05)
        print(f"    ε={eps:.3f}: M = {M:,} measurements")
    
    print("\n  XEB (requires exponential classical computation):")
    for n in [9, 16, 25, 36]:
        # XEB needs O(2^n) classical operations
        classical_ops = 2**n
        print(f"    N={n}: ~{classical_ops:,} classical operations (infeasible for N>50)")
    
    print("\n  Key insight: DFE is both sample-efficient AND computationally efficient")
    print("              XEB is sample-efficient but computationally EXPONENTIAL")


def example_6_full_benchmark():
    """
    Example 6: Full Benchmarking Suite
    
    Comprehensive comparison across multiple metrics.
    """
    print("\n" + "=" * 70)
    print("Example 6: Full Benchmarking Suite")
    print("=" * 70)
    
    # Test different cluster sizes
    sizes = [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4)]
    
    print("\nCluster State Benchmarks (p_meas=0.01, 500 measurements):")
    print("-" * 70)
    print(f"{'Size':<10} {'Qubits':<8} {'Max k':<8} {'DFE F':<12} {'TVD Bound':<12}")
    print("-" * 70)
    
    noise = NoiseParameters(p3=0.01)
    
    for rows, cols in sizes:
        tableau = StabilizerTableau.cluster_state(rows, cols)
        n_qubits = tableau.n_qubits
        
        # Find max k-uniformity
        calc = KUniformityCalculator(tableau)
        max_k = calc.find_max_k_uniformity()
        
        # DFE
        dfe = DirectFidelityEstimator(tableau, noise)
        result = dfe.estimate_fidelity(n_measurements=500)
        
        print(f"{rows}x{cols:<7} {n_qubits:<8} {max_k:<8} "
              f"{result.fidelity:<12.4f} {result.tvd_bound:<12.4f}")
    
    print("-" * 70)


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("UNIFIED QUANTUM VERIFICATION FRAMEWORK - EXAMPLES")
    print("=" * 70)
    print("\nImplementation of research proposal algorithms for")
    print("fault-tolerant verification of quantum random sampling.")
    print("\nReferences:")
    print("  [1] Ringbauer et al., Nature Communications 16, 106 (2025)")
    print("  [2] Majidy, Hangleiter & Gullans, arXiv:2503.14506 (2025)")
    
    # Run examples
    example_1_basic_verification()
    example_2_adaptive_sampling()
    example_3_hybrid_protocol()
    example_4_noise_analysis()
    example_5_sample_complexity()
    example_6_full_benchmark()
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
