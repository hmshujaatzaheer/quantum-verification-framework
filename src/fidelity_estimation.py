"""
Direct Fidelity Estimation for MBQC Verification

Implements efficient verification protocols for quantum random sampling
using stabilizer measurements on cluster states.

Based on:
- Ringbauer et al. (2025): Nature Communications 16, 106
- Flammia & Liu, Phys. Rev. Lett. 106, 230501 (2011)

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

from stabilizer_tableau import StabilizerTableau, KUniformityCalculator


class NoiseModel(Enum):
    """Noise models for quantum circuit simulation."""
    NOISELESS = "noiseless"
    DEPOLARIZING = "depolarizing"
    DEPHASING = "dephasing"
    AMPLITUDE_DAMPING = "amplitude_damping"


@dataclass
class NoiseParameters:
    """
    Noise parameters for depolarizing channel model.
    
    Following Majidy et al. (2025), we use:
    - p0: idle error rate
    - p1: single-qubit gate error rate
    - p2: two-qubit gate error rate
    - p3: measurement error rate
    """
    p0: float = 1e-5  # Idle
    p1: float = 1e-4  # Single-qubit
    p2: float = 1e-3  # Two-qubit
    p3: float = 1e-3  # Measurement
    
    def total_error_per_qubit(self, n_single: int, n_two: int, 
                               n_idle: int, n_meas: int) -> float:
        """Calculate total error probability for a qubit."""
        return (1 - (1 - self.p0)**n_idle * 
                    (1 - self.p1)**n_single * 
                    (1 - self.p2)**n_two *
                    (1 - self.p3)**n_meas)


@dataclass
class FidelityEstimate:
    """Result of fidelity estimation."""
    fidelity: float
    variance: float
    confidence_interval: Tuple[float, float]
    n_measurements: int
    root_infidelity: float
    tvd_bound: float  # Upper bound on total variation distance
    
    def __repr__(self):
        return (f"FidelityEstimate(F={self.fidelity:.4f} ± {np.sqrt(self.variance):.4f}, "
                f"√(1-F)={self.root_infidelity:.4f}, TVD≤{self.tvd_bound:.4f})")


class DirectFidelityEstimator:
    """
    Direct Fidelity Estimation for stabilizer states.
    
    Estimates fidelity by measuring random elements of the stabilizer group:
    
    F = (1/2^N) Σ_{s∈S} ⟨s⟩_ρ
    
    Reference: Ringbauer et al. (2025), Eq. (2)
    """
    
    def __init__(self, tableau: StabilizerTableau, 
                 noise: Optional[NoiseParameters] = None):
        """
        Initialize DFE with a target stabilizer state.
        
        Args:
            tableau: StabilizerTableau of the target state
            noise: Optional noise parameters for simulation
        """
        self.tableau = tableau
        self.n_qubits = tableau.n_qubits
        self.noise = noise or NoiseParameters()
        
    def estimate_fidelity(self, n_measurements: int,
                          state_preparation: Optional[Callable] = None,
                          confidence: float = 0.95) -> FidelityEstimate:
        """
        Estimate fidelity using random stabilizer measurements.
        
        Args:
            n_measurements: Number of stabilizer measurements M
            state_preparation: Optional function to prepare experimental state
            confidence: Confidence level for interval (default 0.95)
            
        Returns:
            FidelityEstimate with fidelity bounds
        """
        outcomes = []
        
        for _ in range(n_measurements):
            # Sample random stabilizer group element
            pauli, phase = self.tableau.sample_stabilizer()
            
            # Simulate measurement (with noise if applicable)
            if state_preparation is not None:
                outcome = state_preparation(pauli, phase)
            else:
                outcome = self._simulate_measurement(pauli, phase)
                
            outcomes.append(outcome)
        
        # Compute fidelity estimate
        outcomes = np.array(outcomes)
        fidelity = np.mean(outcomes)
        
        # Variance estimation (Eq. 8 from Ringbauer et al.)
        # For M=1 shots per stabilizer: Var[F̂] = (4/KM) E[p_s(1-p_s)]
        variance = np.var(outcomes) / n_measurements
        
        # Confidence interval
        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        margin = z * np.sqrt(variance)
        ci = (max(0, fidelity - margin), min(1, fidelity + margin))
        
        # Root infidelity bounds TVD
        root_infidelity = np.sqrt(max(0, 1 - fidelity))
        
        return FidelityEstimate(
            fidelity=fidelity,
            variance=variance,
            confidence_interval=ci,
            n_measurements=n_measurements,
            root_infidelity=root_infidelity,
            tvd_bound=root_infidelity
        )
    
    def _simulate_measurement(self, pauli: np.ndarray, phase: int) -> float:
        """
        Simulate measurement of a Pauli operator on the ideal state with noise.
        
        For ideal stabilizer state, all stabilizers have eigenvalue +1.
        Noise introduces errors that flip the measurement outcome.
        
        Args:
            pauli: Binary representation of Pauli operator
            phase: Phase of the stabilizer (0=+1, 1=-1)
            
        Returns:
            Measurement outcome σ ∈ {+1, -1}
        """
        # Ideal outcome is +1 for stabilizers (or -1 if phase=1)
        ideal_outcome = 1 if phase == 0 else -1
        
        # Apply measurement noise
        if np.random.random() < self.noise.p3:
            return -ideal_outcome
        
        return ideal_outcome
    
    def required_measurements(self, epsilon: float, delta: float) -> int:
        """
        Calculate required number of measurements for (ε, δ)-accuracy.
        
        From Hoeffding bound: M ≥ (4/ε²) ln(2/δ)
        
        Args:
            epsilon: Additive error bound
            delta: Failure probability
            
        Returns:
            Required number of measurements
        """
        return int(np.ceil(4 * np.log(2/delta) / epsilon**2))


class AdaptiveStabilizerSampler:
    """
    Adaptive Stabilizer Sampling for Verification (Algorithm 1)
    
    Exploits k-uniform structure to prioritize stabilizers for
    more efficient fidelity estimation.
    
    This is the main algorithmic contribution of the research proposal.
    """
    
    def __init__(self, tableau: StabilizerTableau):
        """
        Initialize adaptive sampler.
        
        Args:
            tableau: StabilizerTableau of target state
        """
        self.tableau = tableau
        self.n_qubits = tableau.n_qubits
        self.k_calculator = KUniformityCalculator(tableau)
        self._priority_stabilizers = None
        self._k_uniform_regions = None
        
    def identify_k_uniform_regions(self, k_max: int = None) -> dict:
        """
        Identify k-uniform subsystems in the state.
        
        Args:
            k_max: Maximum k to check (default: n_qubits // 2)
            
        Returns:
            Dictionary mapping k to list of k-uniform subsystems
        """
        if k_max is None:
            k_max = min(self.n_qubits // 2, 4)  # Limit for tractability
            
        k_uniform_regions = {}
        
        for k in range(1, k_max + 1):
            regions = self.k_calculator.get_k_uniform_subsystems(k)
            if regions:
                k_uniform_regions[k] = regions
                
        self._k_uniform_regions = k_uniform_regions
        return k_uniform_regions
    
    def compute_priority_stabilizers(self) -> List[np.ndarray]:
        """
        Identify stabilizers supported on k-uniform regions.
        
        These stabilizers have lower variance under uniform noise
        and should be prioritized for measurement.
        
        Returns:
            List of priority stabilizer bitstrings
        """
        if self._k_uniform_regions is None:
            self.identify_k_uniform_regions()
            
        priority_stabilizers = []
        
        # Find stabilizers whose support is contained in k-uniform regions
        for k, regions in self._k_uniform_regions.items():
            for region in regions:
                # Create stabilizer bitstring for this region
                bitstring = np.zeros(self.n_qubits, dtype=np.int8)
                for idx in region:
                    bitstring[idx] = 1
                priority_stabilizers.append(bitstring)
                
        self._priority_stabilizers = priority_stabilizers
        return priority_stabilizers
    
    def adaptive_estimate(self, epsilon: float, delta: float,
                          noise: Optional[NoiseParameters] = None,
                          priority_fraction: float = 0.5) -> FidelityEstimate:
        """
        Perform adaptive fidelity estimation (Algorithm 1 from proposal).
        
        Args:
            epsilon: Target accuracy
            delta: Failure probability
            noise: Noise parameters
            priority_fraction: Fraction of measurements on priority stabilizers
            
        Returns:
            FidelityEstimate result
        """
        # Compute required measurements
        M = int(np.ceil(4 * np.log(2/delta) / epsilon**2))
        
        # Identify priority stabilizers
        if self._priority_stabilizers is None:
            self.compute_priority_stabilizers()
            
        n_priority = int(M * priority_fraction)
        n_uniform = M - n_priority
        
        noise = noise or NoiseParameters()
        outcomes = []
        
        # Phase 1: Priority stabilizers
        for i in range(n_priority):
            if self._priority_stabilizers:
                # Sample from priority set
                idx = np.random.randint(len(self._priority_stabilizers))
                bitstring = self._priority_stabilizers[idx]
            else:
                # Fall back to uniform sampling
                bitstring = np.random.randint(0, 2, self.n_qubits)
                
            pauli, phase = self.tableau.get_stabilizer_group_element(bitstring)
            outcome = self._measure_with_noise(pauli, phase, noise)
            outcomes.append(outcome)
            
        # Phase 2: Uniform sampling from full stabilizer group
        for i in range(n_uniform):
            pauli, phase = self.tableau.sample_stabilizer()
            outcome = self._measure_with_noise(pauli, phase, noise)
            outcomes.append(outcome)
            
        # Compute estimate
        outcomes = np.array(outcomes)
        fidelity = np.mean(outcomes)
        variance = np.var(outcomes) / M
        
        z = 1.96
        margin = z * np.sqrt(variance)
        ci = (max(0, fidelity - margin), min(1, fidelity + margin))
        root_infidelity = np.sqrt(max(0, 1 - fidelity))
        
        return FidelityEstimate(
            fidelity=fidelity,
            variance=variance,
            confidence_interval=ci,
            n_measurements=M,
            root_infidelity=root_infidelity,
            tvd_bound=root_infidelity
        )
    
    def _measure_with_noise(self, pauli: np.ndarray, phase: int,
                            noise: NoiseParameters) -> float:
        """Simulate noisy measurement."""
        ideal = 1 if phase == 0 else -1
        
        if np.random.random() < noise.p3:
            return -ideal
        return ideal


class HybridVerificationProtocol:
    """
    Hybrid Physical-Logical Verification Protocol
    
    Combines fault-tolerant logical operations with physical
    rotations following Majidy et al. (2025) Section IV.
    
    The protocol:
    1. Prepare logical-physical Bell state
    2. Perform verification on logical qubits
    3. Apply physical rotations
    4. Measure and decode
    """
    
    def __init__(self, code_distance: int, n_logical: int):
        """
        Initialize hybrid protocol.
        
        Args:
            code_distance: Distance of the error-correcting code
            n_logical: Number of logical qubits
        """
        self.code_distance = code_distance
        self.n_logical = n_logical
        
    def prepare_mixed_bell_state(self) -> np.ndarray:
        """
        Prepare mixed logical-physical Bell state.
        
        |Φ⟩ = (1/√2)(|0̄0⟩ + |1̄1⟩)
        
        Returns:
            State vector (simplified representation)
        """
        # Simplified - actual implementation would use full Hilbert space
        return np.array([1, 0, 0, 1]) / np.sqrt(2)
    
    def logical_error_rate(self, physical_error: float) -> float:
        """
        Estimate logical error rate from physical error rate.
        
        For surface code: p_L ≈ (p/p_th)^((d+1)/2)
        
        Args:
            physical_error: Physical error rate
            
        Returns:
            Estimated logical error rate
        """
        threshold = 0.01  # Approximate surface code threshold
        if physical_error >= threshold:
            return 1.0
            
        return (physical_error / threshold) ** ((self.code_distance + 1) / 2)
    
    def verification_fidelity_bound(self, 
                                   noise: NoiseParameters) -> float:
        """
        Compute verification fidelity bound for hybrid scheme.
        
        Args:
            noise: Noise parameters
            
        Returns:
            Lower bound on verification fidelity
        """
        # Logical operations are protected
        logical_error = self.logical_error_rate(noise.p2)
        
        # Physical operations accumulate errors
        physical_error = noise.p1 + noise.p3
        
        # Total fidelity bound
        return (1 - logical_error) * (1 - physical_error) ** self.n_logical


class CrossEntropyBenchmark:
    """
    Cross-Entropy Benchmarking for comparison with DFE.
    
    Implements linear and logarithmic XEB fidelities as defined
    in Ringbauer et al. (2025) Eqs. (4-6).
    
    Note: XEB requires classical simulation and is thus exponentially
    costly, unlike DFE.
    """
    
    @staticmethod
    def linear_xeb(samples: np.ndarray, 
                   ideal_probs: np.ndarray) -> float:
        """
        Compute linear XEB fidelity.
        
        f_lin(Q,P) = 2^n Σ_x Q(x)P(x) - 1
        
        Args:
            samples: Array of sample bitstrings
            ideal_probs: Ideal output probabilities
            
        Returns:
            Linear XEB fidelity
        """
        n = int(np.log2(len(ideal_probs)))
        
        # Empirical distribution from samples
        empirical = np.zeros_like(ideal_probs)
        for sample in samples:
            idx = int(''.join(map(str, sample)), 2)
            empirical[idx] += 1
        empirical /= len(samples)
        
        return 2**n * np.sum(empirical * ideal_probs) - 1
    
    @staticmethod  
    def log_xeb(samples: np.ndarray,
                ideal_probs: np.ndarray) -> float:
        """
        Compute logarithmic XEB fidelity.
        
        f_log(Q,P) = Σ_x Q(x) log P(x) + n log 2 + γ
        
        where γ is Euler's constant.
        
        Args:
            samples: Array of sample bitstrings
            ideal_probs: Ideal output probabilities
            
        Returns:
            Logarithmic XEB fidelity
        """
        n = int(np.log2(len(ideal_probs)))
        euler_gamma = 0.5772156649
        
        log_probs = []
        for sample in samples:
            idx = int(''.join(map(str, sample)), 2)
            if ideal_probs[idx] > 0:
                log_probs.append(np.log(ideal_probs[idx]))
                
        if not log_probs:
            return 0.0
            
        return np.mean(log_probs) + n * np.log(2) + euler_gamma


if __name__ == "__main__":
    print("=" * 60)
    print("Direct Fidelity Estimation Demo")
    print("=" * 60)
    
    # Create test cluster state
    from stabilizer_tableau import StabilizerTableau
    
    tableau = StabilizerTableau.cluster_state(2, 3)
    
    # Standard DFE
    print("\n1. Standard Direct Fidelity Estimation:")
    dfe = DirectFidelityEstimator(tableau, NoiseParameters(p3=0.01))
    result = dfe.estimate_fidelity(n_measurements=1000)
    print(f"   {result}")
    
    # Adaptive estimation
    print("\n2. Adaptive Stabilizer Sampling:")
    adaptive = AdaptiveStabilizerSampler(tableau)
    regions = adaptive.identify_k_uniform_regions()
    print(f"   Found k-uniform regions: {list(regions.keys())}")
    
    result_adaptive = adaptive.adaptive_estimate(
        epsilon=0.05, 
        delta=0.05,
        noise=NoiseParameters(p3=0.01)
    )
    print(f"   {result_adaptive}")
    
    # Hybrid protocol
    print("\n3. Hybrid Verification Protocol:")
    hybrid = HybridVerificationProtocol(code_distance=3, n_logical=6)
    noise = NoiseParameters(p1=1e-4, p2=1e-3, p3=1e-3)
    bound = hybrid.verification_fidelity_bound(noise)
    print(f"   Fidelity bound: {bound:.4f}")
    
    # Sample complexity
    print("\n4. Sample Complexity Analysis:")
    for eps in [0.1, 0.05, 0.01]:
        M = dfe.required_measurements(eps, 0.05)
        print(f"   ε={eps}: M = {M} measurements")
