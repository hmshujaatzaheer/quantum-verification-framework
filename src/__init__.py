"""
Unified Quantum Verification Framework

A Python implementation for fault-tolerant verification of quantum random 
sampling, bridging k-uniform states and measurement-based computation.

Based on:
- Ringbauer et al. (2025): Nature Communications 16, 106
- Majidy, Hangleiter & Gullans (2025): arXiv:2503.14506

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
Repository: https://github.com/hmshujaatzaheer/quantum-verification-framework
"""

__version__ = "0.1.0"
__author__ = "H M Shujaat Zaheer"
__email__ = "shujabis@gmail.com"

from .stabilizer_tableau import (
    StabilizerTableau,
    KUniformityCalculator,
    create_surface_code_kuniform_circuit,
    create_color_code_kuniform_circuit
)

from .fidelity_estimation import (
    DirectFidelityEstimator,
    AdaptiveStabilizerSampler,
    HybridVerificationProtocol,
    CrossEntropyBenchmark,
    NoiseParameters,
    FidelityEstimate,
    NoiseModel
)

from .circuits import (
    QuantumCircuit,
    Gate,
    GateType,
    create_surface_code_circuit,
    create_color_code_circuit,
    create_cluster_state_circuit,
    VerificationVisualizer
)

__all__ = [
    # Stabilizer Tableau
    'StabilizerTableau',
    'KUniformityCalculator',
    'create_surface_code_kuniform_circuit',
    'create_color_code_kuniform_circuit',
    
    # Fidelity Estimation
    'DirectFidelityEstimator',
    'AdaptiveStabilizerSampler',
    'HybridVerificationProtocol',
    'CrossEntropyBenchmark',
    'NoiseParameters',
    'FidelityEstimate',
    'NoiseModel',
    
    # Circuits
    'QuantumCircuit',
    'Gate',
    'GateType',
    'create_surface_code_circuit',
    'create_color_code_circuit',
    'create_cluster_state_circuit',
    'VerificationVisualizer'
]
