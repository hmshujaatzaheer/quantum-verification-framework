# Unified Quantum Verification Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2503.14506-b31b1b.svg)](https://arxiv.org/abs/2503.14506)

**A Python implementation for fault-tolerant verification of quantum random sampling, bridging k-uniform states and measurement-based quantum computation.**

This repository implements the algorithms and protocols described in the PhD research proposal: *"Unified Verification Framework for Fault-Tolerant Quantum Random Sampling: Bridging k-Uniform States and Measurement-Based Computation"*.

## 📚 Based On

1. **Ringbauer et al. (2025)**: "Verifiable measurement-based quantum random sampling with trapped ions" - *Nature Communications* 16, 106
2. **Majidy, Hangleiter & Gullans (2025)**: "Scalable and fault-tolerant preparation of encoded k-uniform states" - *arXiv:2503.14506*

## 🎯 Key Features

- **Stabilizer Tableau Operations**: Binary symplectic representation for efficient stabilizer state manipulation
- **k-Uniformity Calculator**: Determine k-uniformity of quantum states via stabilizer tableau analysis
- **Direct Fidelity Estimation (DFE)**: Efficient verification through random stabilizer measurements
- **Adaptive Stabilizer Sampling**: Novel algorithm exploiting k-uniform structure for improved verification (Algorithm 1 from proposal)
- **Hybrid Physical-Logical Protocol**: Verification scheme combining fault-tolerant and physical operations
- **Circuit Construction**: Generate circuits for surface code, color code, and cluster state preparation

## 🏗️ Architecture

```
quantum-verification-framework/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── stabilizer_tableau.py    # Stabilizer formalism implementation
│   ├── fidelity_estimation.py   # DFE and adaptive verification
│   └── circuits.py              # Circuit construction and visualization
├── examples/
│   └── verification_demo.py     # Complete workflow examples
├── tests/
│   └── test_verification.py     # Unit tests
├── docs/
│   └── research_proposal.pdf    # Full research proposal
└── figures/
    └── ...                      # Generated figures
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/hmshujaatzaheer/quantum-verification-framework.git
cd quantum-verification-framework
pip install -r requirements.txt
```

### Basic Usage

```python
from src import (
    StabilizerTableau, 
    KUniformityCalculator,
    DirectFidelityEstimator,
    AdaptiveStabilizerSampler,
    NoiseParameters
)

# Create a 3x3 cluster state
tableau = StabilizerTableau.cluster_state(3, 3)

# Analyze k-uniformity
calculator = KUniformityCalculator(tableau)
for k in range(1, 4):
    delta = calculator.compute_delta(k)
    print(f"k={k}: Δ = {delta:.4f}")

# Perform Direct Fidelity Estimation
noise = NoiseParameters(p3=0.01)  # 1% measurement error
dfe = DirectFidelityEstimator(tableau, noise)
result = dfe.estimate_fidelity(n_measurements=500)
print(f"Fidelity: {result.fidelity:.4f} ± {result.variance**0.5:.4f}")
print(f"TVD bound: {result.tvd_bound:.4f}")

# Use Adaptive Stabilizer Sampling (Algorithm 1)
adaptive = AdaptiveStabilizerSampler(tableau)
regions = adaptive.identify_k_uniform_regions()
result = adaptive.adaptive_estimate(epsilon=0.05, delta=0.05, noise=noise)
print(f"Adaptive estimate: {result.fidelity:.4f}")
```

## 📖 Core Algorithms

### Algorithm 1: Adaptive Stabilizer Sampling

The main algorithmic contribution exploits k-uniform structure for more efficient verification:

```
Input: Cluster state |ψ_β⟩, accuracy ε, k-uniformity bound k*
Output: Fidelity estimate F̂

1. Compute stabilizer tableau T for |ψ_β⟩
2. Identify k-uniform subsystems via Δ-criterion
3. S_priority ← stabilizers supported on k-uniform regions
4. M ← ⌈4ε⁻²log(2/δ)⌉
5. For i = 1 to M:
   6.   If i ≤ M/2: sample s_i from S_priority
   7.   Else: sample s_i uniformly from S
   8.   Measure s_i, obtain σ_i ∈ {+1, -1}
9. Return F̂ = (1/M)Σσ_i
```

### k-Uniformity Verification

Based on stabilizer tableau analysis:

$$\Delta = 2 - 2^{1-r}, \quad r = 2k - I_A$$

where $I_A$ is the number of independent stabilizers restricted to subsystem $A$.

### Fidelity-TVD Bound

From Ringbauer et al. (2025):

$$d_{\text{TV}} \leq d_{\text{Tr}} \leq \sqrt{1-F}$$

## 🧪 Running Tests

```bash
cd tests
python -m pytest test_verification.py -v
```

Or run directly:

```bash
python tests/test_verification.py
```

## 📊 Examples

Run the complete demonstration:

```bash
python examples/verification_demo.py
```

This runs six examples:
1. Basic Direct Fidelity Estimation
2. Adaptive Stabilizer Sampling
3. Hybrid Physical-Logical Protocol
4. Noise Model Analysis
5. Sample Complexity Comparison
6. Full Benchmarking Suite

## 📈 Sample Output

```
Example 1: Basic Direct Fidelity Estimation
=====================================================
Created 3x3 cluster state (9 qubits)

k-Uniformity Analysis:
  k=1: ✓ k-uniform
  k=2: Δ = 0.0000
  k=3: Δ = 1.0000

Direct Fidelity Estimation:
  p_meas=0.000: F=1.0000 ± 0.0000, TVD≤0.0000
  p_meas=0.010: F=0.9780 ± 0.0094, TVD≤0.1483
```

## 🔬 Research Context

This implementation supports research addressing three foundational questions:

1. **Quantum-Classical Boundaries**: What properties of k-uniform cluster states determine classical simulability?

2. **Guiding Principles**: Can k-uniformity degree serve as a sufficient condition for quantum advantage verification?

3. **Device Constraints**: How does the hybrid scheme affect verification fidelity under realistic noise?

## 📄 Citation

If you use this code, please cite:

```bibtex
@article{ringbauer2025verifiable,
  title={Verifiable measurement-based quantum random sampling with trapped ions},
  author={Ringbauer, Martin and Hinsche, Marcel and Feldker, Thomas and others},
  journal={Nature Communications},
  volume={16},
  pages={106},
  year={2025}
}

@article{majidy2025scalable,
  title={Scalable and fault-tolerant preparation of encoded k-uniform states},
  author={Majidy, Shayan and Hangleiter, Dominik and Gullans, Michael J},
  journal={arXiv preprint arXiv:2503.14506},
  year={2025}
}
```

## 👤 Author

**H M Shujaat Zaheer**
- Email: shujabis@gmail.com
- GitHub: [@hmshujaatzaheer](https://github.com/hmshujaatzaheer)
- Education: MSc Computer Science (AI/ML), University of Sialkot

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Prof. Dominik Hangleiter (ETH Zurich) for research direction and foundational papers
- The quantum computing research community for open-source tools and collaboration

---

*This implementation is part of a PhD research proposal submitted to ETH Zurich.*
