"""
Unit Tests for Unified Quantum Verification Framework

Tests the core functionality of stabilizer tableau operations,
k-uniformity calculations, and fidelity estimation.

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
"""

import sys
sys.path.insert(0, '../src')

import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from stabilizer_tableau import StabilizerTableau, KUniformityCalculator
from fidelity_estimation import (
    DirectFidelityEstimator,
    AdaptiveStabilizerSampler,
    NoiseParameters,
    HybridVerificationProtocol
)
from circuits import (
    QuantumCircuit,
    create_cluster_state_circuit,
    create_surface_code_circuit
)


class TestStabilizerTableau(unittest.TestCase):
    """Tests for StabilizerTableau class."""
    
    def test_initialization(self):
        """Test tableau initialization."""
        tableau = StabilizerTableau(4)
        self.assertEqual(tableau.n_qubits, 4)
        self.assertEqual(tableau.tableau.shape, (4, 8))
        
    def test_from_generators_bell_state(self):
        """Test creation from stabilizer generators for Bell state."""
        # Bell state |Φ+⟩ has stabilizers XX and ZZ
        generators = ['XX', 'ZZ']
        tableau = StabilizerTableau.from_generators(generators)
        
        self.assertEqual(tableau.n_qubits, 2)
        # Check X part: XX should have X on both qubits
        assert_array_equal(tableau.tableau[0, :2], [1, 1])
        # Check Z part: ZZ should have Z on both qubits  
        assert_array_equal(tableau.tableau[1, 2:], [1, 1])
        
    def test_cluster_state_2x2(self):
        """Test 2x2 cluster state creation."""
        tableau = StabilizerTableau.cluster_state(2, 2)
        self.assertEqual(tableau.n_qubits, 4)
        
    def test_gf2_rank(self):
        """Test GF(2) rank computation."""
        # Full rank matrix
        M1 = np.array([[1, 0], [0, 1]], dtype=np.int8)
        self.assertEqual(StabilizerTableau._gf2_rank(M1), 2)
        
        # Rank-deficient matrix
        M2 = np.array([[1, 1], [1, 1]], dtype=np.int8)
        self.assertEqual(StabilizerTableau._gf2_rank(M2), 1)
        
        # Zero matrix
        M3 = np.zeros((2, 2), dtype=np.int8)
        self.assertEqual(StabilizerTableau._gf2_rank(M3), 0)
        
    def test_restrict_to_subsystem(self):
        """Test subsystem restriction."""
        tableau = StabilizerTableau.cluster_state(2, 2)
        restricted = tableau.restrict_to_subsystem([0, 1])
        
        # Should have 4 columns (2 X + 2 Z for 2 qubits)
        self.assertEqual(restricted.shape[1], 4)
        
    def test_sample_stabilizer(self):
        """Test random stabilizer sampling."""
        tableau = StabilizerTableau.cluster_state(2, 2)
        
        # Sample should return valid Pauli and phase
        pauli, phase = tableau.sample_stabilizer()
        
        self.assertEqual(len(pauli), 8)  # 2*n_qubits
        self.assertIn(phase, [0, 1, 2, 3])


class TestKUniformityCalculator(unittest.TestCase):
    """Tests for k-uniformity calculations."""
    
    def test_bell_state_1uniform(self):
        """Test that Bell state is 1-uniform."""
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2 is 1-uniform
        generators = ['XX', 'ZZ']
        tableau = StabilizerTableau.from_generators(generators)
        calc = KUniformityCalculator(tableau)
        
        # Should be 1-uniform
        self.assertTrue(calc.is_k_uniform(1))
        
    def test_ghz_state(self):
        """Test GHZ state k-uniformity."""
        # 3-qubit GHZ has stabilizers XXX, ZZI, IZZ
        generators = ['XXX', 'ZZI', 'IZZ']
        tableau = StabilizerTableau.from_generators(generators)
        calc = KUniformityCalculator(tableau)
        
        # GHZ is 1-uniform but not 2-uniform
        self.assertTrue(calc.is_k_uniform(1))
        
    def test_compute_delta(self):
        """Test Δ computation."""
        tableau = StabilizerTableau.cluster_state(2, 2)
        calc = KUniformityCalculator(tableau)
        
        # Δ should be in [0, 2]
        for k in range(1, 3):
            delta = calc.compute_delta(k)
            self.assertGreaterEqual(delta, 0)
            self.assertLessEqual(delta, 2)
            
    def test_alpha_separated(self):
        """Test α-separated k-uniformity."""
        tableau = StabilizerTableau.cluster_state(3, 3)
        calc = KUniformityCalculator(tableau)
        
        # α-separated should be computable
        delta = calc.compute_alpha_separated_delta(k=2, alpha=2)
        self.assertGreaterEqual(delta, 0)


class TestDirectFidelityEstimator(unittest.TestCase):
    """Tests for DFE."""
    
    def test_noiseless_estimation(self):
        """Test DFE on noiseless ideal state."""
        tableau = StabilizerTableau.cluster_state(2, 2)
        dfe = DirectFidelityEstimator(tableau, NoiseParameters(p3=0))
        
        result = dfe.estimate_fidelity(n_measurements=100)
        
        # Noiseless should give F ≈ 1
        self.assertGreater(result.fidelity, 0.95)
        
    def test_noisy_estimation(self):
        """Test DFE with measurement noise."""
        tableau = StabilizerTableau.cluster_state(2, 2)
        noise = NoiseParameters(p3=0.1)
        dfe = DirectFidelityEstimator(tableau, noise)
        
        result = dfe.estimate_fidelity(n_measurements=100)
        
        # Noisy should reduce fidelity
        self.assertLess(result.fidelity, 1.0)
        
    def test_required_measurements(self):
        """Test sample complexity calculation."""
        tableau = StabilizerTableau.cluster_state(2, 2)
        dfe = DirectFidelityEstimator(tableau)
        
        M = dfe.required_measurements(epsilon=0.1, delta=0.05)
        
        # Should require reasonable number of measurements
        self.assertGreater(M, 100)
        self.assertLess(M, 10000)
        
    def test_tvd_bound(self):
        """Test TVD bound from root infidelity."""
        tableau = StabilizerTableau.cluster_state(2, 2)
        dfe = DirectFidelityEstimator(tableau)
        
        result = dfe.estimate_fidelity(n_measurements=100)
        
        # TVD bound should equal root infidelity
        self.assertAlmostEqual(result.tvd_bound, result.root_infidelity)


class TestAdaptiveSampler(unittest.TestCase):
    """Tests for adaptive stabilizer sampling."""
    
    def test_region_identification(self):
        """Test k-uniform region identification."""
        tableau = StabilizerTableau.cluster_state(3, 3)
        sampler = AdaptiveStabilizerSampler(tableau)
        
        regions = sampler.identify_k_uniform_regions(k_max=2)
        
        # Should find some regions
        self.assertIsInstance(regions, dict)
        
    def test_adaptive_estimate(self):
        """Test adaptive estimation."""
        tableau = StabilizerTableau.cluster_state(2, 3)
        sampler = AdaptiveStabilizerSampler(tableau)
        
        result = sampler.adaptive_estimate(
            epsilon=0.1,
            delta=0.1,
            noise=NoiseParameters(p3=0.01)
        )
        
        # Should return valid estimate
        self.assertGreater(result.fidelity, 0)
        self.assertLessEqual(result.fidelity, 1)


class TestHybridProtocol(unittest.TestCase):
    """Tests for hybrid verification protocol."""
    
    def test_logical_error_rate(self):
        """Test logical error rate calculation."""
        hybrid = HybridVerificationProtocol(code_distance=3, n_logical=1)
        
        # Below threshold should suppress errors
        p_log = hybrid.logical_error_rate(0.001)
        self.assertLess(p_log, 0.001)
        
    def test_fidelity_bound(self):
        """Test fidelity bound calculation."""
        hybrid = HybridVerificationProtocol(code_distance=5, n_logical=5)
        noise = NoiseParameters(p1=1e-4, p2=1e-3, p3=1e-3)
        
        bound = hybrid.verification_fidelity_bound(noise)
        
        self.assertGreater(bound, 0)
        self.assertLessEqual(bound, 1)


class TestCircuits(unittest.TestCase):
    """Tests for circuit construction."""
    
    def test_circuit_creation(self):
        """Test basic circuit creation."""
        circ = QuantumCircuit(4, "test")
        circ.h(0).cnot(0, 1).barrier().cz(2, 3)
        
        self.assertEqual(circ.n_qubits, 4)
        self.assertGreater(circ.gate_count, 0)
        
    def test_cluster_circuit(self):
        """Test cluster state circuit creation."""
        circ = create_cluster_state_circuit(2, 3)
        
        self.assertEqual(circ.n_qubits, 6)
        self.assertGreater(circ.depth, 0)
        
    def test_surface_code_circuit(self):
        """Test surface code circuit creation."""
        for k in range(1, 4):
            circ = create_surface_code_circuit(8, k)
            self.assertEqual(circ.n_qubits, 8)
            self.assertGreater(circ.two_qubit_gate_count(), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def test_full_verification_pipeline(self):
        """Test complete verification workflow."""
        # Create state
        tableau = StabilizerTableau.cluster_state(2, 3)
        
        # Analyze k-uniformity
        calc = KUniformityCalculator(tableau)
        max_k = calc.find_max_k_uniformity()
        
        # Perform DFE
        dfe = DirectFidelityEstimator(tableau)
        result = dfe.estimate_fidelity(n_measurements=100)
        
        # All steps should complete without error
        self.assertIsInstance(max_k, int)
        self.assertIsInstance(result.fidelity, float)
        
    def test_adaptive_vs_standard(self):
        """Compare adaptive and standard estimation."""
        tableau = StabilizerTableau.cluster_state(3, 3)
        noise = NoiseParameters(p3=0.01)
        
        # Standard
        dfe = DirectFidelityEstimator(tableau, noise)
        standard_result = dfe.estimate_fidelity(n_measurements=200)
        
        # Adaptive
        adaptive = AdaptiveStabilizerSampler(tableau)
        adaptive_result = adaptive.adaptive_estimate(
            epsilon=0.1, delta=0.1, noise=noise
        )
        
        # Both should give reasonable estimates
        self.assertGreater(standard_result.fidelity, 0.5)
        self.assertGreater(adaptive_result.fidelity, 0.5)


def run_tests():
    """Run all tests with verbose output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestStabilizerTableau))
    suite.addTests(loader.loadTestsFromTestCase(TestKUniformityCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestDirectFidelityEstimator))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveSampler))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridProtocol))
    suite.addTests(loader.loadTestsFromTestCase(TestCircuits))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    run_tests()
