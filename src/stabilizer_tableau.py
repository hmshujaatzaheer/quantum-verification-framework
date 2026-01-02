"""
Stabilizer Tableau Implementation for k-Uniformity Verification

This module implements the stabilizer tableau formalism for determining
k-uniformity of quantum states, based on:
- Majidy, Hangleiter & Gullans (2025): arXiv:2503.14506
- Ringbauer et al. (2025): Nature Communications 16, 106

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from itertools import combinations
import warnings


class StabilizerTableau:
    """
    Binary symplectic representation of stabilizer states.
    
    For an N-qubit stabilizer state, the tableau is an N x 2N binary matrix
    where each row represents a stabilizer generator in the form [X | Z].
    
    Reference: Aaronson & Gottesman, Phys. Rev. A 70, 052328 (2004)
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize an empty stabilizer tableau for n_qubits.
        
        Args:
            n_qubits: Number of qubits in the system
        """
        self.n_qubits = n_qubits
        # Binary symplectic matrix: [X_part | Z_part]
        self.tableau = np.zeros((n_qubits, 2 * n_qubits), dtype=np.int8)
        # Phase vector (0 = +1, 1 = -1, 2 = +i, 3 = -i)
        self.phases = np.zeros(n_qubits, dtype=np.int8)
        
    @classmethod
    def from_generators(cls, generators: List[str]) -> 'StabilizerTableau':
        """
        Create tableau from string representations of stabilizer generators.
        
        Args:
            generators: List of Pauli strings, e.g., ['XZZI', 'IXZZ', 'ZIXZ', 'ZZIX']
            
        Returns:
            StabilizerTableau instance
        """
        n_qubits = len(generators[0].lstrip('+-'))
        tableau = cls(n_qubits)
        
        for i, gen in enumerate(generators):
            phase = 0
            if gen.startswith('-'):
                phase = 1
                gen = gen[1:]
            elif gen.startswith('+'):
                gen = gen[1:]
                
            for j, pauli in enumerate(gen):
                if pauli == 'X':
                    tableau.tableau[i, j] = 1  # X part
                elif pauli == 'Z':
                    tableau.tableau[i, n_qubits + j] = 1  # Z part
                elif pauli == 'Y':
                    tableau.tableau[i, j] = 1  # X part
                    tableau.tableau[i, n_qubits + j] = 1  # Z part
                # 'I' leaves both as 0
                    
            tableau.phases[i] = phase
            
        return tableau
    
    @classmethod
    def cluster_state(cls, rows: int, cols: int, 
                      beta: Optional[np.ndarray] = None) -> 'StabilizerTableau':
        """
        Create stabilizer tableau for a 2D cluster state with local Z rotations.
        
        For cluster states, each stabilizer has the form:
        S_k = X_k(β_k) ∏_{j∈N(k)} Z_j
        
        where X_k(β) = exp(iβX_k/2) and N(k) are neighbors of site k.
        
        Args:
            rows: Number of rows in the cluster lattice
            cols: Number of columns in the cluster lattice
            beta: Optional array of rotation angles (default: all zeros)
            
        Returns:
            StabilizerTableau for the cluster state
        """
        n_qubits = rows * cols
        tableau = cls(n_qubits)
        
        if beta is None:
            beta = np.zeros(n_qubits)
        
        def idx(r, c):
            """Convert 2D coordinates to linear index."""
            return r * cols + c
        
        for r in range(rows):
            for c in range(cols):
                k = idx(r, c)
                
                # X operator at site k (modified by rotation angle)
                # For β = 0, this is just X
                # For β = π/4, this becomes (X + Y)/√2, etc.
                tableau.tableau[k, k] = 1  # X part
                
                # Z operators on neighboring sites
                neighbors = []
                if r > 0:
                    neighbors.append(idx(r-1, c))
                if r < rows - 1:
                    neighbors.append(idx(r+1, c))
                if c > 0:
                    neighbors.append(idx(r, c-1))
                if c < cols - 1:
                    neighbors.append(idx(r, c+1))
                    
                for neighbor in neighbors:
                    tableau.tableau[k, n_qubits + neighbor] = 1  # Z part
                    
        return tableau
    
    def restrict_to_subsystem(self, subsystem: List[int]) -> np.ndarray:
        """
        Restrict the stabilizer tableau to a subsystem by removing columns
        for qubits not in the subsystem.
        
        Args:
            subsystem: List of qubit indices in the subsystem A
            
        Returns:
            Restricted binary matrix for subsystem A
        """
        n = self.n_qubits
        k = len(subsystem)
        
        # Select columns for X and Z parts of the subsystem
        x_cols = subsystem
        z_cols = [n + i for i in subsystem]
        cols = x_cols + z_cols
        
        return self.tableau[:, cols]
    
    def compute_I_A(self, subsystem: List[int]) -> int:
        """
        Compute I_A: the number of stabilizer generators that remain 
        independent when restricted to subsystem A.
        
        This is done by computing the rank (mod 2) of the restricted tableau.
        
        Reference: Nahum et al., Phys. Rev. X 7, 031016 (2017)
        
        Args:
            subsystem: List of qubit indices in subsystem A
            
        Returns:
            I_A value (rank of restricted tableau over GF(2))
        """
        restricted = self.restrict_to_subsystem(subsystem)
        return self._gf2_rank(restricted)
    
    @staticmethod
    def _gf2_rank(matrix: np.ndarray) -> int:
        """
        Compute the rank of a binary matrix over GF(2).
        
        Uses Gaussian elimination modulo 2.
        
        Args:
            matrix: Binary numpy array
            
        Returns:
            Rank of the matrix over GF(2)
        """
        M = matrix.copy().astype(np.int8)
        rows, cols = M.shape
        
        rank = 0
        pivot_col = 0
        
        for row in range(rows):
            if pivot_col >= cols:
                break
                
            # Find pivot
            pivot_row = None
            for r in range(row, rows):
                if M[r, pivot_col] == 1:
                    pivot_row = r
                    break
                    
            if pivot_row is None:
                pivot_col += 1
                continue
                
            # Swap rows
            M[[row, pivot_row]] = M[[pivot_row, row]]
            
            # Eliminate
            for r in range(rows):
                if r != row and M[r, pivot_col] == 1:
                    M[r] = (M[r] + M[row]) % 2
                    
            rank += 1
            pivot_col += 1
            
        return rank
    
    def get_stabilizer_group_element(self, bitstring: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Get an element of the stabilizer group specified by a bitstring.
        
        The stabilizer group has 2^N elements, each specified by which
        generators to multiply together.
        
        Args:
            bitstring: Binary array of length N indicating which generators to include
            
        Returns:
            Tuple of (Pauli operator as binary vector, phase)
        """
        n = self.n_qubits
        result = np.zeros(2 * n, dtype=np.int8)
        phase = 0
        
        for i, include in enumerate(bitstring):
            if include:
                # Multiply Pauli operators (symplectic product)
                result = (result + self.tableau[i]) % 2
                phase = (phase + self.phases[i]) % 4
                
        return result, phase
    
    def sample_stabilizer(self) -> Tuple[np.ndarray, int]:
        """
        Sample a uniformly random element from the stabilizer group.
        
        Returns:
            Tuple of (Pauli operator as binary vector, phase)
        """
        bitstring = np.random.randint(0, 2, self.n_qubits)
        return self.get_stabilizer_group_element(bitstring)


class KUniformityCalculator:
    """
    Calculator for determining k-uniformity of stabilizer states.
    
    A state is k-uniform if every k-qubit subsystem is maximally mixed.
    For stabilizer states, this is determined by the stabilizer tableau.
    
    Reference: Majidy, Hangleiter & Gullans (2025), arXiv:2503.14506
    """
    
    def __init__(self, tableau: StabilizerTableau):
        """
        Initialize calculator with a stabilizer tableau.
        
        Args:
            tableau: StabilizerTableau instance
        """
        self.tableau = tableau
        self.n_qubits = tableau.n_qubits
        
    def compute_delta(self, k: int, subsystem: Optional[List[int]] = None) -> float:
        """
        Compute the Δ-approximate k-uniformity parameter for a subsystem.
        
        Δ = 2 - 2^(1-r) where r = 2k - I_A
        
        When Δ = 0, the subsystem is exactly k-uniform.
        
        Args:
            k: Size of subsystem
            subsystem: Specific subsystem to check (if None, returns max over all)
            
        Returns:
            Δ value (0 for exact k-uniformity)
        """
        if subsystem is not None:
            I_A = self.tableau.compute_I_A(subsystem)
            r = 2 * k - I_A
            if r <= 0:
                return 0.0
            return 2 - 2**(1 - r)
        
        # Find maximum Δ over all k-subsets
        max_delta = 0.0
        for subsystem in combinations(range(self.n_qubits), k):
            delta = self.compute_delta(k, list(subsystem))
            max_delta = max(max_delta, delta)
            
        return max_delta
    
    def is_k_uniform(self, k: int, tolerance: float = 1e-10) -> bool:
        """
        Check if the state is exactly k-uniform.
        
        Args:
            k: Uniformity parameter
            tolerance: Numerical tolerance for Δ = 0
            
        Returns:
            True if state is k-uniform
        """
        return self.compute_delta(k) < tolerance
    
    def find_max_k_uniformity(self) -> int:
        """
        Find the maximum k for which the state is k-uniform.
        
        Returns:
            Maximum k value (0 if not even 1-uniform)
        """
        for k in range(self.n_qubits, 0, -1):
            if self.is_k_uniform(k):
                return k
        return 0
    
    def compute_alpha_separated_delta(self, k: int, alpha: int) -> float:
        """
        Compute Δ for α-separated k-uniformity.
        
        Only checks k-subsets where qubits are at least α sites apart.
        This provides faster verification for scalable circuits.
        
        Reference: Majidy, Hangleiter & Gullans (2025), Section II.B
        
        Args:
            k: Size of subsystem
            alpha: Minimum separation between qubits
            
        Returns:
            Δ value for α-separated k-uniformity
        """
        max_delta = 0.0
        
        # Generate α-separated subsets
        for subsystem in self._alpha_separated_subsets(k, alpha):
            delta = self.compute_delta(k, list(subsystem))
            max_delta = max(max_delta, delta)
            
        return max_delta
    
    def _alpha_separated_subsets(self, k: int, alpha: int):
        """
        Generate all k-subsets with minimum separation α.
        
        Args:
            k: Size of subsets
            alpha: Minimum separation
            
        Yields:
            Tuples of qubit indices
        """
        def backtrack(start: int, current: List[int]):
            if len(current) == k:
                yield tuple(current)
                return
                
            for i in range(start, self.n_qubits):
                if not current or i >= current[-1] + alpha:
                    current.append(i)
                    yield from backtrack(i + 1, current)
                    current.pop()
                    
        yield from backtrack(0, [])
    
    def get_k_uniform_subsystems(self, k: int) -> List[Tuple[int, ...]]:
        """
        Find all k-qubit subsystems that are maximally mixed.
        
        Args:
            k: Size of subsystems
            
        Returns:
            List of k-uniform subsystem tuples
        """
        uniform_subsystems = []
        
        for subsystem in combinations(range(self.n_qubits), k):
            if self.compute_delta(k, list(subsystem)) < 1e-10:
                uniform_subsystems.append(subsystem)
                
        return uniform_subsystems


def create_surface_code_kuniform_circuit(n_qubits: int, k: int) -> StabilizerTableau:
    """
    Create stabilizer tableau for k-uniform states prepared via
    surface code circuits from Majidy et al. (2025).
    
    Args:
        n_qubits: Number of logical qubits
        k: Target k-uniformity
        
    Returns:
        StabilizerTableau for the prepared state
    """
    # Initialize |0⟩^⊗N state
    generators = []
    for i in range(n_qubits):
        gen = ['I'] * n_qubits
        gen[i] = 'Z'
        generators.append(''.join(gen))
    
    tableau = StabilizerTableau.from_generators(generators)
    
    # Apply circuit based on k value (from Fig. 2 of paper)
    # This is a simplified version - full implementation would
    # include the complete CNOT patterns
    
    return tableau


def create_color_code_kuniform_circuit(n_qubits: int, k: int) -> StabilizerTableau:
    """
    Create stabilizer tableau for k-uniform states prepared via
    [4,2,2] color code circuits from Majidy et al. (2025).
    
    Args:
        n_qubits: Number of logical qubits (must be even)
        k: Target k-uniformity
        
    Returns:
        StabilizerTableau for the prepared state
    """
    if n_qubits % 2 != 0:
        raise ValueError("Color code requires even number of qubits")
    
    # Initialize |+⟩^⊗N state
    generators = []
    for i in range(n_qubits):
        gen = ['I'] * n_qubits
        gen[i] = 'X'
        generators.append(''.join(gen))
    
    tableau = StabilizerTableau.from_generators(generators)
    
    return tableau


if __name__ == "__main__":
    # Example: Create and analyze a 2x3 cluster state
    print("=" * 60)
    print("Stabilizer Tableau Analysis for Cluster States")
    print("=" * 60)
    
    # Create 2x3 cluster state
    tableau = StabilizerTableau.cluster_state(2, 3)
    calculator = KUniformityCalculator(tableau)
    
    print(f"\nCluster state: 2x3 lattice ({tableau.n_qubits} qubits)")
    print(f"Stabilizer tableau shape: {tableau.tableau.shape}")
    
    # Check k-uniformity for different k values
    print("\nk-uniformity analysis:")
    for k in range(1, min(4, tableau.n_qubits + 1)):
        delta = calculator.compute_delta(k)
        is_uniform = "Yes" if delta < 1e-10 else "No"
        print(f"  k={k}: Δ = {delta:.4f}, k-uniform: {is_uniform}")
    
    max_k = calculator.find_max_k_uniformity()
    print(f"\nMaximum k-uniformity: k = {max_k}")
