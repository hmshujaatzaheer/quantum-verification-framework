"""
Circuit Simulation and Visualization for Quantum Verification

Implements circuit construction for k-uniform state preparation
and visualization tools for analysis.

Based on circuits from:
- Majidy, Hangleiter & Gullans (2025), Figures 2-3

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class GateType(Enum):
    """Quantum gate types."""
    H = "H"
    X = "X"
    Y = "Y"
    Z = "Z"
    CNOT = "CNOT"
    CZ = "CZ"
    RZ = "RZ"
    MEASURE = "M"


@dataclass
class Gate:
    """Representation of a quantum gate."""
    gate_type: GateType
    qubits: Tuple[int, ...]
    params: Optional[Tuple[float, ...]] = None
    
    def __repr__(self):
        if self.params:
            return f"{self.gate_type.value}({self.params})@{self.qubits}"
        return f"{self.gate_type.value}@{self.qubits}"


class QuantumCircuit:
    """
    Simple quantum circuit representation for verification protocols.
    """
    
    def __init__(self, n_qubits: int, name: str = "circuit"):
        """
        Initialize quantum circuit.
        
        Args:
            n_qubits: Number of qubits
            name: Circuit name
        """
        self.n_qubits = n_qubits
        self.name = name
        self.gates: List[List[Gate]] = [[]]  # List of layers
        
    def h(self, qubit: int) -> 'QuantumCircuit':
        """Add Hadamard gate."""
        self.gates[-1].append(Gate(GateType.H, (qubit,)))
        return self
        
    def cnot(self, control: int, target: int) -> 'QuantumCircuit':
        """Add CNOT gate."""
        self.gates[-1].append(Gate(GateType.CNOT, (control, target)))
        return self
        
    def cz(self, q1: int, q2: int) -> 'QuantumCircuit':
        """Add CZ gate."""
        self.gates[-1].append(Gate(GateType.CZ, (q1, q2)))
        return self
        
    def rz(self, qubit: int, angle: float) -> 'QuantumCircuit':
        """Add RZ rotation."""
        self.gates[-1].append(Gate(GateType.RZ, (qubit,), (angle,)))
        return self
        
    def measure(self, qubit: int) -> 'QuantumCircuit':
        """Add measurement."""
        self.gates[-1].append(Gate(GateType.MEASURE, (qubit,)))
        return self
        
    def barrier(self) -> 'QuantumCircuit':
        """Add layer barrier."""
        self.gates.append([])
        return self
        
    @property
    def depth(self) -> int:
        """Circuit depth (number of layers)."""
        return len([layer for layer in self.gates if layer])
        
    @property
    def gate_count(self) -> int:
        """Total gate count."""
        return sum(len(layer) for layer in self.gates)
    
    def two_qubit_gate_count(self) -> int:
        """Count two-qubit gates."""
        count = 0
        for layer in self.gates:
            for gate in layer:
                if gate.gate_type in [GateType.CNOT, GateType.CZ]:
                    count += 1
        return count


def create_surface_code_circuit(n_qubits: int, k: int) -> QuantumCircuit:
    """
    Create k-uniform state preparation circuit for surface code.
    
    Based on Figure 2 of Majidy, Hangleiter & Gullans (2025).
    
    Args:
        n_qubits: Number of qubits
        k: Target k-uniformity (1-4)
        
    Returns:
        QuantumCircuit for state preparation
    """
    if n_qubits < 2 * k:
        raise ValueError(f"Need at least {2*k} qubits for k={k}")
        
    circuit = QuantumCircuit(n_qubits, f"surface_k{k}")
    
    # Apply Hadamard to all qubits
    for i in range(n_qubits):
        circuit.h(i)
    circuit.barrier()
    
    if k == 1:
        # Simple entangling layer
        for i in range(0, n_qubits - 1, 2):
            circuit.cnot(i, i + 1)
        circuit.barrier()
        for i in range(1, n_qubits - 1, 2):
            circuit.cnot(i, i + 1)
            
    elif k == 2:
        # Four CNOT layers
        for layer in range(4):
            if layer % 2 == 0:
                for i in range(0, n_qubits - 1, 2):
                    circuit.cnot(i, i + 1)
            else:
                for i in range(1, n_qubits - 1, 2):
                    circuit.cnot(i + 1, i)  # Reversed direction
            circuit.barrier()
            
    elif k == 3:
        # Six CNOT layers
        for layer in range(6):
            if layer in [1, 2, 3]:
                for i in range(0, n_qubits - 1, 2):
                    circuit.cnot(i, i + 1)
            else:
                for i in range(1, n_qubits - 1, 2):
                    circuit.cnot(i, i + 1)
            circuit.barrier()
            
    elif k == 4:
        # Twelve CNOT layers
        down_layers = [1, 2, 3, 6, 7]
        for layer in range(12):
            if layer in down_layers:
                for i in range(1, n_qubits - 1, 2):
                    circuit.cnot(i + 1, i)
            else:
                for i in range(0, n_qubits - 1, 2):
                    circuit.cnot(i, i + 1)
            circuit.barrier()
    
    return circuit


def create_color_code_circuit(n_qubits: int, k: int) -> QuantumCircuit:
    """
    Create k-uniform state preparation circuit for [4,2,2] color code.
    
    Based on Figure 3 of Majidy, Hangleiter & Gullans (2025).
    
    Args:
        n_qubits: Number of qubits (must be even)
        k: Target k-uniformity (1-4)
        
    Returns:
        QuantumCircuit for state preparation
    """
    if n_qubits % 2 != 0:
        raise ValueError("Color code requires even number of qubits")
    if n_qubits < 2 * k:
        raise ValueError(f"Need at least {2*k} qubits for k={k}")
        
    circuit = QuantumCircuit(n_qubits, f"color_k{k}")
    
    # Initialize in |+⟩ state
    for i in range(n_qubits):
        circuit.h(i)
    circuit.barrier()
    
    if k == 1:
        # Intra-block CZ gates
        for i in range(0, n_qubits - 1, 2):
            circuit.cz(i, i + 1)
        circuit.barrier()
        # Inter-block CNOTs (upward)
        for i in range(0, n_qubits - 2, 2):
            circuit.cnot(i, i + 2)
            circuit.cnot(i + 1, i + 3)
            
    elif k == 2:
        # k=1 circuit + additional downward CNOTs
        for i in range(0, n_qubits - 1, 2):
            circuit.cz(i, i + 1)
        circuit.barrier()
        for i in range(0, n_qubits - 2, 2):
            circuit.cnot(i, i + 2)
            circuit.cnot(i + 1, i + 3)
        circuit.barrier()
        # Downward layer
        for i in range(2, n_qubits, 2):
            if i + 1 < n_qubits:
                circuit.cnot(i + 1, i - 1)
                
    elif k >= 3:
        # Multiple repetitions of k=1 pattern
        reps = 2 if k == 3 else 3
        for _ in range(reps):
            for i in range(0, n_qubits - 1, 2):
                circuit.cz(i, i + 1)
            circuit.barrier()
            for i in range(0, n_qubits - 2, 2):
                circuit.cnot(i, i + 2)
                if i + 3 < n_qubits:
                    circuit.cnot(i + 1, i + 3)
            circuit.barrier()
    
    return circuit


def create_cluster_state_circuit(rows: int, cols: int,
                                 beta: Optional[np.ndarray] = None) -> QuantumCircuit:
    """
    Create circuit for 2D cluster state with local rotations.
    
    Based on Ringbauer et al. (2025) MBQC protocol.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        beta: Rotation angles (default: zeros)
        
    Returns:
        QuantumCircuit for cluster state preparation
    """
    n_qubits = rows * cols
    circuit = QuantumCircuit(n_qubits, f"cluster_{rows}x{cols}")
    
    if beta is None:
        beta = np.zeros(n_qubits)
    
    def idx(r, c):
        return r * cols + c
    
    # Initialize in |+⟩
    for i in range(n_qubits):
        circuit.h(i)
    circuit.barrier()
    
    # Apply Z rotations
    for i in range(n_qubits):
        if beta[i] != 0:
            circuit.rz(i, beta[i])
    circuit.barrier()
    
    # Apply CZ gates for cluster state
    # Horizontal bonds
    for r in range(rows):
        for c in range(cols - 1):
            circuit.cz(idx(r, c), idx(r, c + 1))
    circuit.barrier()
    
    # Vertical bonds
    for r in range(rows - 1):
        for c in range(cols):
            circuit.cz(idx(r, c), idx(r + 1, c))
    
    return circuit


class VerificationVisualizer:
    """
    Visualization tools for quantum verification analysis.
    """
    
    @staticmethod
    def plot_fidelity_vs_noise(noise_levels: np.ndarray,
                               fidelities: np.ndarray,
                               labels: List[str],
                               title: str = "Fidelity vs Noise"):
        """
        Plot fidelity estimates against noise level.
        
        Args:
            noise_levels: Array of noise parameter values
            fidelities: Array of fidelity values (can be 2D for multiple methods)
            labels: Labels for each method
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#0066CC', '#CC3333', '#339933', '#FF9900']
        
        if fidelities.ndim == 1:
            fidelities = fidelities.reshape(1, -1)
            
        for i, (fid, label) in enumerate(zip(fidelities, labels)):
            ax.plot(noise_levels, fid, 'o-', color=colors[i % len(colors)],
                   label=label, linewidth=2, markersize=8)
            
        ax.set_xlabel('Noise Parameter (p)', fontsize=12)
        ax.set_ylabel('Fidelity', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        return fig, ax
    
    @staticmethod
    def plot_k_uniformity_analysis(n_qubits_range: List[int],
                                   k_values: np.ndarray,
                                   delta_values: np.ndarray):
        """
        Plot k-uniformity analysis across system sizes.
        
        Args:
            n_qubits_range: List of qubit numbers
            k_values: Array of k values tested
            delta_values: 2D array of Δ values [n_qubits, k]
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cmap = plt.cm.viridis
        
        for i, n in enumerate(n_qubits_range):
            color = cmap(i / len(n_qubits_range))
            ax.plot(k_values, delta_values[i], 'o-', color=color,
                   label=f'N={n}', linewidth=2, markersize=8)
            
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Exact k-uniform')
        ax.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Δ=1 threshold')
        
        ax.set_xlabel('k', fontsize=12)
        ax.set_ylabel('Δ (approximation parameter)', fontsize=12)
        ax.set_title('k-Uniformity Analysis', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_sample_complexity(k_range: np.ndarray,
                               dfe_samples: np.ndarray,
                               xeb_samples: np.ndarray,
                               n_qubits: int):
        """
        Compare sample complexity of DFE vs XEB.
        
        Args:
            k_range: Range of k values
            dfe_samples: DFE sample requirements
            xeb_samples: XEB sample requirements (exponential in n)
            n_qubits: Number of qubits
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.semilogy(k_range, dfe_samples, 'o-', color='#0066CC',
                   label='DFE (efficient)', linewidth=2, markersize=8)
        ax.semilogy(k_range, xeb_samples, 's--', color='#CC3333',
                   label='XEB (exponential)', linewidth=2, markersize=8)
        
        ax.set_xlabel('k-uniformity', fontsize=12)
        ax.set_ylabel('Sample Complexity', fontsize=12)
        ax.set_title(f'Sample Complexity Comparison (N={n_qubits})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        
        return fig, ax
    
    @staticmethod
    def plot_circuit(circuit: QuantumCircuit, filename: Optional[str] = None):
        """
        Create simple circuit visualization.
        
        Args:
            circuit: QuantumCircuit to visualize
            filename: Optional filename to save
        """
        fig, ax = plt.subplots(figsize=(max(12, circuit.depth * 1.5), 
                                        max(4, circuit.n_qubits * 0.5)))
        
        # Draw qubit lines
        for i in range(circuit.n_qubits):
            ax.axhline(y=i, color='black', linewidth=0.5)
            ax.text(-0.5, i, f'q{i}', ha='right', va='center', fontsize=10)
        
        # Draw gates
        x_pos = 0
        for layer in circuit.gates:
            if not layer:
                continue
            for gate in layer:
                if gate.gate_type == GateType.H:
                    q = gate.qubits[0]
                    ax.add_patch(plt.Rectangle((x_pos - 0.2, q - 0.3), 0.4, 0.6,
                                fill=True, facecolor='lightblue', edgecolor='black'))
                    ax.text(x_pos, q, 'H', ha='center', va='center', fontsize=8)
                    
                elif gate.gate_type == GateType.CNOT:
                    ctrl, tgt = gate.qubits
                    ax.plot([x_pos, x_pos], [ctrl, tgt], 'k-', linewidth=1)
                    ax.plot(x_pos, ctrl, 'ko', markersize=6)
                    ax.plot(x_pos, tgt, 'ko', markersize=10, fillstyle='none')
                    ax.plot(x_pos, tgt, 'k+', markersize=8)
                    
                elif gate.gate_type == GateType.CZ:
                    q1, q2 = gate.qubits
                    ax.plot([x_pos, x_pos], [q1, q2], 'k-', linewidth=1)
                    ax.plot(x_pos, q1, 'ko', markersize=6)
                    ax.plot(x_pos, q2, 'ko', markersize=6)
                    
                elif gate.gate_type == GateType.RZ:
                    q = gate.qubits[0]
                    ax.add_patch(plt.Rectangle((x_pos - 0.2, q - 0.3), 0.4, 0.6,
                                fill=True, facecolor='lightgreen', edgecolor='black'))
                    ax.text(x_pos, q, 'Rz', ha='center', va='center', fontsize=7)
                    
            x_pos += 1
            
        ax.set_xlim(-1, x_pos + 0.5)
        ax.set_ylim(-0.5, circuit.n_qubits - 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(circuit.name, fontsize=12)
        ax.invert_yaxis()
        
        if filename:
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            
        return fig, ax


if __name__ == "__main__":
    print("=" * 60)
    print("Circuit Construction Demo")
    print("=" * 60)
    
    # Create surface code circuits
    print("\nSurface Code Circuits:")
    for k in range(1, 5):
        n = max(4, 2 * k + 2)
        circ = create_surface_code_circuit(n, k)
        print(f"  k={k}: {circ.gate_count} gates, depth={circ.depth}, "
              f"2Q gates={circ.two_qubit_gate_count()}")
    
    # Create color code circuits
    print("\nColor Code Circuits:")
    for k in range(1, 5):
        n = max(6, 2 * k + 2)
        if n % 2 == 1:
            n += 1
        circ = create_color_code_circuit(n, k)
        print(f"  k={k}: {circ.gate_count} gates, depth={circ.depth}, "
              f"2Q gates={circ.two_qubit_gate_count()}")
    
    # Create cluster state circuit
    print("\nCluster State Circuit (3x3):")
    beta = np.random.choice([0, np.pi/4, np.pi/2, 3*np.pi/4], size=9)
    circ = create_cluster_state_circuit(3, 3, beta)
    print(f"  {circ.gate_count} gates, depth={circ.depth}")
