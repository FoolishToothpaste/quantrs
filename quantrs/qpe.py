"""
qpe.py
------
Quantum Phase Estimation (QPE) implemented from scratch using quantrs.

What is QPE?
------------
Quantum Phase Estimation solves this problem:

    Given a unitary gate U and one of its eigenstates |psi> such that:
        U|psi> = e^(2*pi*i*phi) * |psi>

    Estimate the phase phi to n bits of precision.

In other words: if you have a gate U and a state it acts on without
changing the state (only adding a phase), QPE finds that phase.

This is fundamental because:
    - Many quantum algorithms reduce to finding eigenvalues/phases
    - It is a core subroutine inside Shor's Algorithm
    - It is used in quantum chemistry to find ground state energies
    - It underlies quantum linear systems solvers (HHL algorithm)

How the circuit works:
----------------------
The circuit has two registers:

    Phase register  (n qubits): precision — more qubits = more accurate phase
    Eigenstate register (m qubits): holds the state |psi>

Structure:

    Phase register:
        q[0]:  ─H─────────────────────────C^(2^(n-1))U──IQFT──M─
        q[1]:  ─H──────────────────C^(2^(n-2))U──────────────M─
        ...
        q[n-1]:─H──C^(2^0)U───────────────────────────────M─

    Eigenstate register:
        q[n]:  ─────●──────●──────────────────────────────────
        ...

Step by step:
    1. Apply H to all phase register qubits to create uniform superposition
    2. Apply controlled-U^(2^k) for each phase qubit k, where the phase
       qubit acts as the control and the eigenstate register is the target
    3. Apply the inverse QFT to the phase register
    4. Measure the phase register

Why it works:
    After step 2, the phase register encodes e^(2*pi*i*phi*k) in qubit k.
    This is exactly the structure that IQFT decodes back into the binary
    representation of phi.

Usage:
------
    from quantrs.qpe import QPE

    # Estimate the phase of the T gate (phase = 1/8 = 0.125)
    # T|1> = e^(i*pi/4)|1> = e^(2*pi*i * 1/8)|1>  =>  phi = 1/8
    result = QPE.t_gate_demo(precision=4)
    result.print_phase()

    # Estimate the phase of a custom unitary
    from quantrs.linalg import Matrix
    import cmath, math
    phi = 0.3
    U = Matrix([[1,0],[0,cmath.exp(2j*math.pi*phi)]])
    result = QPE.run(U=U, eigenstate=[0,1], precision=4)
    result.print_phase()
"""

from __future__ import annotations
import math
import cmath
from typing import List, Optional, Tuple

from .circuit import Circuit
from .linalg import Matrix, Vector
from .state import QuantumState
from .qft import QFT, _controlled_phase
from . import gates as G


class QPEResult:
    """
    Result of a Quantum Phase Estimation run.

    Attributes
    ----------
    bitstring : str
        The measured bitstring from the phase register.
    phase_estimate : float
        The estimated phase phi in [0, 1).
    precision : int
        Number of qubits used for the phase register.
    true_phase : float, optional
        The known true phase, if provided.
    """

    def __init__(
        self,
        bitstring: str,
        phase_estimate: float,
        precision: int,
        true_phase: Optional[float] = None,
    ) -> None:
        self.bitstring = bitstring
        self.phase_estimate = phase_estimate
        self.precision = precision
        self.true_phase = true_phase

    def print_phase(self) -> None:
        """Print a formatted summary of the phase estimation result."""
        print(f"\nQuantum Phase Estimation Result")
        print("─" * 40)
        print(f"  Precision       : {self.precision} qubits")
        print(f"  Phase register  : |{self.bitstring}>")
        print(f"  Phase estimate  : {self.phase_estimate:.6f}")
        print(f"  As fraction     : {self.bitstring} in binary")
        print(f"  = {self._as_fraction()}")
        if self.true_phase is not None:
            error = abs(self.phase_estimate - self.true_phase)
            print(f"  True phase      : {self.true_phase:.6f}")
            print(f"  Error           : {error:.6f}")
        print()

    def _as_fraction(self) -> str:
        """Express the phase as a fraction k/2^n."""
        k = int(self.bitstring, 2)
        denom = 2 ** self.precision
        from math import gcd
        g = gcd(k, denom)
        return f"{k // g}/{denom // g}"

    def __repr__(self) -> str:
        return (
            f"QPEResult(phase={self.phase_estimate:.6f}, "
            f"bits={self.bitstring!r}, precision={self.precision})"
        )


class QPE:
    """
    Quantum Phase Estimation algorithm.

    Given a unitary U and an eigenstate |psi>, estimates the phase phi
    such that U|psi> = e^(2*pi*i*phi)|psi>.

    Parameters
    ----------
    U : Matrix
        The unitary gate to estimate the phase of (2x2 for a single qubit).
    eigenstate : list[int]
        The eigenstate |psi> as a list of bits, e.g. [1] for |1>.
        Must be an eigenstate of U.
    precision : int
        Number of qubits in the phase register. More precision = more bits
        of the phase. Resolution is 1/2^precision.
    true_phase : float, optional
        The known true phase (for verification/display).
    """

    def __init__(
        self,
        U: Matrix,
        eigenstate: List[int],
        precision: int,
        true_phase: Optional[float] = None,
    ) -> None:
        if precision < 1:
            raise ValueError("Precision must be at least 1 qubit.")
        if U.rows != 2 or U.cols != 2:
            raise ValueError("U must be a 2×2 single-qubit unitary.")
        self.U = U
        self.eigenstate = eigenstate
        self.precision = precision
        self.true_phase = true_phase
        self.n_eigenstate = len(eigenstate)
        self.total_qubits = precision + self.n_eigenstate

    # ── main entry point ──────────────────────────────────────────────────────

    @classmethod
    def run(
        cls,
        U: Matrix,
        eigenstate: List[int],
        precision: int,
        true_phase: Optional[float] = None,
        verbose: bool = True,
    ) -> QPEResult:
        """
        Run QPE and return the estimated phase.

        Parameters
        ----------
        U : Matrix
            The unitary gate (2×2).
        eigenstate : list[int]
            The eigenstate as a bit list.
        precision : int
            Number of phase qubits.
        true_phase : float, optional
            Known true phase for error reporting.
        verbose : bool
            Print step-by-step output.

        Returns
        -------
        QPEResult
        """
        qpe = cls(U, eigenstate, precision, true_phase)
        if verbose:
            print(f"\nRunning QPE with {precision} precision qubits")
            if true_phase is not None:
                print(f"True phase: {true_phase:.6f}")

        bitstring, phase = qpe._execute()
        result = QPEResult(bitstring, phase, precision, true_phase)
        if verbose:
            result.print_phase()
        return result

    # ── execution ─────────────────────────────────────────────────────────────

    def _execute(self) -> Tuple[str, float]:
        """
        Execute the QPE circuit once and return (bitstring, phase).

        Steps:
        1. Initialise total register (precision + eigenstate qubits)
        2. Prepare eigenstate in the last n_eigenstate qubits
        3. Apply H to all precision qubits
        4. Apply controlled-U^(2^k) for each precision qubit k
        5. Apply inverse QFT to precision register
        6. Measure precision register
        7. Convert bitstring to phase
        """
        state = QuantumState(self.total_qubits, seed=42)

        # ── Step 2: Prepare eigenstate register ────────────────────────────
        for i, bit in enumerate(self.eigenstate):
            if bit == 1:
                qubit_idx = self.precision + i
                state.apply_single(G.X, qubit_idx)

        # ── Step 3: Hadamard on all precision qubits ───────────────────────
        for q in range(self.precision):
            state.apply_single(G.H, q)

        # ── Step 4: Controlled-U^(2^k) gates ──────────────────────────────
        # Qubit q in the phase register controls U^(2^(precision-1-q))
        # applied to the eigenstate register.
        # We process from the last phase qubit (controls U^1) to the first
        # (controls U^(2^(precision-1))).
        for q in range(self.precision - 1, -1, -1):
            power = 2 ** (self.precision - 1 - q)
            # Build U^power by repeated matrix multiplication
            U_power = _matrix_power(self.U, power)
            # Build controlled-U^power as a 4×4 matrix
            CU = _build_controlled_unitary(U_power)
            # Apply: control = phase qubit q, target = eigenstate qubit
            target = self.precision  # first eigenstate qubit
            state.apply_two(CU, q, target)

        # ── Step 5: Inverse QFT on precision register ──────────────────────
        iqft = QFT(self.precision).inverse()
        for instr in iqft.circuit._instructions:
            if instr.name == "barrier":
                continue
            n_q = len(instr.qubits)
            if n_q == 1:
                state.apply_single(instr.gate, instr.qubits[0])
            elif n_q == 2:
                state.apply_two(instr.gate, instr.qubits[0], instr.qubits[1])

        # ── Step 6: Measure precision register ─────────────────────────────
        bits = []
        for q in range(self.precision):
            bits.append(str(state.measure(q)))
        bitstring = "".join(bits)

        # ── Step 7: Convert bitstring to phase ──────────────────────────────
        # The bitstring is the binary fraction: phi = 0.b0 b1 b2 ... b_{n-1}
        # = b0/2 + b1/4 + b2/8 + ... + b_{n-1}/2^n
        phase = sum(int(b) / (2 ** (i + 1)) for i, b in enumerate(bitstring))

        return bitstring, phase

    # ── built-in demonstrations ───────────────────────────────────────────────

    @classmethod
    def t_gate_demo(cls, precision: int = 4, verbose: bool = True) -> QPEResult:
        """
        Demonstrate QPE using the T gate.

        T|1> = e^(i*pi/4)|1> = e^(2*pi*i * 1/8)|1>

        So the phase is exactly phi = 1/8 = 0.125.
        With 4 precision qubits we can represent 0.125 = 0.0010 in binary
        exactly (since 1/8 = 1/2^3).

        Expected result: bitstring = "0010", phase = 0.125
        """
        T = G.T
        return cls.run(
            U=T,
            eigenstate=[1],
            precision=precision,
            true_phase=1/8,
            verbose=verbose,
        )

    @classmethod
    def s_gate_demo(cls, precision: int = 4, verbose: bool = True) -> QPEResult:
        """
        Demonstrate QPE using the S gate.

        S|1> = e^(i*pi/2)|1> = e^(2*pi*i * 1/4)|1>

        So the phase is phi = 1/4 = 0.25.
        Expected result: bitstring = "0100", phase = 0.25
        """
        S = G.S
        return cls.run(
            U=S,
            eigenstate=[1],
            precision=precision,
            true_phase=1/4,
            verbose=verbose,
        )

    @classmethod
    def z_gate_demo(cls, precision: int = 4, verbose: bool = True) -> QPEResult:
        """
        Demonstrate QPE using the Z gate.

        Z|1> = -|1> = e^(i*pi)|1> = e^(2*pi*i * 1/2)|1>

        So the phase is phi = 1/2 = 0.5.
        Expected result: bitstring = "1000", phase = 0.5
        """
        return cls.run(
            U=G.Z,
            eigenstate=[1],
            precision=precision,
            true_phase=1/2,
            verbose=verbose,
        )

    @classmethod
    def custom_phase_demo(
        cls,
        phi: float,
        precision: int = 4,
        verbose: bool = True,
    ) -> QPEResult:
        """
        Demonstrate QPE with any custom phase phi in [0, 1).

        Builds U = [[1, 0], [0, e^(2*pi*i*phi)]] whose eigenstate |1>
        has eigenvalue e^(2*pi*i*phi).

        Parameters
        ----------
        phi : float
            The phase to encode, between 0 and 1.
        precision : int
            Number of phase qubits.
        """
        phase_factor = cmath.exp(2j * math.pi * phi)
        U = Matrix([
            [1, 0           ],
            [0, phase_factor],
        ])
        return cls.run(
            U=U,
            eigenstate=[1],
            precision=precision,
            true_phase=phi,
            verbose=verbose,
        )

    def __repr__(self) -> str:
        return (
            f"QPE(precision={self.precision}, "
            f"eigenstate={self.eigenstate}, "
            f"true_phase={self.true_phase})"
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _matrix_power(M: Matrix, power: int) -> Matrix:
    """
    Compute M^power by repeated matrix multiplication.
    M^0 = Identity, M^1 = M, M^2 = M@M, etc.
    """
    from .linalg import eye
    if power == 0:
        return eye(M.rows)
    result = M
    for _ in range(power - 1):
        result = result @ M
    return result


def _build_controlled_unitary(U: Matrix) -> Matrix:
    """
    Build a 4×4 controlled-U matrix from a 2×2 unitary U.

    CU = |0><0| ⊗ I  +  |1><1| ⊗ U
       = [[I, 0],
          [0, U]]

    In matrix form for basis |00>,|01>,|10>,|11>:
        (control=0: identity on target)
        (control=1: apply U to target)
    """
    return Matrix([
        [1,        0,        0,        0       ],
        [0,        1,        0,        0       ],
        [0,        0,        U[0, 0],  U[0, 1] ],
        [0,        0,        U[1, 0],  U[1, 1] ],
    ])
