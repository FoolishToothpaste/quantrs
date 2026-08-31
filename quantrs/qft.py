"""
qft.py
------
Quantum Fourier Transform (QFT) implemented from scratch using quantrs.

What is the QFT?
----------------
The Quantum Fourier Transform is the quantum version of the Discrete
Fourier Transform (DFT). Classically, the DFT takes a list of N numbers
and converts them from the time/amplitude domain into the frequency domain.
The QFT does the same thing but on quantum amplitudes.

Given an n-qubit register in state |x>, the QFT produces:

    QFT|x> = (1/sqrt(2^n)) * sum_{k=0}^{2^n - 1} e^(2*pi*i*x*k / 2^n) |k>

This is used as a subroutine in:
    - Quantum Phase Estimation
    - Shor's Algorithm
    - Quantum simulation algorithms

How the circuit works:
----------------------
For n qubits, the QFT circuit has this structure (qubit 0 is top/MSB):

    qubit 0: ─H─CP(pi/2)─CP(pi/4)─...─CP(pi/2^(n-1))─────────────SWAP─
    qubit 1: ──────●──────────────────────────────────H─CP(pi/2)─...────
    qubit 2: ─────────────●──────────────────────────────●──────H─...───
    ...

The building blocks are:
    1. Hadamard (H) on the current qubit
    2. Controlled Phase gates CP(2*pi/2^k) where the phase depends on
       the distance between the control and target qubit
    3. SWAP gates at the end to reverse the bit order

The key insight: each qubit gets a phase kick that encodes its contribution
to the Fourier frequency, determined by all qubits below it.

Usage:
------
    from quantrs.qft import QFT

    # Build a 3-qubit QFT circuit
    qft = QFT(3)
    qft.draw()

    # Apply QFT to a specific input state and read the statevector
    sv = qft.apply(input_state=[1, 0, 0])   # QFT of |100>
    qft.print_statevector(sv)

    # Run with measurements
    result = qft.run(input_state=[1, 0, 0], shots=1024)
    result.print()
"""

from __future__ import annotations
import math
import cmath
from typing import List, Optional

from .circuit import Circuit
from .linalg import Matrix
from .state import QuantumState
from . import gates as G


class QFT:
    """
    Quantum Fourier Transform on n qubits.

    The QFT circuit is built once at initialisation. You can then:
    - draw() the circuit diagram
    - apply() it to an input state and get the exact statevector
    - run() it with measurements to get sampled outputs
    - inverse() to get the inverse QFT (IQFT) circuit

    Parameters
    ----------
    n : int
        Number of qubits. The QFT acts on a 2^n dimensional space.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("QFT requires at least 1 qubit.")
        self.n = n
        self.circuit = self._build_circuit()

    # ── circuit construction ──────────────────────────────────────────────────

    def _build_circuit(self) -> Circuit:
        """
        Build the QFT circuit.

        The algorithm for n qubits:
            for each qubit q from 0 to n-1:
                1. Apply H to qubit q
                2. For each qubit j from q+1 to n-1:
                   Apply CP(2*pi / 2^(j-q+1)) controlled on qubit j,
                   targeting qubit q
            After all rotations:
                3. Apply SWAP gates to reverse the qubit order
                   (because the QFT naturally outputs bits in reversed order)
        """
        c = Circuit(self.n, 0, name=f"QFT_{self.n}")

        # ── Hadamard + controlled phase rotations ──────────────────────────
        for q in range(self.n):
            # H gate on current qubit
            c.h(q)

            # Controlled phase rotation from each qubit below q
            for j in range(q + 1, self.n):
                # The rotation angle: 2*pi / 2^(j - q + 1)
                # When j - q = 1: angle = pi     (this is the CP(pi) = CZ-like gate)
                # When j - q = 2: angle = pi/2
                # When j - q = 3: angle = pi/4
                # etc.
                k = j - q + 1
                angle = 2 * math.pi / (2 ** k)
                cp_gate = _controlled_phase(angle)
                c._add("cp", [j, q], cp_gate, params=[angle])

        # ── SWAP to reverse bit order ────────────────────────────────────
        # The QFT outputs the bits in reversed order compared to the
        # standard binary representation, so we swap to fix that.
        for i in range(self.n // 2):
            c.swap(i, self.n - 1 - i)

        return c

    # ── public API ────────────────────────────────────────────────────────────

    def apply(self, input_state: Optional[List[int]] = None) -> List[complex]:
        """
        Apply the QFT to an input computational basis state and return
        the exact output statevector.

        Parameters
        ----------
        input_state : list[int], optional
            A list of n bits (0 or 1) specifying the input basis state.
            e.g. [1, 0, 0] for |100> on a 3-qubit register.
            Defaults to |0...0> if not given.

        Returns
        -------
        list[complex]
            The 2^n output amplitudes after applying the QFT.
        """
        state = QuantumState(self.n)

        # Prepare input state
        if input_state is not None:
            if len(input_state) != self.n:
                raise ValueError(
                    f"input_state must have {self.n} bits, got {len(input_state)}."
                )
            # Build the index of the basis state from the bit list
            # MSB = qubit 0
            idx = sum(b << (self.n - 1 - i) for i, b in enumerate(input_state))
            data = [0.0] * (2 ** self.n)
            data[idx] = 1.0
            from .linalg import Vector
            state._vec = Vector(data)

        # Apply each instruction
        for instr in self.circuit._instructions:
            if instr.name == "barrier":
                continue
            n_q = len(instr.qubits)
            if n_q == 1:
                state.apply_single(instr.gate, instr.qubits[0])
            elif n_q == 2:
                state.apply_two(instr.gate, instr.qubits[0], instr.qubits[1])

        return state.amplitudes()

    def run(
        self,
        input_state: Optional[List[int]] = None,
        shots: int = 1024,
        seed: Optional[int] = None,
    ):
        """
        Apply the QFT and then measure all qubits.

        This adds measurement instructions to a copy of the circuit, prepares
        the input state, and runs the simulation.

        Parameters
        ----------
        input_state : list[int], optional
            Input basis state as a list of bits. Defaults to |0...0>.
        shots : int
            Number of measurement samples.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        MeasurementResult
        """
        # Build a measuring version of the circuit
        c = Circuit(self.n, self.n, name=f"QFT_{self.n}_meas")

        # Prepare the input
        if input_state is not None:
            if len(input_state) != self.n:
                raise ValueError(
                    f"input_state must have {self.n} bits, got {len(input_state)}."
                )
            for i, bit in enumerate(input_state):
                if bit == 1:
                    c.x(i)

        # Copy QFT instructions
        for instr in self.circuit._instructions:
            c._instructions.append(instr)

        # Measure all
        c.measure_all()
        return c.run(shots=shots, seed=seed)

    def inverse(self) -> "QFT":
        """
        Return the inverse QFT (IQFT).

        The IQFT is just the QFT circuit reversed, with all phase
        angles negated. It is used in Quantum Phase Estimation to
        read out the phase register.

        Returns
        -------
        QFT
            A new QFT object whose circuit is the IQFT.
        """
        iqft = QFT.__new__(QFT)
        iqft.n = self.n
        iqft.circuit = self._build_inverse_circuit()
        return iqft

    def _build_inverse_circuit(self) -> Circuit:
        """
        Build the IQFT circuit by reversing the QFT and negating phases.

        IQFT = QFT† (conjugate transpose of QFT)

        Structure (reverse of QFT):
            1. Undo the SWAP gates first (SWAPs are self-inverse)
            2. For each qubit from n-1 down to 0:
               - Apply CP(-angle) gates (negated phases)
               - Apply H (H is self-inverse)
        """
        c = Circuit(self.n, 0, name=f"IQFT_{self.n}")

        # Undo swaps first
        for i in range(self.n // 2):
            c.swap(i, self.n - 1 - i)

        # Reverse rotations in reverse qubit order
        for q in range(self.n - 1, -1, -1):
            # Undo controlled phase rotations in reverse order
            for j in range(self.n - 1, q, -1):
                k = j - q + 1
                angle = -2 * math.pi / (2 ** k)   # negated angle
                cp_gate = _controlled_phase(angle)
                c._add("cp_inv", [j, q], cp_gate, params=[angle])

            # Undo H (H is self-inverse)
            c.h(q)

        return c

    def draw(self) -> None:
        """Print the QFT circuit diagram."""
        print(f"\nQFT on {self.n} qubits")
        print("─" * 40)
        self.circuit.draw()

    def print_statevector(self, sv: List[complex]) -> None:
        """
        Print the statevector in a readable format showing
        each basis state with its amplitude and probability.
        """
        print(f"\nQFT output statevector ({self.n} qubits, {2**self.n} basis states):")
        print("─" * 50)
        for i, amp in enumerate(sv):
            prob = abs(amp) ** 2
            if prob > 1e-9:
                label = format(i, f"0{self.n}b")
                angle = math.atan2(amp.imag, amp.real)
                print(
                    f"  |{label}>  amp={amp:.4f}  "
                    f"prob={prob:.4f}  phase={angle:.4f} rad"
                )

    def __repr__(self) -> str:
        return f"QFT(n={self.n}, depth={self.circuit.depth()})"


# ── Internal helper ───────────────────────────────────────────────────────────

def _controlled_phase(angle: float) -> Matrix:
    """
    Build a 4×4 controlled phase gate matrix CP(angle).

    CP(angle)|00> = |00>
    CP(angle)|01> = |01>
    CP(angle)|10> = |10>
    CP(angle)|11> = e^(i*angle)|11>

    This applies a phase shift only when both qubits are |1>.
    In matrix form:
        [[1, 0, 0, 0      ],
         [0, 1, 0, 0      ],
         [0, 0, 1, 0      ],
         [0, 0, 0, e^(ia) ]]
    """
    phase = cmath.exp(1j * angle)
    return Matrix([
        [1, 0, 0, 0    ],
        [0, 1, 0, 0    ],
        [0, 0, 1, 0    ],
        [0, 0, 0, phase],
    ])
