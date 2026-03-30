"""
bell.py
-------
BellState prepares and runs all four maximally entangled two-qubit Bell states.

All four states are produced using the same two-gate encoder (H then CX)
applied to different initial states:

    Phi+  |00>  →  (|00> + |11>) / √2
    Phi-  |00>  →  Z on q0 first  →  (|00> - |11>) / √2
    Psi+  |00>  →  X on q1 first  →  (|01> + |10>) / √2
    Psi-  |00>  →  X on q1 + Z on q0  →  (|01> - |10>) / √2
"""

from __future__ import annotations
import math
from typing import List, Optional

from .circuit import Circuit
from .result import MeasurementResult


class BellState:
    """
    Factory and container for the four Bell states.

    Each class method returns a BellState instance with an internal Circuit
    ready to run or inspect.

    Attributes
    ----------
    name    : str     — human-readable label, e.g. 'Phi+'
    formula : str     — e.g. '(|00> + |11>) / sqrt(2)'
    circuit : Circuit — the two-qubit circuit that produces this state
    """

    def __init__(self, name: str, formula: str, circuit: Circuit) -> None:
        self.name = name
        self.formula = formula
        self.circuit = circuit

    # ── factory methods ───────────────────────────────────────────────────────

    @classmethod
    def phi_plus(cls) -> "BellState":
        """
        Phi+: (|00> + |11>) / sqrt(2)

        The canonical Bell state. Both qubits always agree when measured.
        Created by H on q0 followed by CX(0, 1) acting on |00>.
        """
        c = Circuit(2, 2, name="Phi+")
        c.h(0)
        c.cx(0, 1)
        return cls("Phi+", "(|00> + |11>) / sqrt(2)", c)

    @classmethod
    def phi_minus(cls) -> "BellState":
        """
        Phi-: (|00> - |11>) / sqrt(2)

        H creates superposition, then Z flips the phase of |1> so the
        superposition becomes (|0>-|1>)/√2, then CX entangles to give
        (|00>-|11>)/√2.  Z must come AFTER H, not before.
        """
        c = Circuit(2, 2, name="Phi-")
        c.h(0)
        c.z(0)
        c.cx(0, 1)
        return cls("Phi-", "(|00> - |11>) / sqrt(2)", c)

    @classmethod
    def psi_plus(cls) -> "BellState":
        """
        Psi+: (|01> + |10>) / sqrt(2)

        X on q1 before the encoder flips the second qubit.
        Both qubits always disagree when measured.
        """
        c = Circuit(2, 2, name="Psi+")
        c.x(1)
        c.h(0)
        c.cx(0, 1)
        return cls("Psi+", "(|01> + |10>) / sqrt(2)", c)

    @classmethod
    def psi_minus(cls) -> "BellState":
        """
        Psi-: (|01> - |10>) / sqrt(2)

        X on q1 flips the second qubit so the pair always disagrees.
        H on q0 creates superposition, then Z on q0 flips the sign of
        the |1> branch, giving (|0>-|1>)/√2 ⊗ |1>.  CX then entangles
        to give (|01>-|10>)/√2.  Z must come AFTER H.
        """
        c = Circuit(2, 2, name="Psi-")
        c.x(1)
        c.h(0)
        c.z(0)
        c.cx(0, 1)
        return cls("Psi-", "(|01> - |10>) / sqrt(2)", c)

    @classmethod
    def all_four(cls) -> List["BellState"]:
        """Return a list of all four Bell states."""
        return [cls.phi_plus(), cls.phi_minus(), cls.psi_plus(), cls.psi_minus()]

    # ── simulation ────────────────────────────────────────────────────────────

    def run(self, shots: int = 1024, seed: Optional[int] = None) -> MeasurementResult:
        """
        Simulate the Bell state and return measurement counts.

        Expected results:
            Phi+, Phi-  →  ~50% |00>, ~50% |11>
            Psi+, Psi-  →  ~50% |01>, ~50% |10>
        """
        return self.circuit.run(shots=shots, seed=seed)

    def statevector(self) -> List[complex]:
        """
        Return the exact statevector without measurement noise.

        Index ordering (big-endian, q0=MSB):
            [amp_00, amp_01, amp_10, amp_11]

        For Phi+: [1/√2, 0, 0, 1/√2]
        """
        return self.circuit.statevector()

    def verify(self, tol: float = 1e-6) -> bool:
        """
        Verify the Bell state by checking the statevector amplitudes match
        the expected theoretical values.

        Returns True if correct, False otherwise.
        """
        sv = self.statevector()
        expected = _EXPECTED[self.name]
        for got, want in zip(sv, expected):
            if abs(got - want) > tol:
                return False
        return True

    # ── display ───────────────────────────────────────────────────────────────

    def draw(self) -> None:
        """Print the Bell state name and circuit diagram."""
        print(f"\n{self.name}:  {self.formula}")
        self.circuit.draw()

    def __repr__(self) -> str:
        return f"BellState({self.name}: {self.formula})"


# Expected statevectors (big-endian: [|00>, |01>, |10>, |11>])
_s = 1 / math.sqrt(2)
_EXPECTED = {
    "Phi+":  [_s,   0,    0,   _s],
    "Phi-":  [_s,   0,    0,  -_s],
    "Psi+":  [0,    _s,  _s,   0 ],
    "Psi-":  [0,    _s, -_s,   0 ],
}
