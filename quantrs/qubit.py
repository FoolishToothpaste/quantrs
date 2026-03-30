"""
qubit.py
--------
Qubit wraps a single-qubit Circuit and exposes a fluent gate API.
Every method returns self so calls can be chained:

    q = Qubit()
    q.h().x().rz(0.5).measure()
    result = q.run()
    result.print()
"""

from __future__ import annotations
from typing import Optional

from .circuit import Circuit
from .result import MeasurementResult


class Qubit:
    """
    A single-qubit register with a fluent gate API.

    The qubit starts in the |0> state.  Every gate method modifies
    the internal Circuit in place and returns self for chaining.

    The underlying Circuit is accessible via Qubit.circuit if needed.
    """

    def __init__(self, name: str = "q") -> None:
        self.circuit = Circuit(1, 1, name=name)
        self._measured = False

    # ── single-qubit gates ────────────────────────────────────────────────────

    def h(self) -> "Qubit":
        """Hadamard — creates equal superposition of |0> and |1>."""
        self.circuit.h(0); return self

    def x(self) -> "Qubit":
        """Pauli-X (NOT) — flips |0> to |1> and vice versa."""
        self.circuit.x(0); return self

    def y(self) -> "Qubit":
        """Pauli-Y — bit flip with phase."""
        self.circuit.y(0); return self

    def z(self) -> "Qubit":
        """Pauli-Z — phase flip on |1>."""
        self.circuit.z(0); return self

    def s(self) -> "Qubit":
        """S gate — pi/2 phase rotation."""
        self.circuit.s(0); return self

    def sdg(self) -> "Qubit":
        """S-dagger — conjugate transpose of S."""
        self.circuit.sdg(0); return self

    def t(self) -> "Qubit":
        """T gate — pi/4 phase rotation."""
        self.circuit.t(0); return self

    def tdg(self) -> "Qubit":
        """T-dagger — conjugate transpose of T."""
        self.circuit.tdg(0); return self

    def sx(self) -> "Qubit":
        """Square-root-X — half bit flip."""
        self.circuit.sx(0); return self

    def id(self) -> "Qubit":
        """Identity — no-op."""
        self.circuit.id(0); return self

    def rx(self, theta: float) -> "Qubit":
        """Rotation around the X-axis by theta radians."""
        self.circuit.rx(theta, 0); return self

    def ry(self, theta: float) -> "Qubit":
        """Rotation around the Y-axis by theta radians."""
        self.circuit.ry(theta, 0); return self

    def rz(self, phi: float) -> "Qubit":
        """Rotation around the Z-axis by phi radians."""
        self.circuit.rz(phi, 0); return self

    def p(self, theta: float) -> "Qubit":
        """Phase gate — shift phase of |1> by theta."""
        self.circuit.p(theta, 0); return self

    def u(self, theta: float, phi: float, lam: float) -> "Qubit":
        """Generic single-qubit unitary (3 Euler angles)."""
        self.circuit.u(theta, phi, lam, 0); return self

    # ── measurement ───────────────────────────────────────────────────────────

    def measure(self) -> "Qubit":
        """Add a measurement instruction to the circuit."""
        self.circuit.measure(0, 0)
        self._measured = True
        return self

    def run(self, shots: int = 1024, seed: Optional[int] = None) -> MeasurementResult:
        """
        Run the circuit on the built-in statevector simulator.

        Adds a measurement automatically if none has been added.

        Parameters
        ----------
        shots : int
            Number of simulation samples.
        seed  : int, optional
            Random seed for reproducibility.
        """
        if not self._measured:
            self.measure()
        return self.circuit.run(shots=shots, seed=seed)

    # ── inspection ────────────────────────────────────────────────────────────

    def statevector(self):
        """
        Return the exact statevector (no measurement noise).
        Only valid if no measure() has been added.
        """
        return self.circuit.statevector()

    def draw(self) -> str:
        """Print and return the ASCII circuit diagram."""
        return self.circuit.draw()

    def reset(self) -> "Qubit":
        """Reset the qubit to |0>."""
        self.circuit.reset(0); return self

    def barrier(self) -> "Qubit":
        """Insert a barrier."""
        self.circuit.barrier(); return self

    def __repr__(self) -> str:
        ops = self.circuit.count_ops()
        return f"Qubit(gates={ops}, measured={self._measured})"
