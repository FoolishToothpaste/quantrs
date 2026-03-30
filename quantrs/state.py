"""
state.py
--------
QuantumState manages the full statevector of an n-qubit register.

The state is a Vector of length 2^n where each entry is the probability
amplitude for the corresponding computational basis state.  The ordering
convention is big-endian: qubit 0 is the most significant bit, so the
basis state |q0 q1 q2 ... q_{n-1}> has index:

    index = q0 * 2^(n-1) + q1 * 2^(n-2) + ... + q_{n-1} * 2^0

Key operations
--------------
    apply_single(gate, qubit)       — apply 2×2 gate to one qubit
    apply_two(gate, q0, q1)         — apply 4×4 gate to two qubits
    apply_three(gate, q0, q1, q2)   — apply 8×8 gate to three qubits
    measure(qubit)                  — collapse one qubit, return classical bit
    measure_all()                   — collapse all qubits, return bit string
    probabilities()                 — list of |amplitude|^2 for each basis state
    peek()                          — read probabilities without collapsing
"""

from __future__ import annotations
import math
import random
from typing import List, Dict, Tuple, Optional

from .linalg import Vector, Matrix, eye, kron


class QuantumState:
    """
    The statevector of an n-qubit quantum register.

    Starts in |0...0> by default. Gate operations modify the statevector
    in place.  Measurement operations collapse the state and return
    classical bit values.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the register.
    seed : int, optional
        Random seed for reproducible measurements.
    """

    def __init__(self, num_qubits: int, seed: Optional[int] = None) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be at least 1.")
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits

        # Initialise to |0...0>: amplitude 1 at index 0, 0 elsewhere
        data = [0.0] * self.dim
        data[0] = 1.0
        self._vec = Vector(data)

        self._rng = random.Random(seed)

    # ── state initialisation ──────────────────────────────────────────────────

    def initialise(self, amplitudes: List[complex]) -> None:
        """
        Set the statevector to an arbitrary list of amplitudes.
        The list must have length 2^num_qubits and will be normalised.

        Used by Teleporter to prepare a specific state on a qubit.
        """
        if len(amplitudes) != self.dim:
            raise ValueError(
                f"Expected {self.dim} amplitudes for {self.num_qubits} qubits, "
                f"got {len(amplitudes)}."
            )
        self._vec = Vector(amplitudes).normalise()

    def initialise_qubit(self, qubit: int, alpha: complex, beta: complex) -> None:
        """
        Set a single qubit to alpha|0> + beta|1> while leaving all other
        qubits in |0>.  The state is normalised automatically.

        This is only well-defined when all other qubits are in |0>,
        which is the case at the start of a circuit.
        """
        # Build the full state vector as |psi> ⊗ |0...0>
        # qubit 0 is MSB, so qubit k contributes at bit position (n-1-k)
        norm = math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
        if norm < 1e-12:
            raise ValueError("Cannot initialise qubit to the zero vector.")
        a = alpha / norm
        b = beta / norm

        # The new state: sum over all basis states where qubit `qubit` = 0 -> * a,
        # qubit `qubit` = 1 -> * b.  All other qubits stay in |0>, so only
        # indices where all non-target bits are 0 get non-zero amplitudes.
        bit_pos = self.num_qubits - 1 - qubit   # bit position in the integer index
        data = [0.0] * self.dim
        # Index where all qubits = 0 except target
        idx_0 = 0                           # all qubits = 0
        idx_1 = 1 << bit_pos               # target qubit = 1, rest = 0
        data[idx_0] = a
        data[idx_1] = b
        self._vec = Vector(data)

    # ── gate application ──────────────────────────────────────────────────────

    def apply_single(self, gate: Matrix, qubit: int) -> None:
        """
        Apply a 2×2 single-qubit gate to `qubit`.

        We build the full 2^n × 2^n unitary by tensoring the gate with
        identity matrices on all other qubits, then multiply the statevector.

        For large n this is expensive, but it is correct and transparent.
        """
        if gate.rows != 2 or gate.cols != 2:
            raise ValueError("Single-qubit gate must be 2×2.")
        full = _lift_single(gate, qubit, self.num_qubits)
        self._vec = full @ self._vec

    def apply_two(self, gate: Matrix, q0: int, q1: int) -> None:
        """
        Apply a 4×4 two-qubit gate to qubits (q0, q1).

        q0 is the most significant qubit in the gate's own index space.
        For CX: q0 = control, q1 = target.

        If q0 and q1 are not adjacent or not in order 0,1 in the full
        register, we permute the state, apply the gate, then unpermute.
        """
        if gate.rows != 4 or gate.cols != 4:
            raise ValueError("Two-qubit gate must be 4×4.")

        if self.num_qubits == 2 and q0 == 0 and q1 == 1:
            # Trivial case: gate acts on the full register directly
            self._vec = gate @ self._vec
            return

        # General case: use the permutation approach
        self._vec = _apply_two_qubit_gate(gate, q0, q1, self.num_qubits, self._vec)

    def apply_three(self, gate: Matrix, q0: int, q1: int, q2: int) -> None:
        """
        Apply an 8×8 three-qubit gate to qubits (q0, q1, q2).
        q0 is MSB, q2 is LSB in the gate's index space.
        For CCX: q0=ctrl1, q1=ctrl2, q2=target.
        """
        if gate.rows != 8 or gate.cols != 8:
            raise ValueError("Three-qubit gate must be 8×8.")
        if self.num_qubits == 3 and q0 == 0 and q1 == 1 and q2 == 2:
            self._vec = gate @ self._vec
            return
        self._vec = _apply_three_qubit_gate(gate, q0, q1, q2, self.num_qubits, self._vec)

    def apply_matrix(self, matrix: Matrix) -> None:
        """Apply an arbitrary 2^n × 2^n unitary directly to the full state."""
        if matrix.rows != self.dim or matrix.cols != self.dim:
            raise ValueError(
                f"Matrix size {matrix.rows}×{matrix.cols} does not match "
                f"state dimension {self.dim}×{self.dim}."
            )
        self._vec = matrix @ self._vec

    # ── measurement ───────────────────────────────────────────────────────────

    def measure(self, qubit: int) -> int:
        """
        Measure a single qubit in the computational basis.

        1. Calculate P(qubit = 0) by summing |amplitude|^2 over all basis
           states where qubit is 0.
        2. Sample a random outcome (0 or 1) weighted by these probabilities.
        3. Collapse the state: zero out all amplitudes inconsistent with the
           outcome and renormalise.

        Returns
        -------
        int  —  0 or 1
        """
        bit_pos = self.num_qubits - 1 - qubit

        # Sum probability for outcome = 0
        p0 = 0.0
        for idx in range(self.dim):
            if not (idx >> bit_pos & 1):   # qubit is 0 at this index
                p0 += abs(self._vec[idx]) ** 2

        outcome = 0 if self._rng.random() < p0 else 1

        # Collapse: zero out inconsistent amplitudes and renormalise
        norm_sq = 0.0
        new_data = list(self._vec.data)
        for idx in range(self.dim):
            bit_val = (idx >> bit_pos) & 1
            if bit_val != outcome:
                new_data[idx] = 0.0
            else:
                norm_sq += abs(new_data[idx]) ** 2

        norm = math.sqrt(norm_sq)
        if norm > 1e-12:
            new_data = [x / norm for x in new_data]

        self._vec = Vector(new_data)
        return outcome

    def measure_all(self) -> str:
        """
        Measure all qubits at once and return the result as a bitstring.

        Samples a single basis state according to the Born rule
        (probability = |amplitude|^2), then collapses the state.

        Returns
        -------
        str  —  e.g. '010' for |010>
        """
        probs = [abs(x) ** 2 for x in self._vec.data]
        # Cumulative distribution for sampling
        r = self._rng.random()
        cumulative = 0.0
        chosen = self.dim - 1
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                chosen = i
                break

        # Collapse to the chosen state
        new_data = [0.0] * self.dim
        new_data[chosen] = 1.0
        self._vec = Vector(new_data)

        # Convert index to bitstring (MSB = qubit 0)
        return format(chosen, f"0{self.num_qubits}b")

    def measure_qubit_no_collapse(self, qubit: int) -> Tuple[float, float]:
        """
        Return (P(0), P(1)) for a qubit without collapsing the state.
        Used internally and for testing.
        """
        bit_pos = self.num_qubits - 1 - qubit
        p0 = sum(
            abs(self._vec[idx]) ** 2
            for idx in range(self.dim)
            if not ((idx >> bit_pos) & 1)
        )
        return p0, 1.0 - p0

    # ── inspection ────────────────────────────────────────────────────────────

    def probabilities(self) -> List[float]:
        """Return list of |amplitude|^2 for each basis state."""
        return [abs(x) ** 2 for x in self._vec.data]

    def amplitudes(self) -> List[complex]:
        """Return the raw complex amplitudes."""
        return list(self._vec.data)

    def basis_label(self, index: int) -> str:
        """Return the bitstring label for basis state at index."""
        return format(index, f"0{self.num_qubits}b")

    def peek(self) -> Dict[str, float]:
        """
        Return a dict of {bitstring: probability} for all non-negligible
        basis states.  Does not collapse the state.
        """
        result = {}
        for i, p in enumerate(self.probabilities()):
            if p > 1e-9:
                result[self.basis_label(i)] = p
        return result

    def copy(self) -> "QuantumState":
        """Return a deep copy of this state."""
        new = QuantumState(self.num_qubits)
        new._vec = self._vec.copy()
        new._rng = random.Random()
        new._rng.setstate(self._rng.getstate())
        return new

    def __repr__(self) -> str:
        terms = []
        for i, amp in enumerate(self._vec.data):
            if abs(amp) > 1e-9:
                label = self.basis_label(i)
                if amp.imag == 0:
                    terms.append(f"{amp.real:.4f}|{label}>")
                else:
                    terms.append(f"({amp:.4f})|{label}>")
        return " + ".join(terms) if terms else "0"


# ── Internal helpers for gate application ─────────────────────────────────────

def _lift_single(gate: Matrix, qubit: int, num_qubits: int) -> Matrix:
    """
    Lift a 2×2 gate to the full 2^n space by tensoring with identities.
    qubit 0 = MSB, qubit n-1 = LSB.
    """
    result = eye(1)
    for i in range(num_qubits):
        result = kron(result, gate if i == qubit else eye(2))
    return result


def _apply_two_qubit_gate(
    gate: Matrix,
    q0: int,
    q1: int,
    num_qubits: int,
    vec: Vector,
) -> Vector:
    """
    Apply a 4×4 gate to qubits q0 (MSB) and q1 (LSB) within an n-qubit register.

    Strategy: permute the statevector so q0 and q1 are in positions 0 and 1,
    apply the gate to the first two qubits, then unpermute.
    """
    # Build permutation that moves q0 -> 0, q1 -> 1, others -> 2,3,...
    other_qubits = [i for i in range(num_qubits) if i not in (q0, q1)]
    perm = [q0, q1] + other_qubits       # new qubit order
    inv_perm = [0] * num_qubits
    for new_pos, old_pos in enumerate(perm):
        inv_perm[old_pos] = new_pos

    # Permute the statevector
    perm_vec = _permute_state(vec, perm, num_qubits)

    # Apply gate to the first two qubits of the permuted register
    # Build: gate ⊗ I ⊗ ... ⊗ I  (gate on qubits 0,1; I on rest)
    n_rest = num_qubits - 2
    if n_rest == 0:
        full_gate = gate
    else:
        full_gate = kron(gate, eye(2 ** n_rest))

    result_vec = full_gate @ perm_vec

    # Unpermute back to original qubit order
    return _permute_state(result_vec, inv_perm, num_qubits)


def _apply_three_qubit_gate(
    gate: Matrix,
    q0: int,
    q1: int,
    q2: int,
    num_qubits: int,
    vec: Vector,
) -> Vector:
    """
    Apply an 8×8 gate to qubits q0, q1, q2 within an n-qubit register.
    Same permutation strategy as _apply_two_qubit_gate.
    """
    other_qubits = [i for i in range(num_qubits) if i not in (q0, q1, q2)]
    perm = [q0, q1, q2] + other_qubits
    inv_perm = [0] * num_qubits
    for new_pos, old_pos in enumerate(perm):
        inv_perm[old_pos] = new_pos

    perm_vec = _permute_state(vec, perm, num_qubits)

    n_rest = num_qubits - 3
    if n_rest == 0:
        full_gate = gate
    else:
        full_gate = kron(gate, eye(2 ** n_rest))

    result_vec = full_gate @ perm_vec
    return _permute_state(result_vec, inv_perm, num_qubits)


def _permute_state(vec: Vector, perm: List[int], num_qubits: int) -> Vector:
    """
    Permute the qubit ordering in a statevector.

    perm[new_pos] = old_pos means: the qubit at old_pos in the original
    register moves to new_pos in the permuted register.

    Each basis state index is an n-bit integer where bit k (from MSB)
    corresponds to qubit k.  We rearrange the bits according to perm.
    """
    dim = 2 ** num_qubits
    new_data = [complex(0)] * dim

    for old_idx in range(dim):
        # Extract bits of old_idx
        old_bits = [(old_idx >> (num_qubits - 1 - k)) & 1 for k in range(num_qubits)]
        # Build new index: new bit at position new_pos = old bit at perm[new_pos]
        new_bits = [old_bits[perm[new_pos]] for new_pos in range(num_qubits)]
        new_idx = sum(b << (num_qubits - 1 - pos) for pos, b in enumerate(new_bits))
        new_data[new_idx] = vec[old_idx]

    return Vector(new_data)
