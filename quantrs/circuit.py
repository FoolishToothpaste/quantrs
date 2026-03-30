"""
circuit.py
----------
Circuit is the central class of the package.  It holds an ordered list of
instructions and an execution engine that runs them against a QuantumState.

Design
------
An Instruction is a named operation with a gate matrix, a list of target
qubits, and an optional list of classical control bits.

Circuit.run() steps through the instruction list in order:
  - Gate instructions: call QuantumState.apply_single/two/three
  - Measure instructions: call QuantumState.measure, record the outcome
  - Classical-control instructions: check recorded bits, conditionally apply gate


Usage
-----
    c = Circuit(2)
    c.h(0).cx(0, 1).measure_all()
    result = c.run(shots=1024)
    result.print()
"""

from __future__ import annotations
import math
import random
from collections import Counter
from typing import List, Optional, Dict, Tuple, Any

from .linalg import Matrix, eye
from .state import QuantumState
from . import gates as G
from .result import MeasurementResult


# ── Instruction dataclass ─────────────────────────────────────────────────────

class Instruction:
    """
    A single instruction in a quantum circuit.

    Fields
    ------
    name      : str            — human-readable name ("h", "cx", "measure", ...)
    qubits    : list[int]      — qubit indices this instruction acts on
    clbits    : list[int]      — classical bit indices (for measure / if_test)
    gate      : Matrix | None  — the gate matrix (None for barrier/measure)
    params    : list[float]    — rotation angles for parametric gates
    condition : tuple | None   — (clbit_index, value) for classically conditioned ops
    """

    def __init__(
        self,
        name: str,
        qubits: List[int],
        clbits: Optional[List[int]] = None,
        gate: Optional[Matrix] = None,
        params: Optional[List[float]] = None,
        condition: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.name = name
        self.qubits = qubits
        self.clbits = clbits or []
        self.gate = gate
        self.params = params or []
        self.condition = condition   # (clbit_index, expected_value)

    def __repr__(self) -> str:
        parts = [self.name, str(self.qubits)]
        if self.params:
            parts.append(f"params={[round(p, 4) for p in self.params]}")
        if self.condition:
            parts.append(f"if c[{self.condition[0]}]=={self.condition[1]}")
        return f"Instruction({', '.join(parts)})"


# ── Circuit ───────────────────────────────────────────────────────────────────

class Circuit:
    """
    A quantum circuit: an ordered list of instructions operating on
    `num_qubits` qubits and `num_clbits` classical bits.

    Parameters
    ----------
    num_qubits : int
    num_clbits : int, optional  — defaults to num_qubits
    name       : str, optional
    """

    def __init__(
        self,
        num_qubits: int,
        num_clbits: Optional[int] = None,
        name: str = "circuit",
    ) -> None:
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits if num_clbits is not None else num_qubits
        self.name = name
        self._instructions: List[Instruction] = []
        self._measured: bool = False

    # ── private append ────────────────────────────────────────────────────────

    def _add(
        self,
        name: str,
        qubits: List[int],
        gate: Optional[Matrix] = None,
        clbits: Optional[List[int]] = None,
        params: Optional[List[float]] = None,
        condition: Optional[Tuple[int, int]] = None,
    ) -> "Circuit":
        self._validate_qubits(qubits)
        self._instructions.append(
            Instruction(name, qubits, clbits, gate, params, condition)
        )
        return self

    def _validate_qubits(self, qubits: List[int]) -> None:
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(
                    f"Qubit index {q} out of range for {self.num_qubits}-qubit circuit."
                )

    # ── single-qubit gates ────────────────────────────────────────────────────

    def h(self, qubit: int) -> "Circuit":
        """Hadamard."""
        return self._add("h", [qubit], G.H)

    def x(self, qubit: int) -> "Circuit":
        """Pauli-X (bit flip)."""
        return self._add("x", [qubit], G.X)

    def y(self, qubit: int) -> "Circuit":
        """Pauli-Y."""
        return self._add("y", [qubit], G.Y)

    def z(self, qubit: int) -> "Circuit":
        """Pauli-Z (phase flip)."""
        return self._add("z", [qubit], G.Z)

    def s(self, qubit: int) -> "Circuit":
        """S gate (pi/2 phase)."""
        return self._add("s", [qubit], G.S)

    def sdg(self, qubit: int) -> "Circuit":
        """S-dagger."""
        return self._add("sdg", [qubit], G.Sdg)

    def t(self, qubit: int) -> "Circuit":
        """T gate (pi/4 phase)."""
        return self._add("t", [qubit], G.T)

    def tdg(self, qubit: int) -> "Circuit":
        """T-dagger."""
        return self._add("tdg", [qubit], G.Tdg)

    def sx(self, qubit: int) -> "Circuit":
        """Square-root-X."""
        return self._add("sx", [qubit], G.SX)

    def id(self, qubit: int) -> "Circuit":
        """Identity (no-op)."""
        return self._add("id", [qubit], G.I)

    def rx(self, theta: float, qubit: int) -> "Circuit":
        """Rx rotation by theta."""
        return self._add("rx", [qubit], G.Rx(theta), params=[theta])

    def ry(self, theta: float, qubit: int) -> "Circuit":
        """Ry rotation by theta."""
        return self._add("ry", [qubit], G.Ry(theta), params=[theta])

    def rz(self, phi: float, qubit: int) -> "Circuit":
        """Rz rotation by phi."""
        return self._add("rz", [qubit], G.Rz(phi), params=[phi])

    def p(self, theta: float, qubit: int) -> "Circuit":
        """Phase gate."""
        return self._add("p", [qubit], G.P(theta), params=[theta])

    def u(self, theta: float, phi: float, lam: float, qubit: int) -> "Circuit":
        """Generic single-qubit unitary."""
        return self._add("u", [qubit], G.U(theta, phi, lam), params=[theta, phi, lam])

    # ── two-qubit gates ───────────────────────────────────────────────────────

    def cx(self, control: int, target: int) -> "Circuit":
        """CNOT gate."""
        return self._add("cx", [control, target], G.CX)

    def cz(self, control: int, target: int) -> "Circuit":
        """Controlled-Z gate."""
        return self._add("cz", [control, target], G.CZ)

    def swap(self, q0: int, q1: int) -> "Circuit":
        """SWAP gate."""
        return self._add("swap", [q0, q1], G.SWAP)

    # ── three-qubit gate ──────────────────────────────────────────────────────

    def ccx(self, ctrl1: int, ctrl2: int, target: int) -> "Circuit":
        """Toffoli (CCX) gate."""
        return self._add("ccx", [ctrl1, ctrl2, target], G.CCX)

    # ── measurement ───────────────────────────────────────────────────────────

    def measure(self, qubit: int, clbit: int) -> "Circuit":
        """Measure qubit into classical bit clbit."""
        if clbit < 0 or clbit >= self.num_clbits:
            raise ValueError(f"Classical bit {clbit} out of range.")
        self._measured = True
        return self._add("measure", [qubit], clbits=[clbit])

    def measure_all(self) -> "Circuit":
        """Measure all qubits into their corresponding classical bits."""
        for i in range(min(self.num_qubits, self.num_clbits)):
            self._add("measure", [i], clbits=[i])
        self._measured = True
        return self

    def barrier(self) -> "Circuit":
        """Insert a barrier (no-op at runtime, shown in circuit diagram)."""
        return self._add("barrier", list(range(self.num_qubits)))

    def reset(self, qubit: int) -> "Circuit":
        """Reset a qubit to |0> (measure and conditionally flip)."""
        return self._add("reset", [qubit])

    # ── classical conditioning ─────────────────────────────────────────────────

    def x_if(self, qubit: int, clbit: int, val: int = 1) -> "Circuit":
        """Apply X to qubit if classical bit clbit equals val."""
        return self._add("x_if", [qubit], G.X, condition=(clbit, val))

    def z_if(self, qubit: int, clbit: int, val: int = 1) -> "Circuit":
        """Apply Z to qubit if classical bit clbit equals val."""
        return self._add("z_if", [qubit], G.Z, condition=(clbit, val))

    # ── execution engine ──────────────────────────────────────────────────────

    def _execute_once(self, seed: Optional[int] = None) -> str:
        """
        Execute the circuit once, returning a classical bit string.

        Steps through instructions in order:
        - Gate instructions: apply gate matrix to QuantumState
        - measure: collapse qubit, record classical bit
        - x_if / z_if: check recorded bits, conditionally apply gate
        - reset: measure and flip back to |0> if needed
        - barrier: no-op
        """
        state = QuantumState(self.num_qubits, seed=seed)
        clbits = [0] * self.num_clbits

        for instr in self._instructions:
            # ── classical condition check ──────────────────────────────────
            if instr.condition is not None:
                cbit_idx, expected = instr.condition
                if clbits[cbit_idx] != expected:
                    continue   # condition not met, skip

            name = instr.name

            # ── barriers are no-ops ────────────────────────────────────────
            if name == "barrier":
                continue

            # ── reset ──────────────────────────────────────────────────────
            if name == "reset":
                q = instr.qubits[0]
                outcome = state.measure(q)
                if outcome == 1:
                    state.apply_single(G.X, q)
                continue

            # ── measurement ────────────────────────────────────────────────
            if name == "measure":
                q = instr.qubits[0]
                cb = instr.clbits[0]
                clbits[cb] = state.measure(q)
                continue

            # ── gate application ────────────────────────────────────────────
            gate = instr.gate
            n_qubits = len(instr.qubits)

            if n_qubits == 1:
                state.apply_single(gate, instr.qubits[0])
            elif n_qubits == 2:
                state.apply_two(gate, instr.qubits[0], instr.qubits[1])
            elif n_qubits == 3:
                state.apply_three(gate, instr.qubits[0], instr.qubits[1], instr.qubits[2])
            else:
                raise NotImplementedError(f"Gates on {n_qubits} qubits are not supported.")

        # Build output bitstring (MSB = classical bit 0)
        return "".join(str(clbits[i]) for i in range(self.num_clbits))

    def run(self, shots: int = 1024, seed: Optional[int] = None) -> MeasurementResult:
        """
        Run the circuit `shots` times and return aggregated measurement counts.

        Each shot gets an independent seed derived from the base seed so
        results are reproducible when seed is set.

        Parameters
        ----------
        shots : int
            Number of times the circuit is sampled.
        seed  : int, optional
            Base random seed for reproducibility.

        Returns
        -------
        MeasurementResult
        """
        rng = random.Random(seed)
        counts: Dict[str, int] = {}
        if not self._measured:
            self.measure_all()
        for _ in range(shots):
            shot_seed = rng.randint(0, 2**31)
            outcome = self._execute_once(seed=shot_seed)
            counts[outcome] = counts.get(outcome, 0) + 1
        return MeasurementResult(counts, shots)

    def statevector(self) -> List[complex]:
        """
        Return the exact statevector after running the circuit once with
        no measurements.  Raises RuntimeError if the circuit contains
        measurement instructions (statevectors are undefined post-collapse).
        """
        for instr in self._instructions:
            if instr.name == "measure":
                raise RuntimeError(
                    "Cannot compute statevector for a circuit containing measurements. "
                    "Remove measure() calls or use run() instead."
                )
        state = QuantumState(self.num_qubits)
        for instr in self._instructions:
            if instr.name == "barrier":
                continue
            n_qubits = len(instr.qubits)
            if n_qubits == 1:
                state.apply_single(instr.gate, instr.qubits[0])
            elif n_qubits == 2:
                state.apply_two(instr.gate, instr.qubits[0], instr.qubits[1])
            elif n_qubits == 3:
                state.apply_three(instr.gate, instr.qubits[0], instr.qubits[1], instr.qubits[2])
        return state.amplitudes()

    # ── circuit info ──────────────────────────────────────────────────────────

    def depth(self) -> int:
        """
        Compute the critical-path depth.
        Tracks the latest timestep each qubit was used and takes the max.
        """
        qubit_time = [0] * self.num_qubits
        for instr in self._instructions:
            if instr.name == "barrier":
                continue
            t = max(qubit_time[q] for q in instr.qubits)
            t += 1
            for q in instr.qubits:
                qubit_time[q] = t
        return max(qubit_time) if qubit_time else 0

    def size(self) -> int:
        """Total number of gate instructions (barriers and identity excluded)."""
        return sum(
            1 for i in self._instructions
            if i.name not in ("barrier", "id")
        )

    def count_ops(self) -> Dict[str, int]:
        """Return {gate_name: count} for all instructions."""
        counts: Dict[str, int] = {}
        for instr in self._instructions:
            counts[instr.name] = counts.get(instr.name, 0) + 1
        return counts

    def inverse(self) -> "Circuit":
        """
        Return the inverse circuit by reversing the instruction list and
        replacing each gate with its conjugate transpose (dagger).
        """
        inv = Circuit(self.num_qubits, self.num_clbits, name=self.name + "_inv")
        for instr in reversed(self._instructions):
            if instr.name in ("measure", "barrier", "reset", "x_if", "z_if"):
                # Non-unitary and conditional operations have no inverse
                raise RuntimeError(
                    f"Cannot invert a circuit containing '{instr.name}' instructions."
                )
            dagger_gate = instr.gate.dagger()
            inv._add(instr.name + "_dg", instr.qubits, dagger_gate, params=instr.params)
        return inv

    # ── ASCII drawing ─────────────────────────────────────────────────────────

    def draw(self) -> str:
        """
        Return an ASCII diagram of the circuit.

        Format:
            q0: ─H──●──────
            q1: ─────X──M──
            c:       ↑  0

        Each column in the diagram corresponds to one instruction.
        Two-qubit gates show a vertical line connecting control and target.
        """
        lines = _draw_circuit(self)
        print(lines)
        return lines

    def __repr__(self) -> str:
        return (
            f"Circuit(name={self.name!r}, qubits={self.num_qubits}, "
            f"clbits={self.num_clbits}, depth={self.depth()}, "
            f"ops={self.count_ops()})"
        )


# ── ASCII drawing engine ──────────────────────────────────────────────────────

_GATE_SYMBOLS = {
    "h": "H", "x": "X", "y": "Y", "z": "Z",
    "s": "S", "sdg": "S†", "t": "T", "tdg": "T†",
    "sx": "SX", "id": "I",
    "rx": "Rx", "ry": "Ry", "rz": "Rz",
    "p": "P", "u": "U",
    "cx": ("●", "⊕"),    # (control symbol, target symbol)
    "cz": ("●", "Z"),
    "swap": ("×", "×"),
    "ccx": ("●", "●", "⊕"),
    "measure": "M",
    "barrier": "│",
    "reset": "|0>",
    "x_if": "X?",
    "z_if": "Z?",
}


def _draw_circuit(circuit: Circuit) -> str:
    """Build and return an ASCII circuit diagram string."""
    nq = circuit.num_qubits
    nc = circuit.num_clbits

    # Qubit wire labels
    q_labels = [f"q{i}: " for i in range(nq)]
    c_labels = [f"c{i}: " for i in range(nc)]
    label_w = max(len(l) for l in q_labels + c_labels)

    # Pad labels
    q_labels = [l.ljust(label_w) for l in q_labels]
    c_labels = [l.ljust(label_w) for l in c_labels]

    # Build columns
    q_wires = [[] for _ in range(nq)]
    c_wires = [[] for _ in range(nc)]

    def col_width(sym):
        return max(3, len(sym) + 2)

    for instr in circuit._instructions:
        name = instr.name
        qubits = instr.qubits
        clbits = instr.clbits

        if name == "barrier":
            w = 3
            for q in range(nq):
                q_wires[q].append("─┤─")
            for c in range(nc):
                c_wires[c].append("   ")
            continue

        if name == "measure":
            q = qubits[0]
            cb = clbits[0]
            w = 5
            for i in range(nq):
                if i == q:
                    q_wires[i].append("─[M]─")
                else:
                    q_wires[i].append("─────")
            for i in range(nc):
                if i == cb:
                    c_wires[i].append("══╩══" if True else "─────")
                else:
                    c_wires[i].append("─────")
            continue

        if name in ("cx", "cz", "swap"):
            syms = _GATE_SYMBOLS[name]
            ctrl, tgt = qubits[0], qubits[1]
            w = 5
            for i in range(nq):
                if i == ctrl:
                    q_wires[i].append(f"──{syms[0]}──")
                elif i == tgt:
                    q_wires[i].append(f"──{syms[1]}──")
                elif min(ctrl, tgt) < i < max(ctrl, tgt):
                    q_wires[i].append("──┼──")
                else:
                    q_wires[i].append("─────")
            for i in range(nc):
                c_wires[i].append("─────")
            continue

        if name == "ccx":
            syms = _GATE_SYMBOLS["ccx"]
            c1, c2, tgt = qubits[0], qubits[1], qubits[2]
            w = 5
            involved = {c1, c2, tgt}
            mn, mx = min(c1, c2, tgt), max(c1, c2, tgt)
            for i in range(nq):
                if i == c1:
                    q_wires[i].append(f"──{syms[0]}──")
                elif i == c2:
                    q_wires[i].append(f"──{syms[1]}──")
                elif i == tgt:
                    q_wires[i].append(f"──{syms[2]}──")
                elif mn < i < mx:
                    q_wires[i].append("──┼──")
                else:
                    q_wires[i].append("─────")
            for i in range(nc):
                c_wires[i].append("─────")
            continue

        # Single-qubit gate (including x_if, z_if)
        sym = _GATE_SYMBOLS.get(name, name.upper())
        cell = f"─{sym}─"
        empty = "─" * len(cell)
        for i in range(nq):
            if i == qubits[0]:
                q_wires[i].append(cell)
            else:
                q_wires[i].append(empty)
        for i in range(nc):
            c_wires[i].append("─" * len(cell))

    # Assemble output
    out_lines = []
    for i in range(nq):
        out_lines.append(q_labels[i] + "".join(q_wires[i]))
    for i in range(nc):
        out_lines.append(c_labels[i] + "".join(c_wires[i]))

    return "\n".join(out_lines)
