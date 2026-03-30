"""
gates.py
--------
All quantum gate matrices, built from scratch using only Python's built-in
complex numbers and the linalg module.

Every gate is a Matrix instance.  The module exposes:

    Single-qubit gates (2×2 matrices)
    ----------------------------------
    I, X, Y, Z, H, S, Sdg, T, Tdg, SX
    Rx(theta), Ry(theta), Rz(phi), P(theta), U(theta, phi, lam)

    Two-qubit gates (4×4 matrices)
    --------------------------------
    CX, CZ, SWAP

    Three-qubit gate (8×8 matrix)
    ------------------------------
    CCX (Toffoli)

Each gate is documented with its matrix definition and physical meaning.
"""

from __future__ import annotations
import math
import cmath
from .linalg import Matrix, kron, eye, controlled_gate, lift_gate


# ── helpers ───────────────────────────────────────────────────────────────────

_s = 1 / math.sqrt(2)   # 1/√2, used in H and rotation gates
_i = 1j                  # imaginary unit shorthand


# ══════════════════════════════════════════════════════════════════════════════
# Single-qubit gates
# ══════════════════════════════════════════════════════════════════════════════

# Identity — no operation, preserves all amplitudes
I = Matrix([
    [1, 0],
    [0, 1],
])

# Pauli-X — bit flip:  |0> -> |1>,  |1> -> |0>
X = Matrix([
    [0, 1],
    [1, 0],
])

# Pauli-Y — bit flip with phase:  |0> -> i|1>,  |1> -> -i|0>
Y = Matrix([
    [0, -1j],
    [1j,  0],
])

# Pauli-Z — phase flip:  |0> -> |0>,  |1> -> -|1>
Z = Matrix([
    [1,  0],
    [0, -1],
])

# Hadamard — creates equal superposition:
#   |0> -> (|0>+|1>)/√2
#   |1> -> (|0>-|1>)/√2
H = Matrix([
    [_s,  _s],
    [_s, -_s],
])

# S gate — π/2 phase rotation:  |1> -> i|1>
S = Matrix([
    [1, 0],
    [0, 1j],
])

# S† (S-dagger) — conjugate transpose of S:  |1> -> -i|1>
Sdg = Matrix([
    [1,  0],
    [0, -1j],
])

# T gate — π/4 phase rotation:  |1> -> e^(iπ/4)|1>
T = Matrix([
    [1, 0],
    [0, cmath.exp(1j * math.pi / 4)],
])

# T† (T-dagger) — conjugate transpose of T
Tdg = Matrix([
    [1, 0],
    [0, cmath.exp(-1j * math.pi / 4)],
])

# SX — square root of X:  applies a half bit-flip
SX = Matrix([
    [complex(0.5, 0.5),  complex(0.5, -0.5)],
    [complex(0.5, -0.5), complex(0.5, 0.5)],
])


# ── parametric single-qubit gates ─────────────────────────────────────────────

def Rx(theta: float) -> Matrix:
    """
    Rotation around the X-axis by angle theta.

    Rx(θ) = [[cos(θ/2),   -i·sin(θ/2)],
              [-i·sin(θ/2), cos(θ/2)  ]]
    """
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return Matrix([
        [complex(c, 0),      complex(0, -s)],
        [complex(0, -s),     complex(c, 0) ],
    ])


def Ry(theta: float) -> Matrix:
    """
    Rotation around the Y-axis by angle theta.

    Ry(θ) = [[cos(θ/2),  -sin(θ/2)],
              [sin(θ/2),   cos(θ/2)]]
    """
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return Matrix([
        [c, -s],
        [s,  c],
    ])


def Rz(phi: float) -> Matrix:
    """
    Rotation around the Z-axis by angle phi.

    Rz(φ) = [[e^(-iφ/2), 0         ],
              [0,          e^(iφ/2) ]]
    """
    return Matrix([
        [cmath.exp(-1j * phi / 2), 0                       ],
        [0,                        cmath.exp( 1j * phi / 2)],
    ])


def P(theta: float) -> Matrix:
    """
    Phase gate — applies a phase shift to the |1> state only.

    P(θ) = [[1, 0         ],
             [0, e^(iθ)   ]]

    Equivalent to Rz up to a global phase.
    """
    return Matrix([
        [1, 0                   ],
        [0, cmath.exp(1j * theta)],
    ])


def U(theta: float, phi: float, lam: float) -> Matrix:
    """
    Generic single-qubit unitary parameterised by three Euler angles.
    Every single-qubit gate can be written in this form.

    U(θ,φ,λ) = [[cos(θ/2),              -e^(iλ)·sin(θ/2)     ],
                 [e^(iφ)·sin(θ/2),        e^(i(φ+λ))·cos(θ/2) ]]
    """
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return Matrix([
        [c,                                -cmath.exp( 1j * lam) * s],
        [cmath.exp(1j * phi) * s,           cmath.exp(1j * (phi + lam)) * c],
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Two-qubit gates  (4×4 matrices, qubit ordering: q0 is MSB)
# ══════════════════════════════════════════════════════════════════════════════

# CNOT (CX) — flips target if control is |1>
# Basis order: |00>, |01>, |10>, |11>  (control = q0, target = q1)
CX = Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
])

# CZ — applies Z to target if control is |1>
# Equivalent to H-CX-H on the target
CZ = Matrix([
    [1, 0, 0,  0],
    [0, 1, 0,  0],
    [0, 0, 1,  0],
    [0, 0, 0, -1],
])

# SWAP — exchanges the states of two qubits
SWAP = Matrix([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])


# ══════════════════════════════════════════════════════════════════════════════
# Three-qubit gate
# ══════════════════════════════════════════════════════════════════════════════

def _build_ccx() -> Matrix:
    """
    Toffoli (CCX) gate — 8×8 matrix.
    Flips target (q2) if both controls (q0 and q1) are |1>.

    Basis order: |000>..|111> with q0 as MSB.
    Only the |110> <-> |111> entries are swapped.
    """
    m = eye(8)
    # |110> = index 6, |111> = index 7
    m[6, 6] = 0
    m[7, 7] = 0
    m[6, 7] = 1
    m[7, 6] = 1
    return m


CCX = _build_ccx()


# ══════════════════════════════════════════════════════════════════════════════
# Gate registry
# ══════════════════════════════════════════════════════════════════════════════

# Maps gate name -> Matrix (for fixed gates) or callable (for parametric gates)
GATE_REGISTRY = {
    "i":    I,
    "x":    X,
    "y":    Y,
    "z":    Z,
    "h":    H,
    "s":    S,
    "sdg":  Sdg,
    "t":    T,
    "tdg":  Tdg,
    "sx":   SX,
    "cx":   CX,
    "cz":   CZ,
    "swap": SWAP,
    "ccx":  CCX,
    # Parametric — stored as callables
    "rx":   Rx,
    "ry":   Ry,
    "rz":   Rz,
    "p":    P,
    "u":    U,
}


def get_gate(name: str, *params) -> Matrix:
    """
    Look up a gate by name and return its Matrix.

    For parametric gates (rx, ry, rz, p, u) pass the angles as additional
    arguments.  Raises KeyError for unknown gate names.
    """
    entry = GATE_REGISTRY[name.lower()]
    if callable(entry):
        return entry(*params)
    return entry
