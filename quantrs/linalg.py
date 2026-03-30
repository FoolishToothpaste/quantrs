"""
linalg.py
---------
Pure-Python linear algebra primitives used by the rest of the package. Used for operations, and I have avoided numpy for maximal control.

Provides:
    Vector   — 1-D complex column vector
    Matrix   — 2-D complex matrix
    kron()   — Kronecker (tensor) product of two matrices
    eye(n)   — n×n identity matrix
    zeros()  — zero vector of length n
"""

from __future__ import annotations
import math
import cmath
from typing import List


# ── Vector ────────────────────────────────────────────────────────────────────

class Vector:
    """
    A 1-D column vector of complex numbers.

    Internally stored as a plain Python list of complex values.
    Used to represent quantum state vectors.
    """

    def __init__(self, data: List[complex]) -> None:
        self.data: List[complex] = [complex(x) for x in data]
        self.size: int = len(data)

    # ── arithmetic ────────────────────────────────────────────────────────────

    def __add__(self, other: "Vector") -> "Vector":
        if self.size != other.size:
            raise ValueError(f"Vector size mismatch: {self.size} vs {other.size}")
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar: complex) -> "Vector":
        return Vector([x * scalar for x in self.data])

    def __rmul__(self, scalar: complex) -> "Vector":
        return self.__mul__(scalar)

    def __neg__(self) -> "Vector":
        return Vector([-x for x in self.data])

    # ── inner product (bra-ket) ───────────────────────────────────────────────

    def dot(self, other: "Vector") -> complex:
        """
        Inner product <self|other> = sum(conj(self[i]) * other[i]).
        """
        if self.size != other.size:
            raise ValueError(f"Vector size mismatch: {self.size} vs {other.size}")
        return sum(a.conjugate() * b for a, b in zip(self.data, other.data))

    def norm(self) -> float:
        """Euclidean (L2) norm: sqrt(<self|self>)."""
        return math.sqrt(sum(abs(x) ** 2 for x in self.data))

    def normalise(self) -> "Vector":
        """Return a unit-length copy of this vector."""
        n = self.norm()
        if n < 1e-12:
            raise ValueError("Cannot normalise the zero vector.")
        return Vector([x / n for x in self.data])

    def tensor(self, other: "Vector") -> "Vector":
        """
        Tensor (Kronecker) product self ⊗ other.
        For state vectors: |a> ⊗ |b> = |ab>.
        """
        return Vector([a * b for a in self.data for b in other.data])

    # ── indexing ──────────────────────────────────────────────────────────────

    def __getitem__(self, i: int) -> complex:
        return self.data[i]

    def __setitem__(self, i: int, v: complex) -> None:
        self.data[i] = complex(v)

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        return iter(self.data)

    # ── display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        parts = []
        for x in self.data:
            if x.imag == 0:
                parts.append(f"{x.real:.6f}")
            elif x.real == 0:
                parts.append(f"{x.imag:.6f}j")
            else:
                parts.append(f"{x.real:.6f}+{x.imag:.6f}j")
        return f"Vector([{', '.join(parts)}])"

    def copy(self) -> "Vector":
        return Vector(list(self.data))


# ── Matrix ────────────────────────────────────────────────────────────────────

class Matrix:
    """
    A 2-D complex matrix stored row-major as a list of lists.

    rows × cols elements.  Used to represent quantum gates.
    """

    def __init__(self, data: List[List[complex]]) -> None:
        self.rows: int = len(data)
        self.cols: int = len(data[0]) if data else 0
        self.data: List[List[complex]] = [
            [complex(x) for x in row] for row in data
        ]

    # ── arithmetic ────────────────────────────────────────────────────────────

    def __matmul__(self, other: "Matrix | Vector") -> "Matrix | Vector":
        """
        Matrix multiplication: self @ other.
        If other is a Vector, returns a Vector (gate applied to state).
        If other is a Matrix, returns a Matrix (gate composition).
        """
        if isinstance(other, Vector):
            if self.cols != other.size:
                raise ValueError(
                    f"Dimension mismatch: matrix cols={self.cols}, vector size={other.size}"
                )
            result = []
            for row in self.data:
                val = sum(row[j] * other.data[j] for j in range(self.cols))
                result.append(val)
            return Vector(result)

        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError(
                    f"Dimension mismatch: {self.rows}×{self.cols} @ {other.rows}×{other.cols}"
                )
            result = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    val = sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                    row.append(val)
                result.append(row)
            return Matrix(result)

        raise TypeError(f"Cannot multiply Matrix with {type(other)}")

    def __mul__(self, scalar: complex) -> "Matrix":
        return Matrix([[x * scalar for x in row] for row in self.data])

    def __rmul__(self, scalar: complex) -> "Matrix":
        return self.__mul__(scalar)

    def __add__(self, other: "Matrix") -> "Matrix":
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix shape mismatch for addition.")
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    # ── properties ────────────────────────────────────────────────────────────

    def dagger(self) -> "Matrix":
        """Conjugate transpose (Hermitian adjoint) — used for gate inversion."""
        return Matrix([
            [self.data[j][i].conjugate() for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def tensor(self, other: "Matrix") -> "Matrix":
        """
        Kronecker product self ⊗ other.
        Produces a (self.rows*other.rows) × (self.cols*other.cols) matrix.
        Used to build multi-qubit gate matrices from single-qubit ones.
        """
        return kron(self, other)

    def is_unitary(self, tol: float = 1e-9) -> bool:
        """
        Check whether M†M ≈ I.  A matrix is unitary if applying it
        preserves the norm of any state vector.
        """
        product = self.dagger() @ self
        ident = eye(self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(product.data[i][j] - ident.data[i][j]) > tol:
                    return False
        return True

    # ── indexing ──────────────────────────────────────────────────────────────

    def __getitem__(self, ij):
        i, j = ij
        return self.data[i][j]

    def __setitem__(self, ij, val: complex) -> None:
        i, j = ij
        self.data[i][j] = complex(val)

    # ── display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        lines = []
        for row in self.data:
            parts = []
            for x in row:
                if abs(x.imag) < 1e-10:
                    parts.append(f"{x.real:7.4f}")
                else:
                    parts.append(f"{x.real:7.4f}+{x.imag:.4f}j")
            lines.append("  [" + "  ".join(parts) + "]")
        return "Matrix(\n" + "\n".join(lines) + "\n)"

    def copy(self) -> "Matrix":
        return Matrix([list(row) for row in self.data])


# ── Module-level helpers ──────────────────────────────────────────────────────

def eye(n: int) -> Matrix:
    """Return the n×n identity matrix."""
    return Matrix([
        [1 if i == j else 0 for j in range(n)]
        for i in range(n)
    ])


def zeros_vec(n: int) -> Vector:
    """Return a zero vector of length n."""
    return Vector([0] * n)


def kron(a: Matrix, b: Matrix) -> Matrix:
    """
    Kronecker (tensor) product A ⊗ B.

    If A is m×n and B is p×q the result is (m*p) × (n*q).
    This is how single-qubit gates are lifted to multi-qubit registers:
    applying gate G to qubit k of an n-qubit system means computing
    I ⊗ ... ⊗ G ⊗ ... ⊗ I  (G in position k).
    """
    rows = a.rows * b.rows
    cols = a.cols * b.cols
    data = [[0] * cols for _ in range(rows)]
    for i in range(a.rows):
        for j in range(a.cols):
            for p in range(b.rows):
                for q in range(b.cols):
                    data[i * b.rows + p][j * b.cols + q] = a.data[i][j] * b.data[p][q]
    return Matrix(data)


def lift_gate(gate: Matrix, target: int, num_qubits: int) -> Matrix:
    """
    Lift a single-qubit gate to act on qubit `target` in an
    `num_qubits`-qubit system by tensoring with identity matrices.

    lift_gate(H, target=1, num_qubits=3) = I ⊗ H ⊗ I
    """
    result = eye(1)
    for i in range(num_qubits):
        result = kron(result, gate if i == target else eye(2))
    return result


def controlled_gate(gate: Matrix, control: int, target: int, num_qubits: int) -> Matrix:
    """
    Build a controlled-U gate matrix for a general single-qubit unitary U.

    The resulting matrix acts on the full `num_qubits`-qubit space.
    It applies U to `target` if and only if `control` is |1>.

    Implementation:
        CU = |0><0| ⊗ I  +  |1><1| ⊗ U
    expanded to the full register via tensor products.
    """
    dim = 2 ** num_qubits

    # Projectors onto |0> and |1> for the control qubit
    p0 = Matrix([[1, 0], [0, 0]])  # |0><0|
    p1 = Matrix([[0, 0], [0, 1]])  # |1><1|

    # |0><0| ⊗ I on the rest — control is |0>, nothing happens
    part0 = lift_gate_with_proj(p0, control, eye(2), target, num_qubits)

    # |1><1| ⊗ U — control is |1>, apply gate
    part1 = lift_gate_with_proj(p1, control, gate, target, num_qubits)

    return part0 + part1


def lift_gate_with_proj(
    proj: Matrix,
    proj_qubit: int,
    gate: Matrix,
    gate_qubit: int,
    num_qubits: int,
) -> Matrix:
    """
    Build a matrix that applies `proj` to `proj_qubit` and `gate` to
    `gate_qubit` simultaneously, with identity on all other qubits.
    """
    result = eye(1)
    for i in range(num_qubits):
        if i == proj_qubit:
            result = kron(result, proj)
        elif i == gate_qubit:
            result = kron(result, gate)
        else:
            result = kron(result, eye(2))
    return result
