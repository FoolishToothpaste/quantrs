# quantrs

Quantum computing from scratch — no Qiskit, no NumPy, no external dependencies.

Built using the same architecture and logic as Qiskit's core: statevector simulation
via matrix multiplication, gate application via tensor products, measurement via the
Born rule.  Everything is pure Python.

## Install

```bash
pip install quantrs
```

Or from source:

```bash
git clone https://github.com/FoolishToothpaste/quantrs
cd quantrs
bash setup_env.sh
conda activate quantrs
```

---

## What's inside

```
quantrs/
├── linalg.py    — Vector, Matrix, kron, eye  (pure Python linear algebra)
├── gates.py     — All gate matrices: H, X, Y, Z, S, T, CX, CZ, SWAP, CCX, Rx, Ry, Rz, P, U
├── state.py     — QuantumState  (statevector + measurement engine)
├── circuit.py   — Circuit  (instruction list + execution engine + ASCII drawing)
├── qubit.py     — Qubit  (single-qubit fluent API)
├── bell.py      — BellState  (all four Bell states)
└── teleport.py  — Teleporter  (full quantum teleportation protocol)
```

---

## 1. Single qubit

```python
from quantrs import Qubit

q = Qubit()
q.h().x().rz(0.5).measure()
result = q.run(shots=1024)
result.print()
```

Every gate method returns `self` for chaining. The qubit starts in |0>.

**Available gates**

| Method | Gate | Description |
|--------|------|-------------|
| `h()` | H | Hadamard |
| `x()` | X | Pauli-X (bit flip) |
| `y()` | Y | Pauli-Y |
| `z()` | Z | Pauli-Z (phase flip) |
| `s()` | S | π/2 phase rotation |
| `sdg()` | S† | Conjugate of S |
| `t()` | T | π/4 phase rotation |
| `tdg()` | T† | Conjugate of T |
| `sx()` | SX | Square root of X |
| `rx(theta)` | Rx | X-axis rotation |
| `ry(theta)` | Ry | Y-axis rotation |
| `rz(phi)` | Rz | Z-axis rotation |
| `p(theta)` | P | Phase gate |
| `u(theta, phi, lam)` | U | Generic 3-angle unitary |

```python
# Exact statevector (no measurement noise)
sv = Qubit().h().statevector()
# sv = [0.707+0j, 0.707+0j]
```

---

## 2. Multi-qubit circuit

```python
from quantrs import Circuit

c = Circuit(2, 2)           # 2 qubits, 2 classical bits
c.h(0).cx(0, 1)             # Bell state
c.measure_all()
result = c.run(shots=1024)
result.print()

# Draw the circuit
c.draw()

# Circuit metrics
print(c.depth())             # critical-path depth
print(c.count_ops())         # {'h': 1, 'cx': 1, 'measure': 2}
```

**Two and three-qubit gates**

```python
c.cx(0, 1)          # CNOT
c.cz(0, 1)          # Controlled-Z
c.swap(0, 1)        # SWAP
c.ccx(0, 1, 2)      # Toffoli
```

**Classical conditioning**  (used in teleportation)

```python
c.x_if(qubit=2, clbit=1, val=1)   # apply X to qubit 2 if c[1] == 1
c.z_if(qubit=2, clbit=0, val=1)   # apply Z to qubit 2 if c[0] == 1
```

---

## 3. Bell states

```python
from quantrs import BellState

# All four Bell states
phi_plus  = BellState.phi_plus()    # (|00⟩ + |11⟩) / √2
phi_minus = BellState.phi_minus()   # (|00⟩ − |11⟩) / √2
psi_plus  = BellState.psi_plus()    # (|01⟩ + |10⟩) / √2
psi_minus = BellState.psi_minus()   # (|01⟩ − |10⟩) / √2

# Draw and run
phi_plus.draw()
result = phi_plus.run(shots=2048)
result.print()
# Expected: ~50% |00⟩, ~50% |11⟩

# Exact statevector
sv = phi_plus.statevector()
# [0.707, 0, 0, 0.707]

# Verify against theoretical values
assert phi_plus.verify()

# All four at once
for bell in BellState.all_four():
    print(bell.name, bell.formula)
    bell.run(shots=500).print()
```

---

## 4. Quantum teleportation

```python
from quantrs import Teleporter

# Teleport named states
Teleporter.run(state="zero")
Teleporter.run(state="one")
Teleporter.run(state="plus")
Teleporter.run(state="minus")
Teleporter.run(state="i")

# Teleport a custom state  alpha|0⟩ + beta|1⟩
from math import sqrt
Teleporter.run(state=[sqrt(1/3), sqrt(2/3)])

# Inspect the result
result = Teleporter.run(state="plus", shots=2000, verbose=False)
print(result.sahaj_probabilities())   # {'0': 0.501, '1': 0.499}
print(result.fidelity())            # ~0.9999

# Build without running
tp = Teleporter(state="plus")
tp.describe()
```

**Named states**

| Name | State |
|------|-------|
| `"zero"` | \|0⟩ |
| `"one"` | \|1⟩ |
| `"plus"` | (\|0⟩ + \|1⟩) / √2 |
| `"minus"` | (\|0⟩ − \|1⟩) / √2 |
| `"i"` | (\|0⟩ + i\|1⟩) / √2 |

---

## How it works (from scratch)

**Linear algebra** (`linalg.py`)
- `Vector` — complex column vector, inner product, tensor product, normalise
- `Matrix` — complex matrix, `@` for matrix multiply and gate-apply, `dagger()`, `is_unitary()`, `kron()`

**Gate matrices** (`gates.py`)
- Every gate is a hardcoded `Matrix` of the correct size
- Fixed gates: `I, X, Y, Z, H, S, Sdg, T, Tdg, SX, CX, CZ, SWAP, CCX`
- Parametric gates: `Rx(theta), Ry(theta), Rz(phi), P(theta), U(theta, phi, lam)`

**Statevector simulation** (`state.py`)
- State is a `Vector` of 2ⁿ complex amplitudes
- Gates are lifted to the full register by `kron`-ing with identity matrices
- Multi-qubit gates on non-adjacent qubits use qubit permutation then unpermutation
- Measurement uses the Born rule: sample from `|amplitude|²` probabilities, then collapse

**Circuit execution** (`circuit.py`)
- Instructions are `(name, qubits, clbits, gate_matrix, params, condition)`
- `_execute_once()` steps through instructions, routing to `QuantumState.apply_*` or `measure()`
- Classical conditioning (`x_if`, `z_if`) checks recorded measurement bits before applying
- `run(shots)` calls `_execute_once` independently for each shot with an independent seed

---

