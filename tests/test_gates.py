"""
test_gates.py
-------------
Unit tests for every function and gate in gates.py.

Coverage:
    Fixed gates : I, X, Y, Z, H, S, Sdg, T, Tdg, SX, CX, CZ, SWAP, CCX
    Parametric  : Rx, Ry, Rz, P, U
    Helpers     : _build_ccx, get_gate
    All gates   : is_unitary check
"""

import unittest
import math
import cmath
from quantrs.gates import (
    I, X, Y, Z, H, S, Sdg, T, Tdg, SX,
    CX, CZ, SWAP, CCX,
    Rx, Ry, Rz, P, U,
    get_gate, GATE_REGISTRY
)
from quantrs.linalg import Vector


def ap(gate, data):
    """Shorthand: apply gate to a state given as a list."""
    return gate @ Vector(data)


# ── Identity ──────────────────────────────────────────────────────────────────

class TestGateI(unittest.TestCase):

    def test_I_leaves_zero_unchanged(self):
        r = ap(I, [1, 0])
        self.assertAlmostEqual(abs(r[0]), 1)
        self.assertAlmostEqual(abs(r[1]), 0)

    def test_I_leaves_one_unchanged(self):
        r = ap(I, [0, 1])
        self.assertAlmostEqual(abs(r[0]), 0)
        self.assertAlmostEqual(abs(r[1]), 1)

    def test_I_is_unitary(self):
        self.assertTrue(I.is_unitary())


# ── Pauli-X ───────────────────────────────────────────────────────────────────

class TestGateX(unittest.TestCase):

    def test_X_flips_zero_to_one(self):
        r = ap(X, [1, 0])
        self.assertAlmostEqual(abs(r[0]), 0)
        self.assertAlmostEqual(abs(r[1]), 1)

    def test_X_flips_one_to_zero(self):
        r = ap(X, [0, 1])
        self.assertAlmostEqual(abs(r[0]), 1)
        self.assertAlmostEqual(abs(r[1]), 0)

    def test_X_squared_is_identity(self):
        result = X @ X
        self.assertAlmostEqual(result[0, 0].real, 1)
        self.assertAlmostEqual(result[1, 1].real, 1)
        self.assertAlmostEqual(result[0, 1].real, 0)

    def test_X_is_unitary(self):
        self.assertTrue(X.is_unitary())


# ── Pauli-Y ───────────────────────────────────────────────────────────────────

class TestGateY(unittest.TestCase):

    def test_Y_on_zero(self):
        r = ap(Y, [1, 0])
        # Y|0> = i|1>
        self.assertAlmostEqual(abs(r[0]), 0)
        self.assertAlmostEqual(r[1].imag, 1)

    def test_Y_on_one(self):
        r = ap(Y, [0, 1])
        # Y|1> = -i|0>
        self.assertAlmostEqual(r[0].imag, -1)
        self.assertAlmostEqual(abs(r[1]), 0)

    def test_Y_is_unitary(self):
        self.assertTrue(Y.is_unitary())

    def test_Y_squared_is_identity(self):
        result = Y @ Y
        self.assertAlmostEqual(result[0, 0].real, 1)
        self.assertAlmostEqual(result[1, 1].real, 1)


# ── Pauli-Z ───────────────────────────────────────────────────────────────────

class TestGateZ(unittest.TestCase):

    def test_Z_leaves_zero_unchanged(self):
        r = ap(Z, [1, 0])
        self.assertAlmostEqual(r[0].real, 1)

    def test_Z_flips_phase_of_one(self):
        r = ap(Z, [0, 1])
        self.assertAlmostEqual(r[1].real, -1)

    def test_Z_is_unitary(self):
        self.assertTrue(Z.is_unitary())

    def test_Z_squared_is_identity(self):
        result = Z @ Z
        self.assertAlmostEqual(result[0, 0].real, 1)
        self.assertAlmostEqual(result[1, 1].real, 1)


# ── Hadamard ──────────────────────────────────────────────────────────────────

class TestGateH(unittest.TestCase):

    def test_H_on_zero_creates_superposition(self):
        s = 1 / math.sqrt(2)
        r = ap(H, [1, 0])
        self.assertAlmostEqual(r[0].real, s)
        self.assertAlmostEqual(r[1].real, s)

    def test_H_on_one(self):
        s = 1 / math.sqrt(2)
        r = ap(H, [0, 1])
        self.assertAlmostEqual(r[0].real, s)
        self.assertAlmostEqual(r[1].real, -s)

    def test_H_squared_is_identity(self):
        result = H @ H
        self.assertAlmostEqual(result[0, 0].real, 1, places=9)
        self.assertAlmostEqual(result[1, 1].real, 1, places=9)
        self.assertAlmostEqual(result[0, 1].real, 0, places=9)

    def test_H_is_unitary(self):
        self.assertTrue(H.is_unitary())

    def test_H_is_hermitian(self):
        # H† = H
        Hd = H.dagger()
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(H[i, j].real, Hd[i, j].real)


# ── S gate ────────────────────────────────────────────────────────────────────

class TestGateS(unittest.TestCase):

    def test_S_leaves_zero_unchanged(self):
        r = ap(S, [1, 0])
        self.assertAlmostEqual(abs(r[0]), 1)
        self.assertAlmostEqual(abs(r[1]), 0)

    def test_S_on_one_gives_i(self):
        r = ap(S, [0, 1])
        self.assertAlmostEqual(r[1].imag, 1)

    def test_S_is_unitary(self):
        self.assertTrue(S.is_unitary())

    def test_S_squared_is_Z(self):
        result = S @ S
        self.assertAlmostEqual(result[0, 0].real, Z[0, 0].real)
        self.assertAlmostEqual(result[1, 1].real, Z[1, 1].real)


# ── Sdg gate ──────────────────────────────────────────────────────────────────

class TestGateSdg(unittest.TestCase):

    def test_Sdg_on_one_gives_minus_i(self):
        r = ap(Sdg, [0, 1])
        self.assertAlmostEqual(r[1].imag, -1)

    def test_Sdg_is_unitary(self):
        self.assertTrue(Sdg.is_unitary())

    def test_S_times_Sdg_is_identity(self):
        result = S @ Sdg
        self.assertAlmostEqual(result[0, 0].real, 1)
        self.assertAlmostEqual(result[1, 1].real, 1)
        self.assertAlmostEqual(result[0, 1].real, 0)


# ── T gate ────────────────────────────────────────────────────────────────────

class TestGateT(unittest.TestCase):

    def test_T_leaves_zero_unchanged(self):
        r = ap(T, [1, 0])
        self.assertAlmostEqual(abs(r[0]), 1)

    def test_T_on_one_is_pi_4_phase(self):
        r = ap(T, [0, 1])
        expected = cmath.exp(1j * math.pi / 4)
        self.assertAlmostEqual(r[1].real, expected.real)
        self.assertAlmostEqual(r[1].imag, expected.imag)

    def test_T_is_unitary(self):
        self.assertTrue(T.is_unitary())

    def test_T_squared_is_S(self):
        result = T @ T
        self.assertAlmostEqual(result[1, 1].imag, S[1, 1].imag)


# ── Tdg gate ──────────────────────────────────────────────────────────────────

class TestGateTdg(unittest.TestCase):

    def test_Tdg_on_one_is_minus_pi_4_phase(self):
        r = ap(Tdg, [0, 1])
        expected = cmath.exp(-1j * math.pi / 4)
        self.assertAlmostEqual(r[1].real, expected.real)
        self.assertAlmostEqual(r[1].imag, expected.imag)

    def test_Tdg_is_unitary(self):
        self.assertTrue(Tdg.is_unitary())

    def test_T_times_Tdg_is_identity(self):
        result = T @ Tdg
        self.assertAlmostEqual(result[0, 0].real, 1)
        self.assertAlmostEqual(result[1, 1].real, 1)


# ── SX gate ───────────────────────────────────────────────────────────────────

class TestGateSX(unittest.TestCase):

    def test_SX_is_unitary(self):
        self.assertTrue(SX.is_unitary())

    def test_SX_squared_is_X(self):
        result = SX @ SX
        self.assertAlmostEqual(result[0, 1].real, X[0, 1].real, places=9)
        self.assertAlmostEqual(result[1, 0].real, X[1, 0].real, places=9)

    def test_SX_on_zero_has_unit_norm(self):
        r = ap(SX, [1, 0])
        norm_sq = abs(r[0])**2 + abs(r[1])**2
        self.assertAlmostEqual(norm_sq, 1.0)


# ── CX gate ───────────────────────────────────────────────────────────────────

class TestGateCX(unittest.TestCase):

    def test_CX_no_flip_when_control_zero(self):
        r = ap(CX, [1, 0, 0, 0])  # |00>
        self.assertAlmostEqual(abs(r[0]), 1)

    def test_CX_flips_target_when_control_one(self):
        r = ap(CX, [0, 0, 1, 0])  # |10>
        self.assertAlmostEqual(abs(r[3]), 1)  # -> |11>

    def test_CX_preserves_01(self):
        r = ap(CX, [0, 1, 0, 0])  # |01> (control=0)
        self.assertAlmostEqual(abs(r[1]), 1)  # stays |01>

    def test_CX_flips_11_to_10(self):
        r = ap(CX, [0, 0, 0, 1])  # |11>
        self.assertAlmostEqual(abs(r[2]), 1)  # -> |10>

    def test_CX_is_unitary(self):
        self.assertTrue(CX.is_unitary())

    def test_CX_squared_is_identity(self):
        result = CX @ CX
        for i in range(4):
            self.assertAlmostEqual(result[i, i].real, 1)
            for j in range(4):
                if i != j:
                    self.assertAlmostEqual(abs(result[i, j]), 0)


# ── CZ gate ───────────────────────────────────────────────────────────────────

class TestGateCZ(unittest.TestCase):

    def test_CZ_no_effect_on_00(self):
        r = ap(CZ, [1, 0, 0, 0])
        self.assertAlmostEqual(abs(r[0]), 1)

    def test_CZ_no_effect_on_01(self):
        r = ap(CZ, [0, 1, 0, 0])
        self.assertAlmostEqual(abs(r[1]), 1)

    def test_CZ_no_effect_on_10(self):
        r = ap(CZ, [0, 0, 1, 0])
        self.assertAlmostEqual(abs(r[2]), 1)

    def test_CZ_flips_phase_of_11(self):
        r = ap(CZ, [0, 0, 0, 1])  # |11>
        self.assertAlmostEqual(r[3].real, -1)

    def test_CZ_is_unitary(self):
        self.assertTrue(CZ.is_unitary())


# ── SWAP gate ─────────────────────────────────────────────────────────────────

class TestGateSWAP(unittest.TestCase):

    def test_SWAP_exchanges_01_to_10(self):
        r = ap(SWAP, [0, 1, 0, 0])  # |01>
        self.assertAlmostEqual(abs(r[2]), 1)  # -> |10>

    def test_SWAP_exchanges_10_to_01(self):
        r = ap(SWAP, [0, 0, 1, 0])  # |10>
        self.assertAlmostEqual(abs(r[1]), 1)  # -> |01>

    def test_SWAP_leaves_00_unchanged(self):
        r = ap(SWAP, [1, 0, 0, 0])
        self.assertAlmostEqual(abs(r[0]), 1)

    def test_SWAP_leaves_11_unchanged(self):
        r = ap(SWAP, [0, 0, 0, 1])
        self.assertAlmostEqual(abs(r[3]), 1)

    def test_SWAP_is_unitary(self):
        self.assertTrue(SWAP.is_unitary())

    def test_SWAP_squared_is_identity(self):
        result = SWAP @ SWAP
        for i in range(4):
            self.assertAlmostEqual(result[i, i].real, 1)


# ── CCX gate ──────────────────────────────────────────────────────────────────

class TestGateCCX(unittest.TestCase):

    def test_CCX_no_flip_when_both_controls_zero(self):
        data = [1] + [0]*7  # |000>
        r = ap(CCX, data)
        self.assertAlmostEqual(abs(r[0]), 1)

    def test_CCX_no_flip_when_one_control_zero(self):
        data = [0]*4 + [1] + [0]*3  # |100>
        r = ap(CCX, data)
        self.assertAlmostEqual(abs(r[4]), 1)

    def test_CCX_no_flip_when_other_control_zero(self):
        data = [0]*2 + [1] + [0]*5  # |010>
        r = ap(CCX, data)
        self.assertAlmostEqual(abs(r[2]), 1)

    def test_CCX_flips_when_both_controls_one(self):
        data = [0]*6 + [1, 0]  # |110>
        r = ap(CCX, data)
        self.assertAlmostEqual(abs(r[7]), 1)  # -> |111>

    def test_CCX_flips_111_to_110(self):
        data = [0]*7 + [1]  # |111>
        r = ap(CCX, data)
        self.assertAlmostEqual(abs(r[6]), 1)  # -> |110>

    def test_CCX_is_unitary(self):
        self.assertTrue(CCX.is_unitary())

    def test_CCX_is_8x8(self):
        self.assertEqual(CCX.rows, 8)
        self.assertEqual(CCX.cols, 8)


# ── Parametric gates ──────────────────────────────────────────────────────────

class TestGateRx(unittest.TestCase):

    def test_Rx_zero_is_identity(self):
        r = Rx(0)
        self.assertAlmostEqual(r[0, 0].real, 1)
        self.assertAlmostEqual(r[1, 1].real, 1)

    def test_Rx_pi_is_minus_i_X(self):
        r = ap(Rx(math.pi), [1, 0])
        self.assertAlmostEqual(abs(r[0]), 0, places=5)
        self.assertAlmostEqual(abs(r[1]), 1, places=5)

    def test_Rx_is_unitary(self):
        for theta in [0.1, 0.5, 1.2, 2.5, math.pi]:
            self.assertTrue(Rx(theta).is_unitary(), f"Rx({theta}) not unitary")

    def test_Rx_2pi_is_minus_identity(self):
        r = Rx(2 * math.pi)
        self.assertAlmostEqual(r[0, 0].real, -1, places=5)
        self.assertAlmostEqual(r[1, 1].real, -1, places=5)


class TestGateRy(unittest.TestCase):

    def test_Ry_zero_is_identity(self):
        r = Ry(0)
        self.assertAlmostEqual(r[0, 0].real, 1)
        self.assertAlmostEqual(r[1, 1].real, 1)

    def test_Ry_pi_flips_zero_to_one(self):
        r = ap(Ry(math.pi), [1, 0])
        self.assertAlmostEqual(abs(r[1]), 1, places=5)

    def test_Ry_is_unitary(self):
        for theta in [0.1, 0.7, 1.5, math.pi]:
            self.assertTrue(Ry(theta).is_unitary(), f"Ry({theta}) not unitary")

    def test_Ry_matrix_is_real(self):
        r = Ry(0.5)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(r[i, j].imag, 0)


class TestGateRz(unittest.TestCase):

    def test_Rz_zero_is_identity(self):
        r = Rz(0)
        self.assertAlmostEqual(abs(r[0, 0]), 1)
        self.assertAlmostEqual(abs(r[1, 1]), 1)

    def test_Rz_is_unitary(self):
        for phi in [0.1, 0.5, 1.0, math.pi]:
            self.assertTrue(Rz(phi).is_unitary(), f"Rz({phi}) not unitary")

    def test_Rz_pi_is_iZ(self):
        r = Rz(math.pi)
        # Rz(pi) = [[e^(-i*pi/2), 0], [0, e^(i*pi/2)]] = [[-i,0],[0,i]]
        self.assertAlmostEqual(abs(r[0, 0]), 1)
        self.assertAlmostEqual(abs(r[1, 1]), 1)
        self.assertAlmostEqual(r[0, 1].real, 0)


class TestGateP(unittest.TestCase):

    def test_P_leaves_zero_unchanged(self):
        r = ap(P(0.5), [1, 0])
        self.assertAlmostEqual(abs(r[0]), 1)
        self.assertAlmostEqual(abs(r[1]), 0)

    def test_P_applies_phase_to_one(self):
        theta = 0.7
        r = ap(P(theta), [0, 1])
        expected = cmath.exp(1j * theta)
        self.assertAlmostEqual(r[1].real, expected.real)
        self.assertAlmostEqual(r[1].imag, expected.imag)

    def test_P_is_unitary(self):
        for theta in [0.1, 0.5, 1.0, math.pi]:
            self.assertTrue(P(theta).is_unitary(), f"P({theta}) not unitary")

    def test_P_zero_is_identity(self):
        r = P(0)
        self.assertAlmostEqual(r[0, 0].real, 1)
        self.assertAlmostEqual(r[1, 1].real, 1)


class TestGateU(unittest.TestCase):

    def test_U_is_unitary(self):
        for args in [(0.3, 0.5, 0.7), (math.pi/2, 0, 0), (0, 0, math.pi)]:
            self.assertTrue(U(*args).is_unitary(), f"U{args} not unitary")

    def test_U_zero_angles_is_identity(self):
        r = U(0, 0, 0)
        self.assertAlmostEqual(r[0, 0].real, 1)
        self.assertAlmostEqual(r[1, 1].real, 1)

    def test_U_preserves_norm(self):
        r = ap(U(0.3, 0.5, 0.7), [1, 0])
        norm_sq = abs(r[0])**2 + abs(r[1])**2
        self.assertAlmostEqual(norm_sq, 1.0)


# ── get_gate ──────────────────────────────────────────────────────────────────

class TestGetGate(unittest.TestCase):

    def test_get_fixed_gate_x(self):
        gate = get_gate("x")
        self.assertAlmostEqual(gate[0, 1].real, X[0, 1].real)

    def test_get_fixed_gate_h(self):
        gate = get_gate("h")
        s = 1 / math.sqrt(2)
        self.assertAlmostEqual(gate[0, 0].real, s)

    def test_get_parametric_rx(self):
        gate = get_gate("rx", 0.5)
        expected = Rx(0.5)
        self.assertAlmostEqual(gate[0, 0].real, expected[0, 0].real)

    def test_get_parametric_u(self):
        gate = get_gate("u", 0.1, 0.2, 0.3)
        expected = U(0.1, 0.2, 0.3)
        self.assertAlmostEqual(gate[0, 0].real, expected[0, 0].real)

    def test_get_unknown_gate_raises(self):
        with self.assertRaises(KeyError):
            get_gate("nonsense")

    def test_all_registered_gates_accessible(self):
        for name in GATE_REGISTRY:
            self.assertIsNotNone(GATE_REGISTRY[name])


if __name__ == "__main__":
    unittest.main()