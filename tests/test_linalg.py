"""
test_linalg.py
--------------
Unit tests for every function and method in linalg.py.

Coverage:
    Vector  : __init__, __add__, __mul__, __rmul__, __neg__, dot, norm,
              normalise, tensor, __getitem__, __setitem__, __len__,
              __iter__, __repr__, copy
    Matrix  : __init__, __matmul__, __mul__, __rmul__, __add__, dagger,
              tensor, is_unitary, __getitem__, __setitem__, __repr__, copy
    Module  : eye, zeros_vec, kron, lift_gate, controlled_gate,
              lift_gate_with_proj
"""

import unittest
import math
from quantrs.linalg import (
    Vector, Matrix, eye, zeros_vec, kron,
    lift_gate, controlled_gate, lift_gate_with_proj
)


# ── Vector ────────────────────────────────────────────────────────────────────

class TestVectorInit(unittest.TestCase):

    def test_stores_values_as_complex(self):
        v = Vector([1, 2, 3])
        self.assertIsInstance(v.data[0], complex)

    def test_size_attribute(self):
        v = Vector([1, 2, 3, 4])
        self.assertEqual(v.size, 4)

    def test_accepts_complex_input(self):
        v = Vector([1+2j, 3+4j])
        self.assertAlmostEqual(v.data[0].real, 1)
        self.assertAlmostEqual(v.data[0].imag, 2)


class TestVectorAdd(unittest.TestCase):

    def test_add_real_vectors(self):
        a = Vector([1, 2])
        b = Vector([3, 4])
        c = a + b
        self.assertAlmostEqual(c[0].real, 4)
        self.assertAlmostEqual(c[1].real, 6)

    def test_add_complex_vectors(self):
        a = Vector([1j, 2j])
        b = Vector([1j, 0])
        c = a + b
        self.assertAlmostEqual(c[0].imag, 2)

    def test_add_size_mismatch_raises(self):
        a = Vector([1, 2])
        b = Vector([1, 2, 3])
        with self.assertRaises(ValueError):
            _ = a + b


class TestVectorMul(unittest.TestCase):

    def test_mul_real_scalar(self):
        v = Vector([1, 2, 3])
        r = v * 2
        self.assertAlmostEqual(r[0].real, 2)
        self.assertAlmostEqual(r[2].real, 6)

    def test_rmul_real_scalar(self):
        v = Vector([1, 2])
        r = 3 * v
        self.assertAlmostEqual(r[0].real, 3)
        self.assertAlmostEqual(r[1].real, 6)

    def test_mul_complex_scalar(self):
        v = Vector([1, 0])
        r = v * 1j
        self.assertAlmostEqual(r[0].imag, 1)
        self.assertAlmostEqual(r[0].real, 0)


class TestVectorNeg(unittest.TestCase):

    def test_negate_real(self):
        v = Vector([1, -2, 3])
        n = -v
        self.assertAlmostEqual(n[0].real, -1)
        self.assertAlmostEqual(n[1].real, 2)
        self.assertAlmostEqual(n[2].real, -3)

    def test_negate_complex(self):
        v = Vector([1j, -2j])
        n = -v
        self.assertAlmostEqual(n[0].imag, -1)
        self.assertAlmostEqual(n[1].imag, 2)


class TestVectorDot(unittest.TestCase):

    def test_orthogonal_dot_is_zero(self):
        a = Vector([1, 0])
        b = Vector([0, 1])
        self.assertAlmostEqual(abs(a.dot(b)), 0)

    def test_self_dot_is_norm_squared(self):
        v = Vector([3, 4])
        self.assertAlmostEqual(v.dot(v).real, 25)

    def test_complex_inner_product(self):
        # <1j|1j> = conj(1j)*1j = -1j*1j = 1
        a = Vector([1j, 0])
        self.assertAlmostEqual(a.dot(a).real, 1)

    def test_dot_size_mismatch_raises(self):
        a = Vector([1, 2])
        b = Vector([1, 2, 3])
        with self.assertRaises(ValueError):
            a.dot(b)


class TestVectorNorm(unittest.TestCase):

    def test_norm_3_4_is_5(self):
        v = Vector([3, 4])
        self.assertAlmostEqual(v.norm(), 5.0)

    def test_norm_unit_vector(self):
        v = Vector([1, 0])
        self.assertAlmostEqual(v.norm(), 1.0)

    def test_norm_zero_vector(self):
        v = Vector([0, 0])
        self.assertAlmostEqual(v.norm(), 0.0)

    def test_norm_complex_entries(self):
        v = Vector([1j, 1j])
        self.assertAlmostEqual(v.norm(), math.sqrt(2))


class TestVectorNormalise(unittest.TestCase):

    def test_result_is_unit(self):
        v = Vector([3, 4]).normalise()
        self.assertAlmostEqual(v.norm(), 1.0)

    def test_direction_preserved(self):
        v = Vector([2, 0]).normalise()
        self.assertAlmostEqual(v[0].real, 1.0)
        self.assertAlmostEqual(v[1].real, 0.0)

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            Vector([0, 0]).normalise()

    def test_original_unchanged(self):
        v = Vector([3, 4])
        v.normalise()
        self.assertAlmostEqual(v.norm(), 5.0)  # original not modified


class TestVectorTensor(unittest.TestCase):

    def test_tensor_length(self):
        a = Vector([1, 0])
        b = Vector([0, 1])
        self.assertEqual(len(a.tensor(b)), 4)

    def test_zero_tensor_one(self):
        # |0> ⊗ |1> = |01>  →  index 1 = 1
        zero = Vector([1, 0])
        one = Vector([0, 1])
        r = zero.tensor(one)
        self.assertAlmostEqual(abs(r[1]), 1)
        self.assertAlmostEqual(abs(r[0]), 0)
        self.assertAlmostEqual(abs(r[2]), 0)
        self.assertAlmostEqual(abs(r[3]), 0)

    def test_one_tensor_zero(self):
        # |1> ⊗ |0> = |10>  →  index 2 = 1
        zero = Vector([1, 0])
        one = Vector([0, 1])
        r = one.tensor(zero)
        self.assertAlmostEqual(abs(r[2]), 1)

    def test_one_tensor_one(self):
        # |1> ⊗ |1> = |11>  →  index 3 = 1
        one = Vector([0, 1])
        r = one.tensor(one)
        self.assertAlmostEqual(abs(r[3]), 1)


class TestVectorIndexing(unittest.TestCase):

    def test_getitem(self):
        v = Vector([10, 20, 30])
        self.assertAlmostEqual(v[1].real, 20)

    def test_setitem(self):
        v = Vector([1, 2, 3])
        v[1] = 99
        self.assertAlmostEqual(v[1].real, 99)

    def test_setitem_complex(self):
        v = Vector([0, 0])
        v[0] = 1+2j
        self.assertAlmostEqual(v[0].real, 1)
        self.assertAlmostEqual(v[0].imag, 2)

    def test_len(self):
        v = Vector([1, 2, 3, 4, 5])
        self.assertEqual(len(v), 5)

    def test_iter(self):
        v = Vector([1, 2, 3])
        items = list(v)
        self.assertEqual(len(items), 3)
        self.assertAlmostEqual(items[0].real, 1)


class TestVectorReprAndCopy(unittest.TestCase):

    def test_repr_returns_string(self):
        v = Vector([1, 2])
        self.assertIsInstance(repr(v), str)
        self.assertIn("Vector", repr(v))

    def test_copy_is_independent(self):
        v = Vector([1, 2, 3])
        c = v.copy()
        c[0] = 999
        self.assertAlmostEqual(v[0].real, 1)  # original untouched

    def test_copy_has_same_values(self):
        v = Vector([1, 2, 3])
        c = v.copy()
        for i in range(3):
            self.assertAlmostEqual(v[i].real, c[i].real)


# ── Matrix ────────────────────────────────────────────────────────────────────

class TestMatrixInit(unittest.TestCase):

    def test_rows_cols(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 3)

    def test_stores_as_complex(self):
        m = Matrix([[1, 0], [0, 1]])
        self.assertIsInstance(m.data[0][0], complex)

    def test_complex_entries(self):
        m = Matrix([[1j, 0], [0, -1j]])
        self.assertAlmostEqual(m.data[0][0].imag, 1)


class TestMatrixMatmul(unittest.TestCase):

    def test_identity_times_vector(self):
        I = eye(2)
        v = Vector([3, 7])
        r = I @ v
        self.assertAlmostEqual(r[0].real, 3)
        self.assertAlmostEqual(r[1].real, 7)

    def test_H_on_zero_state(self):
        s = 1 / math.sqrt(2)
        H = Matrix([[s, s], [s, -s]])
        zero = Vector([1, 0])
        r = H @ zero
        self.assertAlmostEqual(r[0].real, s)
        self.assertAlmostEqual(r[1].real, s)

    def test_X_on_one_state(self):
        X = Matrix([[0, 1], [1, 0]])
        one = Vector([0, 1])
        r = X @ one
        self.assertAlmostEqual(abs(r[0]), 1)
        self.assertAlmostEqual(abs(r[1]), 0)

    def test_X_squared_is_identity(self):
        X = Matrix([[0, 1], [1, 0]])
        result = X @ X
        self.assertAlmostEqual(result[0, 0].real, 1)
        self.assertAlmostEqual(result[1, 1].real, 1)
        self.assertAlmostEqual(result[0, 1].real, 0)
        self.assertAlmostEqual(result[1, 0].real, 0)

    def test_matmul_dimension_mismatch_raises(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])  # 2x3
        B = Matrix([[1, 2], [3, 4]])          # 2x2
        with self.assertRaises(ValueError):
            _ = A @ B

    def test_matmul_wrong_type_raises(self):
        m = Matrix([[1, 0], [0, 1]])
        with self.assertRaises(TypeError):
            _ = m @ 42


class TestMatrixScalarMul(unittest.TestCase):

    def test_mul_by_2(self):
        m = Matrix([[1, 2], [3, 4]])
        r = m * 2
        self.assertAlmostEqual(r[0, 0].real, 2)
        self.assertAlmostEqual(r[1, 1].real, 8)

    def test_rmul_by_3(self):
        m = Matrix([[1, 0], [0, 1]])
        r = 3 * m
        self.assertAlmostEqual(r[0, 0].real, 3)
        self.assertAlmostEqual(r[1, 1].real, 3)

    def test_mul_by_complex(self):
        m = Matrix([[1, 0], [0, 1]])
        r = m * 1j
        self.assertAlmostEqual(r[0, 0].imag, 1)


class TestMatrixAdd(unittest.TestCase):

    def test_add_two_real_matrices(self):
        A = Matrix([[1, 0], [0, 1]])
        B = Matrix([[0, 1], [1, 0]])
        C = A + B
        self.assertAlmostEqual(C[0, 0].real, 1)
        self.assertAlmostEqual(C[0, 1].real, 1)
        self.assertAlmostEqual(C[1, 0].real, 1)
        self.assertAlmostEqual(C[1, 1].real, 1)

    def test_add_shape_mismatch_raises(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[1, 2, 3]])
        with self.assertRaises(ValueError):
            _ = A + B


class TestMatrixDagger(unittest.TestCase):

    def test_dagger_real_matrix(self):
        # Transpose of real matrix
        m = Matrix([[1, 2], [3, 4]])
        d = m.dagger()
        self.assertAlmostEqual(d[0, 1].real, 3)
        self.assertAlmostEqual(d[1, 0].real, 2)

    def test_dagger_complex(self):
        S = Matrix([[1, 0], [0, 1j]])
        Sd = S.dagger()
        self.assertAlmostEqual(Sd[1, 1].imag, -1)

    def test_dagger_twice_is_original(self):
        m = Matrix([[1+2j, 3], [0, 1j]])
        d = m.dagger().dagger()
        self.assertAlmostEqual(d[0, 0].real, m[0, 0].real)
        self.assertAlmostEqual(d[0, 0].imag, m[0, 0].imag)

    def test_H_dagger_equals_H(self):
        # H is Hermitian
        s = 1 / math.sqrt(2)
        H = Matrix([[s, s], [s, -s]])
        Hd = H.dagger()
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(H[i, j].real, Hd[i, j].real)


class TestMatrixTensor(unittest.TestCase):

    def test_tensor_shape(self):
        A = eye(2)
        B = eye(3)
        C = A.tensor(B)
        self.assertEqual(C.rows, 6)
        self.assertEqual(C.cols, 6)

    def test_tensor_identity(self):
        I4 = eye(2).tensor(eye(2))
        for i in range(4):
            self.assertAlmostEqual(I4[i, i].real, 1)
        self.assertAlmostEqual(I4[0, 1].real, 0)


class TestMatrixIsUnitary(unittest.TestCase):

    def test_identity_is_unitary(self):
        self.assertTrue(eye(2).is_unitary())

    def test_H_is_unitary(self):
        s = 1 / math.sqrt(2)
        H = Matrix([[s, s], [s, -s]])
        self.assertTrue(H.is_unitary())

    def test_non_unitary_detected(self):
        m = Matrix([[2, 0], [0, 1]])
        self.assertFalse(m.is_unitary())

    def test_zero_matrix_not_unitary(self):
        m = Matrix([[0, 0], [0, 0]])
        self.assertFalse(m.is_unitary())


class TestMatrixIndexing(unittest.TestCase):

    def test_getitem(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertAlmostEqual(m[0, 1].real, 2)
        self.assertAlmostEqual(m[1, 0].real, 3)

    def test_setitem(self):
        m = Matrix([[1, 2], [3, 4]])
        m[0, 0] = 99
        self.assertAlmostEqual(m[0, 0].real, 99)

    def test_setitem_complex(self):
        m = Matrix([[0, 0], [0, 0]])
        m[1, 1] = 2+3j
        self.assertAlmostEqual(m[1, 1].real, 2)
        self.assertAlmostEqual(m[1, 1].imag, 3)


class TestMatrixReprAndCopy(unittest.TestCase):

    def test_repr_is_string(self):
        m = eye(2)
        self.assertIsInstance(repr(m), str)
        self.assertIn("Matrix", repr(m))

    def test_copy_is_independent(self):
        m = Matrix([[1, 2], [3, 4]])
        c = m.copy()
        c[0, 0] = 999
        self.assertAlmostEqual(m[0, 0].real, 1)

    def test_copy_same_values(self):
        m = Matrix([[1, 2], [3, 4]])
        c = m.copy()
        self.assertAlmostEqual(c[0, 0].real, 1)
        self.assertAlmostEqual(c[1, 1].real, 4)


# ── Module-level functions ────────────────────────────────────────────────────

class TestEye(unittest.TestCase):

    def test_eye_2_is_identity(self):
        I = eye(2)
        self.assertAlmostEqual(I[0, 0].real, 1)
        self.assertAlmostEqual(I[1, 1].real, 1)
        self.assertAlmostEqual(I[0, 1].real, 0)
        self.assertAlmostEqual(I[1, 0].real, 0)

    def test_eye_4_diagonal(self):
        I = eye(4)
        for i in range(4):
            self.assertAlmostEqual(I[i, i].real, 1)

    def test_eye_4_off_diagonal(self):
        I = eye(4)
        self.assertAlmostEqual(I[0, 3].real, 0)
        self.assertAlmostEqual(I[2, 1].real, 0)

    def test_eye_is_unitary(self):
        self.assertTrue(eye(4).is_unitary())

    def test_eye_shape(self):
        I = eye(5)
        self.assertEqual(I.rows, 5)
        self.assertEqual(I.cols, 5)


class TestZerosVec(unittest.TestCase):

    def test_length(self):
        v = zeros_vec(5)
        self.assertEqual(len(v), 5)

    def test_all_zero(self):
        v = zeros_vec(4)
        for x in v:
            self.assertAlmostEqual(abs(x), 0)

    def test_length_1(self):
        v = zeros_vec(1)
        self.assertEqual(len(v), 1)
        self.assertAlmostEqual(abs(v[0]), 0)


class TestKron(unittest.TestCase):

    def test_shape(self):
        A = eye(2)
        B = eye(3)
        C = kron(A, B)
        self.assertEqual(C.rows, 6)
        self.assertEqual(C.cols, 6)

    def test_identity_kron_identity(self):
        I4 = kron(eye(2), eye(2))
        for i in range(4):
            self.assertAlmostEqual(I4[i, i].real, 1)
        self.assertAlmostEqual(I4[0, 1].real, 0)

    def test_X_kron_I(self):
        X = Matrix([[0, 1], [1, 0]])
        I = eye(2)
        result = kron(X, I)
        self.assertEqual(result.rows, 4)
        # X ⊗ I swaps top half with bottom half
        self.assertAlmostEqual(result[0, 2].real, 1)
        self.assertAlmostEqual(result[2, 0].real, 1)

    def test_I_kron_X(self):
        X = Matrix([[0, 1], [1, 0]])
        I = eye(2)
        result = kron(I, X)
        self.assertEqual(result.rows, 4)
        # I ⊗ X flips pairs within each half
        self.assertAlmostEqual(result[0, 1].real, 1)
        self.assertAlmostEqual(result[1, 0].real, 1)


class TestLiftGate(unittest.TestCase):

    def test_single_qubit_system(self):
        X = Matrix([[0, 1], [1, 0]])
        lifted = lift_gate(X, 0, 1)
        self.assertEqual(lifted.rows, 2)
        self.assertAlmostEqual(lifted[0, 1].real, 1)

    def test_lift_to_qubit_0_of_2(self):
        X = Matrix([[0, 1], [1, 0]])
        lifted = lift_gate(X, 0, 2)
        self.assertEqual(lifted.rows, 4)
        self.assertAlmostEqual(lifted[0, 2].real, 1)

    def test_lift_to_qubit_1_of_2(self):
        X = Matrix([[0, 1], [1, 0]])
        lifted = lift_gate(X, 1, 2)
        self.assertEqual(lifted.rows, 4)
        self.assertAlmostEqual(lifted[0, 1].real, 1)

    def test_lifted_gate_is_unitary(self):
        s = 1 / math.sqrt(2)
        H = Matrix([[s, s], [s, -s]])
        lifted = lift_gate(H, 0, 2)
        self.assertTrue(lifted.is_unitary())


class TestControlledGate(unittest.TestCase):

    def test_controlled_X_matches_CX(self):
        X = Matrix([[0, 1], [1, 0]])
        CU = controlled_gate(X, 0, 1, 2)
        CX = Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ])
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(CU[i, j].real, CX[i, j].real, places=9)

    def test_controlled_gate_is_unitary(self):
        X = Matrix([[0, 1], [1, 0]])
        CU = controlled_gate(X, 0, 1, 2)
        self.assertTrue(CU.is_unitary())

    def test_controlled_Z_is_unitary(self):
        Z = Matrix([[1, 0], [0, -1]])
        CZ = controlled_gate(Z, 0, 1, 2)
        self.assertTrue(CZ.is_unitary())


class TestLiftGateWithProj(unittest.TestCase):

    def test_returns_correct_shape_2_qubits(self):
        p0 = Matrix([[1, 0], [0, 0]])
        I2 = eye(2)
        result = lift_gate_with_proj(p0, 0, I2, 1, 2)
        self.assertEqual(result.rows, 4)
        self.assertEqual(result.cols, 4)

    def test_p0_proj_on_zero_state(self):
        p0 = Matrix([[1, 0], [0, 0]])
        I2 = eye(2)
        result = lift_gate_with_proj(p0, 0, I2, 1, 2)
        v = Vector([1, 0, 0, 0])  # |00>
        out = result @ v
        self.assertAlmostEqual(abs(out[0]), 1)

    def test_p1_proj_on_zero_state_gives_zero(self):
        p1 = Matrix([[0, 0], [0, 1]])
        I2 = eye(2)
        result = lift_gate_with_proj(p1, 0, I2, 1, 2)
        v = Vector([1, 0, 0, 0])  # |00>  — control qubit is 0
        out = result @ v
        self.assertAlmostEqual(abs(out[0]), 0)


if __name__ == "__main__":
    unittest.main()