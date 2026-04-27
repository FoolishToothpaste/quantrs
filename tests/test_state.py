"""
test_state.py
-------------
Unit tests for every function and method in state.py.

Coverage:
    QuantumState : __init__, initialise, initialise_qubit,
                   apply_single, apply_two, apply_three, apply_matrix,
                   measure, measure_all, measure_qubit_no_collapse,
                   probabilities, amplitudes, basis_label, peek,
                   copy, __repr__
    Internal     : _lift_single, _apply_two_qubit_gate,
                   _apply_three_qubit_gate, _permute_state
"""

import unittest
import math
from quantrs.state import (
    QuantumState,
    _lift_single,
    _apply_two_qubit_gate,
    _apply_three_qubit_gate,
    _permute_state,
)
from quantrs.linalg import Vector, eye
from quantrs.gates import H, X, Y, Z, S, CX, CZ, SWAP, CCX


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestQuantumStateInit(unittest.TestCase):

    def test_single_qubit_starts_at_zero(self):
        s = QuantumState(1)
        self.assertAlmostEqual(abs(s._vec[0]), 1.0)
        self.assertAlmostEqual(abs(s._vec[1]), 0.0)

    def test_two_qubit_starts_at_00(self):
        s = QuantumState(2)
        self.assertAlmostEqual(abs(s._vec[0]), 1.0)
        for i in range(1, 4):
            self.assertAlmostEqual(abs(s._vec[i]), 0.0)

    def test_three_qubit_starts_at_000(self):
        s = QuantumState(3)
        self.assertAlmostEqual(abs(s._vec[0]), 1.0)
        for i in range(1, 8):
            self.assertAlmostEqual(abs(s._vec[i]), 0.0)

    def test_dim_is_2_to_n(self):
        s = QuantumState(3)
        self.assertEqual(s.dim, 8)

    def test_zero_qubits_raises(self):
        with self.assertRaises(ValueError):
            QuantumState(0)

    def test_seed_is_accepted(self):
        s = QuantumState(1, seed=42)
        self.assertIsNotNone(s)


# ── initialise ────────────────────────────────────────────────────────────────

class TestQuantumStateInitialise(unittest.TestCase):

    def test_initialise_to_one(self):
        s = QuantumState(1)
        s.initialise([0, 1])
        self.assertAlmostEqual(abs(s._vec[0]), 0)
        self.assertAlmostEqual(abs(s._vec[1]), 1)

    def test_initialise_normalises(self):
        s = QuantumState(1)
        s.initialise([2, 0])
        self.assertAlmostEqual(abs(s._vec[0]), 1)

    def test_initialise_wrong_length_raises(self):
        s = QuantumState(2)
        with self.assertRaises(ValueError):
            s.initialise([1, 0])

    def test_initialise_two_qubit_superposition(self):
        s = QuantumState(2)
        v = 1 / math.sqrt(2)
        s.initialise([v, 0, 0, v])
        self.assertAlmostEqual(abs(s._vec[0]), v)
        self.assertAlmostEqual(abs(s._vec[3]), v)


# ── initialise_qubit ──────────────────────────────────────────────────────────

class TestQuantumStateInitialiseQubit(unittest.TestCase):

    def test_initialise_qubit_to_one(self):
        s = QuantumState(1)
        s.initialise_qubit(0, 0, 1)
        self.assertAlmostEqual(abs(s._vec[0]), 0)
        self.assertAlmostEqual(abs(s._vec[1]), 1)

    def test_initialise_qubit_to_zero(self):
        s = QuantumState(1)
        s.initialise_qubit(0, 1, 0)
        self.assertAlmostEqual(abs(s._vec[0]), 1)

    def test_initialise_qubit_normalises(self):
        s = QuantumState(1)
        s.initialise_qubit(0, 3, 4)
        self.assertAlmostEqual(abs(s._vec[0]), 0.6)
        self.assertAlmostEqual(abs(s._vec[1]), 0.8)

    def test_initialise_qubit_zero_vector_raises(self):
        s = QuantumState(1)
        with self.assertRaises(ValueError):
            s.initialise_qubit(0, 0, 0)


# ── apply_single ──────────────────────────────────────────────────────────────

class TestApplySingle(unittest.TestCase):

    def test_X_flips_zero_to_one(self):
        s = QuantumState(1)
        s.apply_single(X, 0)
        self.assertAlmostEqual(abs(s._vec[0]), 0)
        self.assertAlmostEqual(abs(s._vec[1]), 1)

    def test_X_flips_one_to_zero(self):
        s = QuantumState(1)
        s.apply_single(X, 0)
        s.apply_single(X, 0)
        self.assertAlmostEqual(abs(s._vec[0]), 1)

    def test_H_creates_superposition(self):
        s = QuantumState(1)
        s.apply_single(H, 0)
        v = 1 / math.sqrt(2)
        self.assertAlmostEqual(abs(s._vec[0]), v)
        self.assertAlmostEqual(abs(s._vec[1]), v)

    def test_Z_flips_phase(self):
        s = QuantumState(1)
        s.apply_single(X, 0)   # put in |1>
        s.apply_single(Z, 0)
        self.assertAlmostEqual(s._vec[1].real, -1)

    def test_apply_single_on_qubit_1_of_2(self):
        s = QuantumState(2)
        s.apply_single(X, 1)   # flip qubit 1
        # |00> -> |01>  = index 1
        self.assertAlmostEqual(abs(s._vec[1]), 1)

    def test_apply_single_on_qubit_0_of_2(self):
        s = QuantumState(2)
        s.apply_single(X, 0)
        # |00> -> |10>  = index 2
        self.assertAlmostEqual(abs(s._vec[2]), 1)

    def test_wrong_gate_size_raises(self):
        s = QuantumState(1)
        with self.assertRaises(ValueError):
            s.apply_single(CX, 0)   # CX is 4x4, not 2x2

    def test_norm_preserved_after_H(self):
        s = QuantumState(2)
        s.apply_single(H, 0)
        total = sum(abs(x)**2 for x in s._vec)
        self.assertAlmostEqual(total, 1.0)


# ── apply_two ─────────────────────────────────────────────────────────────────

class TestApplyTwo(unittest.TestCase):

    def test_CX_flips_target_when_control_one(self):
        s = QuantumState(2)
        s.apply_single(X, 0)      # |00> -> |10>
        s.apply_two(CX, 0, 1)    # |10> -> |11>
        self.assertAlmostEqual(abs(s._vec[3]), 1)

    def test_CX_does_nothing_when_control_zero(self):
        s = QuantumState(2)
        s.apply_two(CX, 0, 1)
        self.assertAlmostEqual(abs(s._vec[0]), 1)

    def test_SWAP_exchanges_qubits(self):
        s = QuantumState(2)
        s.apply_single(X, 0)     # |10>
        s.apply_two(SWAP, 0, 1)  # -> |01>
        self.assertAlmostEqual(abs(s._vec[1]), 1)

    def test_CZ_flips_phase_of_11(self):
        s = QuantumState(2)
        s.apply_single(X, 0)
        s.apply_single(X, 1)     # |11>
        s.apply_two(CZ, 0, 1)
        self.assertAlmostEqual(s._vec[3].real, -1)

    def test_apply_two_on_non_adjacent_qubits(self):
        s = QuantumState(3)
        s.apply_single(X, 0)     # |100>
        s.apply_two(CX, 0, 2)   # -> |101>  (control=q0, target=q2)
        self.assertAlmostEqual(abs(s._vec[5]), 1)   # |101> = index 5

    def test_wrong_gate_size_raises(self):
        s = QuantumState(2)
        with self.assertRaises(ValueError):
            s.apply_two(X, 0, 1)   # X is 2x2, not 4x4

    def test_norm_preserved(self):
        s = QuantumState(2)
        s.apply_single(H, 0)
        s.apply_two(CX, 0, 1)
        total = sum(abs(x)**2 for x in s._vec)
        self.assertAlmostEqual(total, 1.0)


# ── apply_three ───────────────────────────────────────────────────────────────

class TestApplyThree(unittest.TestCase):

    def test_CCX_flips_target_when_both_controls_one(self):
        s = QuantumState(3)
        s.apply_single(X, 0)
        s.apply_single(X, 1)      # |110>
        s.apply_three(CCX, 0, 1, 2)  # -> |111>
        self.assertAlmostEqual(abs(s._vec[7]), 1)

    def test_CCX_no_flip_one_control_zero(self):
        s = QuantumState(3)
        s.apply_single(X, 0)      # |100>
        s.apply_three(CCX, 0, 1, 2)
        self.assertAlmostEqual(abs(s._vec[4]), 1)  # stays |100>

    def test_CCX_no_flip_both_controls_zero(self):
        s = QuantumState(3)
        s.apply_three(CCX, 0, 1, 2)
        self.assertAlmostEqual(abs(s._vec[0]), 1)  # stays |000>

    def test_wrong_gate_size_raises(self):
        s = QuantumState(3)
        with self.assertRaises(ValueError):
            s.apply_three(CX, 0, 1, 2)   # CX is 4x4, not 8x8

    def test_norm_preserved(self):
        s = QuantumState(3)
        s.apply_single(X, 0)
        s.apply_single(X, 1)
        s.apply_three(CCX, 0, 1, 2)
        total = sum(abs(x)**2 for x in s._vec)
        self.assertAlmostEqual(total, 1.0)


# ── apply_matrix ──────────────────────────────────────────────────────────────

class TestApplyMatrix(unittest.TestCase):

    def test_apply_identity(self):
        s = QuantumState(1)
        s.apply_matrix(eye(2))
        self.assertAlmostEqual(abs(s._vec[0]), 1)

    def test_apply_X_full(self):
        from quantrs.linalg import Matrix
        s = QuantumState(1)
        s.apply_matrix(X)
        self.assertAlmostEqual(abs(s._vec[1]), 1)

    def test_wrong_size_raises(self):
        s = QuantumState(2)
        with self.assertRaises(ValueError):
            s.apply_matrix(X)   # X is 2x2, state is 4-dim


# ── measure ───────────────────────────────────────────────────────────────────

class TestMeasure(unittest.TestCase):

    def test_measure_zero_state_always_zero(self):
        for trial in range(20):
            s = QuantumState(1, seed=trial)
            self.assertEqual(s.measure(0), 0)

    def test_measure_one_state_always_one(self):
        for trial in range(20):
            s = QuantumState(1, seed=trial)
            s.apply_single(X, 0)
            self.assertEqual(s.measure(0), 1)

    def test_measure_returns_0_or_1(self):
        for trial in range(50):
            s = QuantumState(1, seed=trial)
            s.apply_single(H, 0)
            result = s.measure(0)
            self.assertIn(result, (0, 1))

    def test_measure_collapses_state(self):
        s = QuantumState(1, seed=0)
        s.apply_single(H, 0)
        outcome = s.measure(0)
        # After collapse, prob of same outcome should be 1
        p0, p1 = s.measure_qubit_no_collapse(0)
        if outcome == 0:
            self.assertAlmostEqual(p0, 1.0)
        else:
            self.assertAlmostEqual(p1, 1.0)

    def test_measure_second_qubit_of_two(self):
        for trial in range(10):
            s = QuantumState(2, seed=trial)
            # q0=|0>, q1=|1>
            s.apply_single(X, 1)
            result = s.measure(1)
            self.assertEqual(result, 1)

    def test_measure_first_qubit_of_two(self):
        for trial in range(10):
            s = QuantumState(2, seed=trial)
            s.apply_single(X, 0)
            result = s.measure(0)
            self.assertEqual(result, 1)

    def test_norm_preserved_after_measure(self):
        s = QuantumState(1, seed=0)
        s.apply_single(H, 0)
        s.measure(0)
        total = sum(abs(x)**2 for x in s._vec)
        self.assertAlmostEqual(total, 1.0)


# ── measure_all ───────────────────────────────────────────────────────────────

class TestMeasureAll(unittest.TestCase):

    def test_returns_correct_length(self):
        s = QuantumState(3, seed=0)
        result = s.measure_all()
        self.assertEqual(len(result), 3)

    def test_all_zeros_state(self):
        for trial in range(10):
            s = QuantumState(2, seed=trial)
            result = s.measure_all()
            self.assertEqual(result, "00")

    def test_returns_binary_string(self):
        s = QuantumState(3, seed=42)
        result = s.measure_all()
        self.assertTrue(all(c in "01" for c in result))

    def test_measure_all_bell_state(self):
        for trial in range(20):
            s = QuantumState(2, seed=trial)
            s.apply_single(H, 0)
            s.apply_two(CX, 0, 1)
            result = s.measure_all()
            self.assertIn(result, ("00", "11"))

    def test_measure_all_collapses_state(self):
        s = QuantumState(2, seed=0)
        s.apply_single(H, 0)
        s.apply_two(CX, 0, 1)
        result = s.measure_all()
        total = sum(abs(x)**2 for x in s._vec)
        self.assertAlmostEqual(total, 1.0)


# ── measure_qubit_no_collapse ─────────────────────────────────────────────────

class TestMeasureQubitNoCollapse(unittest.TestCase):

    def test_zero_state_p0_is_1(self):
        s = QuantumState(1)
        p0, p1 = s.measure_qubit_no_collapse(0)
        self.assertAlmostEqual(p0, 1.0)
        self.assertAlmostEqual(p1, 0.0)

    def test_one_state_p1_is_1(self):
        s = QuantumState(1)
        s.apply_single(X, 0)
        p0, p1 = s.measure_qubit_no_collapse(0)
        self.assertAlmostEqual(p0, 0.0)
        self.assertAlmostEqual(p1, 1.0)

    def test_superposition_equal_probs(self):
        s = QuantumState(1)
        s.apply_single(H, 0)
        p0, p1 = s.measure_qubit_no_collapse(0)
        self.assertAlmostEqual(p0, 0.5)
        self.assertAlmostEqual(p1, 0.5)

    def test_does_not_collapse(self):
        s = QuantumState(1)
        s.apply_single(H, 0)
        p0_before, _ = s.measure_qubit_no_collapse(0)
        p0_after, _  = s.measure_qubit_no_collapse(0)
        self.assertAlmostEqual(p0_before, p0_after)

    def test_probs_sum_to_one(self):
        s = QuantumState(1)
        s.apply_single(H, 0)
        p0, p1 = s.measure_qubit_no_collapse(0)
        self.assertAlmostEqual(p0 + p1, 1.0)


# ── probabilities ─────────────────────────────────────────────────────────────

class TestProbabilities(unittest.TestCase):

    def test_zero_state_probabilities(self):
        s = QuantumState(2)
        probs = s.probabilities()
        self.assertAlmostEqual(probs[0], 1.0)
        for i in range(1, 4):
            self.assertAlmostEqual(probs[i], 0.0)

    def test_probabilities_sum_to_one(self):
        s = QuantumState(2)
        s.apply_single(H, 0)
        s.apply_two(CX, 0, 1)
        self.assertAlmostEqual(sum(s.probabilities()), 1.0)

    def test_probabilities_length(self):
        s = QuantumState(3)
        self.assertEqual(len(s.probabilities()), 8)

    def test_probabilities_are_non_negative(self):
        s = QuantumState(2)
        s.apply_single(H, 0)
        for p in s.probabilities():
            self.assertGreaterEqual(p, 0)


# ── amplitudes ────────────────────────────────────────────────────────────────

class TestAmplitudes(unittest.TestCase):

    def test_returns_list(self):
        s = QuantumState(1)
        self.assertIsInstance(s.amplitudes(), list)

    def test_length(self):
        s = QuantumState(3)
        self.assertEqual(len(s.amplitudes()), 8)

    def test_initial_amplitude(self):
        s = QuantumState(1)
        amps = s.amplitudes()
        self.assertAlmostEqual(amps[0].real, 1.0)
        self.assertAlmostEqual(amps[1].real, 0.0)

    def test_amplitudes_after_H(self):
        s = QuantumState(1)
        s.apply_single(H, 0)
        amps = s.amplitudes()
        v = 1 / math.sqrt(2)
        self.assertAlmostEqual(amps[0].real, v)
        self.assertAlmostEqual(amps[1].real, v)


# ── basis_label ───────────────────────────────────────────────────────────────

class TestBasisLabel(unittest.TestCase):

    def test_label_0_for_1_qubit(self):
        s = QuantumState(1)
        self.assertEqual(s.basis_label(0), "0")

    def test_label_1_for_1_qubit(self):
        s = QuantumState(1)
        self.assertEqual(s.basis_label(1), "1")

    def test_label_0_for_2_qubits(self):
        s = QuantumState(2)
        self.assertEqual(s.basis_label(0), "00")

    def test_label_3_for_2_qubits(self):
        s = QuantumState(2)
        self.assertEqual(s.basis_label(3), "11")

    def test_label_5_for_3_qubits(self):
        s = QuantumState(3)
        self.assertEqual(s.basis_label(5), "101")


# ── peek ──────────────────────────────────────────────────────────────────────

class TestPeek(unittest.TestCase):

    def test_peek_zero_state(self):
        s = QuantumState(1)
        result = s.peek()
        self.assertIn("0", result)
        self.assertAlmostEqual(result["0"], 1.0)

    def test_peek_superposition(self):
        s = QuantumState(1)
        s.apply_single(H, 0)
        result = s.peek()
        self.assertIn("0", result)
        self.assertIn("1", result)

    def test_peek_does_not_collapse(self):
        s = QuantumState(1)
        s.apply_single(H, 0)
        before = s.peek()
        after = s.peek()
        self.assertAlmostEqual(before["0"], after["0"])

    def test_peek_filters_near_zero(self):
        s = QuantumState(1)
        result = s.peek()
        self.assertNotIn("1", result)


# ── copy ──────────────────────────────────────────────────────────────────────

class TestCopy(unittest.TestCase):

    def test_copy_same_amplitudes(self):
        s = QuantumState(2)
        s.apply_single(H, 0)
        c = s.copy()
        for i in range(4):
            self.assertAlmostEqual(abs(s._vec[i]), abs(c._vec[i]))

    def test_copy_is_independent(self):
        s = QuantumState(1)
        c = s.copy()
        c.apply_single(X, 0)
        self.assertAlmostEqual(abs(s._vec[0]), 1.0)  # original unchanged

    def test_copy_num_qubits(self):
        s = QuantumState(3)
        c = s.copy()
        self.assertEqual(c.num_qubits, 3)


# ── __repr__ ──────────────────────────────────────────────────────────────────

class TestRepr(unittest.TestCase):

    def test_repr_zero_state(self):
        s = QuantumState(1)
        r = repr(s)
        self.assertIn("|0>", r)

    def test_repr_one_state(self):
        s = QuantumState(1)
        s.apply_single(X, 0)
        r = repr(s)
        self.assertIn("|1>", r)

    def test_repr_returns_string(self):
        s = QuantumState(2)
        self.assertIsInstance(repr(s), str)


# ── Internal helper functions ─────────────────────────────────────────────────

class TestLiftSingle(unittest.TestCase):

    def test_lifts_to_correct_size(self):
        result = _lift_single(X, 0, 2)
        self.assertEqual(result.rows, 4)

    def test_lift_X_qubit_0_of_1(self):
        result = _lift_single(X, 0, 1)
        self.assertAlmostEqual(result[0, 1].real, 1)

    def test_lift_X_qubit_0_of_2(self):
        result = _lift_single(X, 0, 2)
        self.assertAlmostEqual(result[0, 2].real, 1)

    def test_lift_X_qubit_1_of_2(self):
        result = _lift_single(X, 1, 2)
        self.assertAlmostEqual(result[0, 1].real, 1)


class TestApplyTwoQubitGate(unittest.TestCase):

    def test_CX_on_10(self):
        # |10> -> |11>
        v = Vector([0, 0, 1, 0])
        result = _apply_two_qubit_gate(CX, 0, 1, 2, v)
        self.assertAlmostEqual(abs(result[3]), 1)

    def test_CX_non_adjacent_qubits(self):
        # 3 qubits: CX(0, 2), |100> -> |101>
        v = Vector([0, 0, 0, 0, 1, 0, 0, 0])   # index 4 = |100>
        result = _apply_two_qubit_gate(CX, 0, 2, 3, v)
        self.assertAlmostEqual(abs(result[5]), 1)   # |101> = index 5


class TestApplyThreeQubitGate(unittest.TestCase):

    def test_CCX_on_110(self):
        v = Vector([0, 0, 0, 0, 0, 0, 1, 0])   # index 6 = |110>
        result = _apply_three_qubit_gate(CCX, 0, 1, 2, 3, v)
        self.assertAlmostEqual(abs(result[7]), 1)   # |111>


class TestPermuteState(unittest.TestCase):

    def test_identity_permutation(self):
        v = Vector([1, 0, 0, 0])
        result = _permute_state(v, [0, 1], 2)
        self.assertAlmostEqual(abs(result[0]), 1)

    def test_swap_permutation(self):
        # Swap q0 and q1: |10> (index 2) -> |01> (index 1)
        v = Vector([0, 0, 1, 0])   # |10>
        result = _permute_state(v, [1, 0], 2)
        self.assertAlmostEqual(abs(result[1]), 1)   # |01>

    def test_permute_preserves_norm(self):
        v = Vector([0.5, 0.5, 0.5, 0.5])
        result = _permute_state(v, [1, 0], 2)
        total = sum(abs(x)**2 for x in result)
        self.assertAlmostEqual(total, 1.0)


if __name__ == "__main__":
    unittest.main()