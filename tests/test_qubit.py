"""
test_qubit.py
-------------
Unit tests for every function and method in qubit.py.

Coverage:
    Qubit : __init__,
            h, x, y, z, s, sdg, t, tdg, sx, id,
            rx, ry, rz, p, u,
            measure, run, statevector,
            draw, reset, barrier,
            __repr__
"""

import unittest
import math
from quantrs import Qubit


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestQubitInit(unittest.TestCase):

    def test_circuit_has_one_qubit(self):
        q = Qubit()
        self.assertEqual(q.circuit.num_qubits, 1)

    def test_circuit_has_one_clbit(self):
        q = Qubit()
        self.assertEqual(q.circuit.num_clbits, 1)

    def test_measured_flag_starts_false(self):
        q = Qubit()
        self.assertFalse(q._measured)

    def test_custom_name(self):
        q = Qubit(name="alice")
        self.assertEqual(q.circuit.name, "alice")

    def test_starts_with_no_instructions(self):
        q = Qubit()
        self.assertEqual(len(q.circuit._instructions), 0)


# ── Single-qubit gate methods ─────────────────────────────────────────────────

class TestQubitGates(unittest.TestCase):

    def _check(self, method_name, *args):
        """Apply a gate, check it returns self and adds one instruction."""
        q = Qubit()
        method = getattr(q, method_name)
        result = method(*args)
        self.assertIs(result, q)
        self.assertEqual(q.circuit._instructions[0].name, method_name)

    def test_h_returns_self_and_adds_instruction(self):
        self._check("h")

    def test_x_returns_self_and_adds_instruction(self):
        self._check("x")

    def test_y_returns_self_and_adds_instruction(self):
        self._check("y")

    def test_z_returns_self_and_adds_instruction(self):
        self._check("z")

    def test_s_returns_self_and_adds_instruction(self):
        self._check("s")

    def test_sdg_returns_self_and_adds_instruction(self):
        self._check("sdg")

    def test_t_returns_self_and_adds_instruction(self):
        self._check("t")

    def test_tdg_returns_self_and_adds_instruction(self):
        self._check("tdg")

    def test_sx_returns_self_and_adds_instruction(self):
        self._check("sx")

    def test_id_returns_self_and_adds_instruction(self):
        self._check("id")

    def test_rx_returns_self_and_adds_instruction(self):
        q = Qubit()
        result = q.rx(0.5)
        self.assertIs(result, q)
        self.assertEqual(q.circuit._instructions[0].name, "rx")
        self.assertAlmostEqual(q.circuit._instructions[0].params[0], 0.5)

    def test_ry_returns_self_and_adds_instruction(self):
        q = Qubit()
        result = q.ry(1.0)
        self.assertIs(result, q)
        self.assertEqual(q.circuit._instructions[0].name, "ry")

    def test_rz_returns_self_and_adds_instruction(self):
        q = Qubit()
        result = q.rz(0.3)
        self.assertIs(result, q)
        self.assertEqual(q.circuit._instructions[0].name, "rz")

    def test_p_returns_self_and_adds_instruction(self):
        q = Qubit()
        result = q.p(0.7)
        self.assertIs(result, q)
        self.assertEqual(q.circuit._instructions[0].name, "p")

    def test_u_returns_self_and_adds_instruction(self):
        q = Qubit()
        result = q.u(0.1, 0.2, 0.3)
        self.assertIs(result, q)
        self.assertEqual(q.circuit._instructions[0].name, "u")
        self.assertEqual(len(q.circuit._instructions[0].params), 3)

    def test_chain_multiple_gates(self):
        q = Qubit()
        q.h().x().z().s().t()
        self.assertEqual(len(q.circuit._instructions), 5)

    def test_chain_returns_same_qubit(self):
        q = Qubit()
        result = q.h().x().rz(0.5)
        self.assertIs(result, q)


# ── measure ───────────────────────────────────────────────────────────────────

class TestQubitMeasure(unittest.TestCase):

    def test_measure_adds_instruction(self):
        q = Qubit()
        q.measure()
        names = [i.name for i in q.circuit._instructions]
        self.assertIn("measure", names)

    def test_measure_sets_flag(self):
        q = Qubit()
        q.measure()
        self.assertTrue(q._measured)

    def test_measure_returns_self(self):
        q = Qubit()
        result = q.measure()
        self.assertIs(result, q)

    def test_measure_only_added_once_when_chained(self):
        q = Qubit()
        q.h().measure()
        measure_instrs = [i for i in q.circuit._instructions if i.name == "measure"]
        self.assertEqual(len(measure_instrs), 1)


# ── run ───────────────────────────────────────────────────────────────────────

class TestQubitRun(unittest.TestCase):

    def test_run_returns_measurement_result(self):
        from quantrs.result import MeasurementResult
        q = Qubit()
        q.measure()
        result = q.run(shots=10)
        self.assertIsInstance(result, MeasurementResult)

    def test_run_correct_shot_count(self):
        q = Qubit()
        q.measure()
        result = q.run(shots=100)
        self.assertEqual(sum(result.counts.values()), 100)

    def test_run_zero_state_always_zero(self):
        q = Qubit()
        q.measure()
        result = q.run(shots=50, seed=0)
        self.assertEqual(result.counts.get("0", 0), 50)

    def test_run_x_gate_always_one(self):
        q = Qubit()
        q.x().measure()
        result = q.run(shots=50, seed=0)
        self.assertEqual(result.counts.get("1", 0), 50)

    def test_run_auto_adds_measure(self):
        q = Qubit()
        q.h()
        result = q.run(shots=100, seed=42)
        self.assertEqual(sum(result.counts.values()), 100)

    def test_run_seed_reproducible(self):
        q1 = Qubit()
        q1.h().measure()
        r1 = q1.run(shots=100, seed=5)

        q2 = Qubit()
        q2.h().measure()
        r2 = q2.run(shots=100, seed=5)

        self.assertEqual(r1.counts, r2.counts)

    def test_run_h_gives_both_outcomes(self):
        q = Qubit()
        q.h().measure()
        result = q.run(shots=1000, seed=0)
        self.assertIn("0", result.counts)
        self.assertIn("1", result.counts)


# ── statevector ───────────────────────────────────────────────────────────────

class TestQubitStatevector(unittest.TestCase):

    def test_initial_statevector(self):
        q = Qubit()
        sv = q.statevector()
        self.assertAlmostEqual(sv[0].real, 1.0)
        self.assertAlmostEqual(sv[1].real, 0.0)

    def test_H_statevector(self):
        q = Qubit()
        q.h()
        sv = q.statevector()
        v = 1 / math.sqrt(2)
        self.assertAlmostEqual(abs(sv[0]), v)
        self.assertAlmostEqual(abs(sv[1]), v)

    def test_X_statevector(self):
        q = Qubit()
        q.x()
        sv = q.statevector()
        self.assertAlmostEqual(abs(sv[0]), 0)
        self.assertAlmostEqual(abs(sv[1]), 1)

    def test_statevector_length(self):
        q = Qubit()
        sv = q.statevector()
        self.assertEqual(len(sv), 2)

    def test_statevector_raises_after_measure(self):
        q = Qubit()
        q.h().measure()
        with self.assertRaises(RuntimeError):
            q.statevector()

    def test_Z_on_zero_statevector_unchanged(self):
        q = Qubit()
        q.z()
        sv = q.statevector()
        self.assertAlmostEqual(sv[0].real, 1.0)

    def test_norm_of_statevector_is_one(self):
        q = Qubit()
        q.rx(0.7).ry(1.2)
        sv = q.statevector()
        norm_sq = sum(abs(x)**2 for x in sv)
        self.assertAlmostEqual(norm_sq, 1.0)


# ── draw ──────────────────────────────────────────────────────────────────────

class TestQubitDraw(unittest.TestCase):

    def test_draw_returns_string(self):
        q = Qubit()
        q.h()
        result = q.draw()
        self.assertIsInstance(result, str)

    def test_draw_contains_qubit_label(self):
        q = Qubit()
        q.h()
        result = q.draw()
        self.assertIn("q0", result)

    def test_draw_contains_gate_symbol(self):
        q = Qubit()
        q.h()
        result = q.draw()
        self.assertIn("H", result)


# ── reset ─────────────────────────────────────────────────────────────────────

class TestQubitReset(unittest.TestCase):

    def test_reset_adds_instruction(self):
        q = Qubit()
        q.reset()
        self.assertEqual(q.circuit._instructions[0].name, "reset")

    def test_reset_returns_self(self):
        q = Qubit()
        self.assertIs(q.reset(), q)

    def test_reset_after_x_returns_to_zero(self):
        q = Qubit()
        q.x().reset().measure()
        result = q.run(shots=50, seed=0)
        self.assertEqual(result.counts.get("0", 0), 50)


# ── barrier ───────────────────────────────────────────────────────────────────

class TestQubitBarrier(unittest.TestCase):

    def test_barrier_adds_instruction(self):
        q = Qubit()
        q.barrier()
        self.assertEqual(q.circuit._instructions[0].name, "barrier")

    def test_barrier_returns_self(self):
        q = Qubit()
        self.assertIs(q.barrier(), q)

    def test_barrier_does_not_affect_result(self):
        q = Qubit()
        q.x().barrier().measure()
        result = q.run(shots=50, seed=0)
        self.assertEqual(result.counts.get("1", 0), 50)


# ── __repr__ ──────────────────────────────────────────────────────────────────

class TestQubitRepr(unittest.TestCase):

    def test_repr_returns_string(self):
        q = Qubit()
        self.assertIsInstance(repr(q), str)

    def test_repr_contains_qubit(self):
        q = Qubit()
        self.assertIn("Qubit", repr(q))

    def test_repr_contains_measured_flag(self):
        q = Qubit()
        self.assertIn("measured", repr(q))

    def test_repr_reflects_measured_true(self):
        q = Qubit()
        q.measure()
        self.assertIn("True", repr(q))

    def test_repr_contains_gate_info(self):
        q = Qubit()
        q.h()
        self.assertIn("h", repr(q))


if __name__ == "__main__":
    unittest.main()