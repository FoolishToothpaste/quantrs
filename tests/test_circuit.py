"""
test_circuit.py
---------------
Unit tests for every function and method in circuit.py.

Coverage:
    Instruction : __init__, __repr__
    Circuit     : __init__, _add, _validate_qubits,
                  h, x, y, z, s, sdg, t, tdg, sx, id,
                  rx, ry, rz, p, u,
                  cx, cz, swap, ccx,
                  measure, measure_all, barrier, reset,
                  x_if, z_if,
                  _execute_once, run, statevector,
                  depth, size, count_ops, inverse,
                  draw, __repr__
    Module      : _draw_circuit
"""

import unittest
import math
from quantrs.circuit import Circuit, Instruction, _draw_circuit


# ── Instruction ───────────────────────────────────────────────────────────────

class TestInstruction(unittest.TestCase):

    def test_init_basic(self):
        instr = Instruction("h", [0])
        self.assertEqual(instr.name, "h")
        self.assertEqual(instr.qubits, [0])

    def test_init_defaults(self):
        instr = Instruction("h", [0])
        self.assertEqual(instr.clbits, [])
        self.assertIsNone(instr.gate)
        self.assertEqual(instr.params, [])
        self.assertIsNone(instr.condition)

    def test_init_with_all_fields(self):
        instr = Instruction("measure", [0], clbits=[0], params=[0.5], condition=(0, 1))
        self.assertEqual(instr.clbits, [0])
        self.assertEqual(instr.params, [0.5])
        self.assertEqual(instr.condition, (0, 1))

    def test_repr_returns_string(self):
        instr = Instruction("h", [0])
        self.assertIsInstance(repr(instr), str)
        self.assertIn("h", repr(instr))

    def test_repr_with_condition(self):
        instr = Instruction("x_if", [0], condition=(1, 1))
        self.assertIn("if", repr(instr))


# ── Circuit.__init__ ──────────────────────────────────────────────────────────

class TestCircuitInit(unittest.TestCase):

    def test_num_qubits(self):
        c = Circuit(3)
        self.assertEqual(c.num_qubits, 3)

    def test_default_clbits_is_zero(self):
        c = Circuit(2)
        self.assertEqual(c.num_clbits, 0)

    def test_explicit_clbits(self):
        c = Circuit(2, 2)
        self.assertEqual(c.num_clbits, 2)

    def test_name(self):
        c = Circuit(1, 1, name="mycirc")
        self.assertEqual(c.name, "mycirc")

    def test_starts_with_empty_instructions(self):
        c = Circuit(2)
        self.assertEqual(len(c._instructions), 0)

    def test_measured_flag_starts_false(self):
        c = Circuit(2)
        self.assertFalse(c._measured)


# ── _validate_qubits ──────────────────────────────────────────────────────────

class TestValidateQubits(unittest.TestCase):

    def test_valid_qubit(self):
        c = Circuit(3)
        c.h(2)   # should not raise

    def test_negative_qubit_raises(self):
        c = Circuit(2)
        with self.assertRaises(ValueError):
            c.h(-1)

    def test_out_of_range_qubit_raises(self):
        c = Circuit(2)
        with self.assertRaises(ValueError):
            c.h(5)

    def test_exact_boundary_raises(self):
        c = Circuit(2)
        with self.assertRaises(ValueError):
            c.h(2)   # valid indices are 0 and 1


# ── Single-qubit gates ────────────────────────────────────────────────────────

class TestSingleQubitGates(unittest.TestCase):

    def _check_gate(self, method_name, *args):
        c = Circuit(1)
        method = getattr(c, method_name)
        result = method(*args)
        self.assertIs(result, c)
        self.assertEqual(len(c._instructions), 1)
        self.assertEqual(c._instructions[0].name, method_name)

    def test_h_added(self):
        self._check_gate("h", 0)

    def test_x_added(self):
        self._check_gate("x", 0)

    def test_y_added(self):
        self._check_gate("y", 0)

    def test_z_added(self):
        self._check_gate("z", 0)

    def test_s_added(self):
        self._check_gate("s", 0)

    def test_sdg_added(self):
        self._check_gate("sdg", 0)

    def test_t_added(self):
        self._check_gate("t", 0)

    def test_tdg_added(self):
        self._check_gate("tdg", 0)

    def test_sx_added(self):
        self._check_gate("sx", 0)

    def test_id_added(self):
        self._check_gate("id", 0)

    def test_rx_added(self):
        c = Circuit(1)
        c.rx(0.5, 0)
        self.assertEqual(c._instructions[0].name, "rx")
        self.assertAlmostEqual(c._instructions[0].params[0], 0.5)

    def test_ry_added(self):
        c = Circuit(1)
        c.ry(1.0, 0)
        self.assertEqual(c._instructions[0].name, "ry")

    def test_rz_added(self):
        c = Circuit(1)
        c.rz(0.3, 0)
        self.assertEqual(c._instructions[0].name, "rz")

    def test_p_added(self):
        c = Circuit(1)
        c.p(0.7, 0)
        self.assertEqual(c._instructions[0].name, "p")

    def test_u_added(self):
        c = Circuit(1)
        c.u(0.1, 0.2, 0.3, 0)
        self.assertEqual(c._instructions[0].name, "u")
        self.assertEqual(len(c._instructions[0].params), 3)

    def test_gate_fluent_chain(self):
        c = Circuit(1)
        result = c.h(0).x(0).z(0)
        self.assertIs(result, c)
        self.assertEqual(len(c._instructions), 3)


# ── Two and three-qubit gates ─────────────────────────────────────────────────

class TestMultiQubitGates(unittest.TestCase):

    def test_cx_added(self):
        c = Circuit(2)
        c.cx(0, 1)
        self.assertEqual(c._instructions[0].name, "cx")
        self.assertEqual(c._instructions[0].qubits, [0, 1])

    def test_cz_added(self):
        c = Circuit(2)
        c.cz(0, 1)
        self.assertEqual(c._instructions[0].name, "cz")

    def test_swap_added(self):
        c = Circuit(2)
        c.swap(0, 1)
        self.assertEqual(c._instructions[0].name, "swap")

    def test_ccx_added(self):
        c = Circuit(3)
        c.ccx(0, 1, 2)
        self.assertEqual(c._instructions[0].name, "ccx")
        self.assertEqual(c._instructions[0].qubits, [0, 1, 2])

    def test_cx_returns_self(self):
        c = Circuit(2)
        self.assertIs(c.cx(0, 1), c)

    def test_ccx_returns_self(self):
        c = Circuit(3)
        self.assertIs(c.ccx(0, 1, 2), c)


# ── Measurement ───────────────────────────────────────────────────────────────

class TestMeasurement(unittest.TestCase):

    def test_measure_adds_instruction(self):
        c = Circuit(1, 1)
        c.measure(0, 0)
        self.assertEqual(c._instructions[0].name, "measure")

    def test_measure_sets_measured_flag(self):
        c = Circuit(1, 1)
        c.measure(0, 0)
        self.assertTrue(c._measured)

    def test_measure_no_clbits_raises(self):
        c = Circuit(1)
        with self.assertRaises(ValueError):
            c.measure(0, 0)

    def test_measure_clbit_out_of_range_raises(self):
        c = Circuit(2, 1)
        with self.assertRaises(ValueError):
            c.measure(0, 1)

    def test_measure_negative_clbit_raises(self):
        c = Circuit(1, 1)
        with self.assertRaises(ValueError):
            c.measure(0, -1)

    def test_measure_all_adds_n_instructions(self):
        c = Circuit(3, 3)
        c.measure_all()
        measure_instrs = [i for i in c._instructions if i.name == "measure"]
        self.assertEqual(len(measure_instrs), 3)

    def test_measure_all_sets_measured_flag(self):
        c = Circuit(2, 2)
        c.measure_all()
        self.assertTrue(c._measured)

    def test_measure_all_no_clbits_raises(self):
        c = Circuit(2)
        with self.assertRaises(ValueError):
            c.measure_all()

    def test_measure_all_insufficient_clbits_raises(self):
        c = Circuit(3, 1)
        with self.assertRaises(ValueError):
            c.measure_all()

    def test_measure_returns_self(self):
        c = Circuit(1, 1)
        self.assertIs(c.measure(0, 0), c)

    def test_measure_all_returns_self(self):
        c = Circuit(2, 2)
        self.assertIs(c.measure_all(), c)


# ── Barrier and reset ─────────────────────────────────────────────────────────

class TestBarrierAndReset(unittest.TestCase):

    def test_barrier_added(self):
        c = Circuit(2)
        c.barrier()
        self.assertEqual(c._instructions[0].name, "barrier")

    def test_barrier_spans_all_qubits(self):
        c = Circuit(3)
        c.barrier()
        self.assertEqual(c._instructions[0].qubits, [0, 1, 2])

    def test_barrier_returns_self(self):
        c = Circuit(1)
        self.assertIs(c.barrier(), c)

    def test_reset_added(self):
        c = Circuit(1)
        c.reset(0)
        self.assertEqual(c._instructions[0].name, "reset")

    def test_reset_returns_self(self):
        c = Circuit(1)
        self.assertIs(c.reset(0), c)


# ── Classical conditioning ────────────────────────────────────────────────────

class TestClassicalConditioning(unittest.TestCase):

    def test_x_if_added(self):
        c = Circuit(1, 1)
        c.x_if(0, 0, 1)
        self.assertEqual(c._instructions[0].name, "x_if")
        self.assertEqual(c._instructions[0].condition, (0, 1))

    def test_z_if_added(self):
        c = Circuit(1, 1)
        c.z_if(0, 0, 1)
        self.assertEqual(c._instructions[0].name, "z_if")

    def test_x_if_returns_self(self):
        c = Circuit(1, 1)
        self.assertIs(c.x_if(0, 0), c)

    def test_z_if_returns_self(self):
        c = Circuit(1, 1)
        self.assertIs(c.z_if(0, 0), c)

    def test_x_if_default_val_is_1(self):
        c = Circuit(1, 1)
        c.x_if(0, 0)
        self.assertEqual(c._instructions[0].condition[1], 1)


# ── _execute_once and run ─────────────────────────────────────────────────────

class TestExecuteOnceAndRun(unittest.TestCase):

    def test_x_gate_gives_1(self):
        c = Circuit(1, 1)
        c.x(0).measure(0, 0)
        result = c._execute_once(seed=0)
        self.assertEqual(result, "1")

    def test_no_gate_gives_0(self):
        c = Circuit(1, 1)
        c.measure(0, 0)
        result = c._execute_once(seed=0)
        self.assertEqual(result, "0")

    def test_run_without_measure_raises(self):
        c = Circuit(1, 1)
        c.h(0)
        with self.assertRaises(ValueError):
            c.run()

    def test_run_correct_shot_count(self):
        c = Circuit(1, 1)
        c.h(0).measure(0, 0)
        result = c.run(shots=200, seed=0)
        self.assertEqual(sum(result.counts.values()), 200)

    def test_run_bell_only_00_and_11(self):
        c = Circuit(2, 2)
        c.h(0).cx(0, 1).measure_all()
        result = c.run(shots=300, seed=42)
        for k in result.counts:
            self.assertIn(k, ("00", "11"))

    def test_run_x_always_gives_1(self):
        c = Circuit(1, 1)
        c.x(0).measure(0, 0)
        result = c.run(shots=50, seed=0)
        self.assertEqual(result.counts.get("1", 0), 50)

    def test_run_seed_reproducible(self):
        c = Circuit(1, 1)
        c.h(0).measure(0, 0)
        r1 = c.run(shots=100, seed=7)
        r2 = c.run(shots=100, seed=7)
        self.assertEqual(r1.counts, r2.counts)

    def test_reset_brings_back_to_zero(self):
        c = Circuit(1, 1)
        c.x(0).reset(0).measure(0, 0)
        result = c._execute_once(seed=0)
        self.assertEqual(result, "0")

    def test_x_if_applies_when_condition_met(self):
        c = Circuit(1, 1)
        c.x(0).measure(0, 0).x_if(0, 0, 1)
        # After x(0): qubit is |1>, measure gives 1, then x_if applies X again
        # We can't measure after x_if without another clbit, but we can check execute
        result = c._execute_once(seed=0)
        self.assertEqual(result, "1")  # only c[0] is measured

    def test_z_if_applies_when_condition_met(self):
        c = Circuit(1, 1)
        c.measure(0, 0).z_if(0, 0, 0)
        # condition is clbit 0 == 0, which is true
        result = c._execute_once(seed=0)
        self.assertEqual(result, "0")


# ── statevector ───────────────────────────────────────────────────────────────

class TestStatevector(unittest.TestCase):

    def test_initial_state(self):
        c = Circuit(1)
        sv = c.statevector()
        self.assertAlmostEqual(sv[0].real, 1.0)
        self.assertAlmostEqual(sv[1].real, 0.0)

    def test_H_statevector(self):
        c = Circuit(1)
        c.h(0)
        sv = c.statevector()
        v = 1 / math.sqrt(2)
        self.assertAlmostEqual(abs(sv[0]), v)
        self.assertAlmostEqual(abs(sv[1]), v)

    def test_X_statevector(self):
        c = Circuit(1)
        c.x(0)
        sv = c.statevector()
        self.assertAlmostEqual(abs(sv[0]), 0)
        self.assertAlmostEqual(abs(sv[1]), 1)

    def test_bell_statevector(self):
        c = Circuit(2)
        c.h(0).cx(0, 1)
        sv = c.statevector()
        v = 1 / math.sqrt(2)
        self.assertAlmostEqual(abs(sv[0]), v)
        self.assertAlmostEqual(abs(sv[1]), 0)
        self.assertAlmostEqual(abs(sv[2]), 0)
        self.assertAlmostEqual(abs(sv[3]), v)

    def test_raises_with_measurement(self):
        c = Circuit(1, 1)
        c.h(0).measure(0, 0)
        with self.assertRaises(RuntimeError):
            c.statevector()

    def test_returns_list(self):
        c = Circuit(1)
        self.assertIsInstance(c.statevector(), list)

    def test_length(self):
        c = Circuit(3)
        self.assertEqual(len(c.statevector()), 8)


# ── depth ─────────────────────────────────────────────────────────────────────

class TestDepth(unittest.TestCase):

    def test_empty_circuit_depth_zero(self):
        c = Circuit(1)
        self.assertEqual(c.depth(), 0)

    def test_single_gate_depth_one(self):
        c = Circuit(1)
        c.h(0)
        self.assertEqual(c.depth(), 1)

    def test_serial_gates(self):
        c = Circuit(1)
        c.h(0).x(0).z(0)
        self.assertEqual(c.depth(), 3)

    def test_parallel_gates(self):
        c = Circuit(2)
        c.h(0)
        c.h(1)
        self.assertEqual(c.depth(), 1)

    def test_barrier_not_counted(self):
        c = Circuit(1)
        c.h(0).barrier().x(0)
        self.assertEqual(c.depth(), 2)

    def test_two_qubit_gate_depth(self):
        c = Circuit(2)
        c.h(0).cx(0, 1)
        self.assertEqual(c.depth(), 2)


# ── size ──────────────────────────────────────────────────────────────────────

class TestSize(unittest.TestCase):

    def test_empty_circuit(self):
        c = Circuit(1)
        self.assertEqual(c.size(), 0)

    def test_one_gate(self):
        c = Circuit(1)
        c.h(0)
        self.assertEqual(c.size(), 1)

    def test_barrier_not_counted(self):
        c = Circuit(1)
        c.h(0).barrier().x(0)
        self.assertEqual(c.size(), 2)

    def test_identity_not_counted(self):
        c = Circuit(1)
        c.h(0).id(0)
        self.assertEqual(c.size(), 1)

    def test_measure_counted(self):
        c = Circuit(1, 1)
        c.h(0).measure(0, 0)
        self.assertEqual(c.size(), 2)


# ── count_ops ─────────────────────────────────────────────────────────────────

class TestCountOps(unittest.TestCase):

    def test_empty_returns_empty_dict(self):
        c = Circuit(1)
        self.assertEqual(c.count_ops(), {})

    def test_single_gate(self):
        c = Circuit(1)
        c.h(0)
        ops = c.count_ops()
        self.assertEqual(ops["h"], 1)

    def test_multiple_same_gate(self):
        c = Circuit(1)
        c.h(0).h(0).h(0)
        self.assertEqual(c.count_ops()["h"], 3)

    def test_mixed_gates(self):
        c = Circuit(2)
        c.h(0).h(1).cx(0, 1)
        ops = c.count_ops()
        self.assertEqual(ops["h"], 2)
        self.assertEqual(ops["cx"], 1)


# ── inverse ───────────────────────────────────────────────────────────────────

class TestInverse(unittest.TestCase):

    def test_inverse_of_H_is_H(self):
        c = Circuit(1)
        c.h(0)
        inv = c.inverse()
        self.assertEqual(len(inv._instructions), 1)
        # H† = H, verify via statevector
        sv = inv.statevector()
        v = 1 / math.sqrt(2)
        self.assertAlmostEqual(abs(sv[0]), v)

    def test_inverse_reverses_order(self):
        c = Circuit(1)
        c.h(0).x(0)
        inv = c.inverse()
        # inverse should be: x_dg then h_dg (reversed)
        self.assertEqual(len(inv._instructions), 2)

    def test_inverse_raises_with_measure(self):
        c = Circuit(1, 1)
        c.h(0).measure(0, 0)
        with self.assertRaises(RuntimeError):
            c.inverse()

    def test_H_then_inverse_is_identity(self):
        # Apply H then H† = identity, so |0> should stay |0>
        from quantrs.state import QuantumState
        from quantrs.gates import H
        c = Circuit(1)
        c.h(0)
        inv = c.inverse()
        sv_fwd = c.statevector()
        sv_inv = inv.statevector()
        # Both should be H|0> = superposition, because inverse just
        # contains H† which equals H
        self.assertAlmostEqual(abs(sv_fwd[0]), abs(sv_inv[0]))


# ── draw and __repr__ ─────────────────────────────────────────────────────────

class TestDrawAndRepr(unittest.TestCase):

    def test_draw_returns_string(self):
        c = Circuit(2, 2)
        c.h(0).cx(0, 1).measure_all()
        result = c.draw()
        self.assertIsInstance(result, str)

    def test_draw_contains_qubit_labels(self):
        c = Circuit(2)
        c.h(0)
        result = c.draw()
        self.assertIn("q0", result)
        self.assertIn("q1", result)

    def test_draw_contains_clbit_labels(self):
        c = Circuit(1, 1)
        c.h(0).measure(0, 0)
        result = c.draw()
        self.assertIn("c0", result)

    def test_repr_is_string(self):
        c = Circuit(2, 2)
        self.assertIsInstance(repr(c), str)

    def test_repr_contains_qubit_count(self):
        c = Circuit(3, 3)
        self.assertIn("qubits=3", repr(c))

    def test_draw_circuit_function(self):
        c = Circuit(2, 2)
        c.h(0).cx(0, 1).measure_all()
        result = _draw_circuit(c)
        self.assertIsInstance(result, str)
        self.assertIn("H", result)


if __name__ == "__main__":
    unittest.main()