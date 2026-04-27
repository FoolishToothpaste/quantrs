"""
test_teleport.py
----------------
Unit tests for every function and method in teleport.py.

Coverage:
    Teleporter    : __init__, run, _run_once, describe, __repr__
    Module        : _resolve_state
"""

import unittest
import math
import io
import sys
from quantrs import Teleporter
from quantrs.teleport import _resolve_state
from quantrs.result import TeleportationResult


# ── _resolve_state ────────────────────────────────────────────────────────────

class TestResolveState(unittest.TestCase):

    def test_named_zero(self):
        sv = _resolve_state("zero")
        self.assertAlmostEqual(abs(sv[0]), 1.0)
        self.assertAlmostEqual(abs(sv[1]), 0.0)

    def test_named_one(self):
        sv = _resolve_state("one")
        self.assertAlmostEqual(abs(sv[0]), 0.0)
        self.assertAlmostEqual(abs(sv[1]), 1.0)

    def test_named_plus(self):
        sv = _resolve_state("plus")
        s = 1 / math.sqrt(2)
        self.assertAlmostEqual(abs(sv[0]), s)
        self.assertAlmostEqual(abs(sv[1]), s)

    def test_named_minus(self):
        sv = _resolve_state("minus")
        s = 1 / math.sqrt(2)
        self.assertAlmostEqual(abs(sv[0]), s)
        self.assertAlmostEqual(abs(sv[1]), s)
        # minus has opposite sign on second component
        self.assertLess(sv[1].real, 0)

    def test_named_i(self):
        sv = _resolve_state("i")
        s = 1 / math.sqrt(2)
        self.assertAlmostEqual(abs(sv[0]), s)
        self.assertAlmostEqual(abs(sv[1]), s)

    def test_case_insensitive(self):
        sv = _resolve_state("ZERO")
        self.assertAlmostEqual(abs(sv[0]), 1.0)

    def test_custom_state_list(self):
        sv = _resolve_state([1, 0])
        self.assertAlmostEqual(abs(sv[0]), 1.0)
        self.assertAlmostEqual(abs(sv[1]), 0.0)

    def test_custom_state_normalised(self):
        sv = _resolve_state([3, 4])
        norm = math.sqrt(abs(sv[0])**2 + abs(sv[1])**2)
        self.assertAlmostEqual(norm, 1.0)

    def test_custom_state_components(self):
        sv = _resolve_state([3, 4])
        self.assertAlmostEqual(abs(sv[0]), 0.6)
        self.assertAlmostEqual(abs(sv[1]), 0.8)

    def test_complex_state(self):
        sv = _resolve_state([1+0j, 1j])
        norm = math.sqrt(abs(sv[0])**2 + abs(sv[1])**2)
        self.assertAlmostEqual(norm, 1.0)

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            _resolve_state("diagonal")

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            _resolve_state([0, 0])

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            _resolve_state([1, 0, 0])

    def test_single_element_raises(self):
        with self.assertRaises(ValueError):
            _resolve_state([1])

    def test_all_named_states_normalised(self):
        for name in ["zero", "one", "plus", "minus", "i"]:
            sv = _resolve_state(name)
            norm = math.sqrt(abs(sv[0])**2 + abs(sv[1])**2)
            self.assertAlmostEqual(norm, 1.0, msg=f"{name} not normalised")


# ── Teleporter.__init__ ───────────────────────────────────────────────────────

class TestTeleporterInit(unittest.TestCase):

    def test_state_vec_stored(self):
        tp = Teleporter("zero")
        self.assertAlmostEqual(abs(tp._state_vec[0]), 1.0)

    def test_state_label_stored_for_named(self):
        tp = Teleporter("plus")
        self.assertEqual(tp._state_label, "plus")

    def test_state_label_custom_for_list(self):
        tp = Teleporter([1, 0])
        self.assertEqual(tp._state_label, "custom")

    def test_default_state_is_zero(self):
        tp = Teleporter()
        self.assertAlmostEqual(abs(tp._state_vec[0]), 1.0)
        self.assertAlmostEqual(abs(tp._state_vec[1]), 0.0)

    def test_invalid_state_raises(self):
        with self.assertRaises(ValueError):
            Teleporter("invalid")

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            Teleporter([0, 0])


# ── Teleporter._run_once ──────────────────────────────────────────────────────

class TestTeleporterRunOnce(unittest.TestCase):

    def test_returns_3_bit_string(self):
        tp = Teleporter("zero")
        result = tp._run_once(seed=0)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(c in "01" for c in result))

    def test_teleport_zero_bob_always_zero(self):
        tp = Teleporter("zero")
        for trial in range(20):
            outcome = tp._run_once(seed=trial)
            self.assertEqual(outcome[-1], "0",
                             f"Bob measured 1 for |0> on trial {trial}")

    def test_teleport_one_bob_always_one(self):
        tp = Teleporter("one")
        for trial in range(20):
            outcome = tp._run_once(seed=trial)
            self.assertEqual(outcome[-1], "1",
                             f"Bob measured 0 for |1> on trial {trial}")

    def test_seed_gives_reproducible_result(self):
        tp = Teleporter("plus")
        r1 = tp._run_once(seed=42)
        r2 = tp._run_once(seed=42)
        self.assertEqual(r1, r2)

    def test_different_seeds_can_differ(self):
        tp = Teleporter("plus")
        results = {tp._run_once(seed=i) for i in range(30)}
        # With |+> state, Bob should measure both 0 and 1 across many trials
        bob_results = {r[-1] for r in results}
        self.assertEqual(bob_results, {"0", "1"})

    def test_teleport_plus_bob_gets_both_values(self):
        tp = Teleporter("plus")
        bob_bits = set()
        for trial in range(50):
            outcome = tp._run_once(seed=trial)
            bob_bits.add(outcome[-1])
        self.assertIn("0", bob_bits)
        self.assertIn("1", bob_bits)

    def test_alice_bits_c0_and_c1_vary(self):
        # Alice's bits should not be fixed
        tp = Teleporter("plus")
        c0_bits = set()
        c1_bits = set()
        for trial in range(30):
            outcome = tp._run_once(seed=trial)
            c0_bits.add(outcome[0])
            c1_bits.add(outcome[1])
        self.assertEqual(c0_bits, {"0", "1"})
        self.assertEqual(c1_bits, {"0", "1"})


# ── Teleporter.run ────────────────────────────────────────────────────────────

class TestTeleporterRun(unittest.TestCase):

    def test_returns_teleportation_result(self):
        result = Teleporter.run(state="zero", shots=50, verbose=False)
        self.assertIsInstance(result, TeleportationResult)

    def test_correct_shot_count(self):
        result = Teleporter.run(state="zero", shots=100, seed=0, verbose=False)
        self.assertEqual(sum(result.counts.values()), 100)

    def test_fidelity_zero_state_high(self):
        result = Teleporter.run(state="zero", shots=500, seed=1, verbose=False)
        self.assertGreater(result.fidelity(), 0.95)

    def test_fidelity_one_state_high(self):
        result = Teleporter.run(state="one", shots=500, seed=2, verbose=False)
        self.assertGreater(result.fidelity(), 0.95)

    def test_fidelity_plus_state_high(self):
        result = Teleporter.run(state="plus", shots=1000, seed=0, verbose=False)
        self.assertGreater(result.fidelity(), 0.90)

    def test_fidelity_minus_state_high(self):
        result = Teleporter.run(state="minus", shots=1000, seed=0, verbose=False)
        self.assertGreater(result.fidelity(), 0.90)

    def test_fidelity_custom_state(self):
        result = Teleporter.run(state=[0.6, 0.8], shots=500, seed=3, verbose=False)
        self.assertGreater(result.fidelity(), 0.90)

    def test_seed_reproducible(self):
        r1 = Teleporter.run(state="plus", shots=100, seed=7, verbose=False)
        r2 = Teleporter.run(state="plus", shots=100, seed=7, verbose=False)
        self.assertEqual(r1.counts, r2.counts)

    def test_state_label_preserved(self):
        result = Teleporter.run(state="plus", shots=50, verbose=False)
        self.assertEqual(result.state_label, "plus")

    def test_verbose_false_no_output(self):
        captured = io.StringIO()
        sys.stdout = captured
        Teleporter.run(state="zero", shots=50, verbose=False)
        sys.stdout = sys.__stdout__
        self.assertEqual(captured.getvalue(), "")

    def test_verbose_true_produces_output(self):
        captured = io.StringIO()
        sys.stdout = captured
        Teleporter.run(state="zero", shots=50, verbose=True)
        sys.stdout = sys.__stdout__
        self.assertGreater(len(captured.getvalue()), 0)

    def test_outcomes_are_3_bit_strings(self):
        result = Teleporter.run(state="plus", shots=50, verbose=False)
        for k in result.counts:
            self.assertEqual(len(k), 3)
            self.assertTrue(all(c in "01" for c in k))

    def test_invalid_state_raises(self):
        with self.assertRaises(ValueError):
            Teleporter.run(state="invalid", verbose=False)


# ── Teleporter.describe ───────────────────────────────────────────────────────

class TestTeleporterDescribe(unittest.TestCase):

    def test_describe_produces_output(self):
        tp = Teleporter("plus")
        captured = io.StringIO()
        sys.stdout = captured
        tp.describe()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertGreater(len(output), 0)

    def test_describe_contains_state_label(self):
        tp = Teleporter("plus")
        captured = io.StringIO()
        sys.stdout = captured
        tp.describe()
        sys.stdout = sys.__stdout__
        self.assertIn("plus", captured.getvalue())

    def test_describe_contains_alpha(self):
        tp = Teleporter("zero")
        captured = io.StringIO()
        sys.stdout = captured
        tp.describe()
        sys.stdout = sys.__stdout__
        self.assertIn("alpha", captured.getvalue())

    def test_describe_contains_beta(self):
        tp = Teleporter("zero")
        captured = io.StringIO()
        sys.stdout = captured
        tp.describe()
        sys.stdout = sys.__stdout__
        self.assertIn("beta", captured.getvalue())

    def test_describe_contains_protocol_steps(self):
        tp = Teleporter("zero")
        captured = io.StringIO()
        sys.stdout = captured
        tp.describe()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("Step", output.replace("step", "Step"))

    def test_describe_contains_teleportation(self):
        tp = Teleporter("zero")
        captured = io.StringIO()
        sys.stdout = captured
        tp.describe()
        sys.stdout = sys.__stdout__
        self.assertIn("Teleportation", captured.getvalue())


# ── Teleporter.__repr__ ───────────────────────────────────────────────────────

class TestTeleporterRepr(unittest.TestCase):

    def test_repr_is_string(self):
        tp = Teleporter("zero")
        self.assertIsInstance(repr(tp), str)

    def test_repr_contains_Teleporter(self):
        tp = Teleporter("zero")
        self.assertIn("Teleporter", repr(tp))

    def test_repr_contains_state_label(self):
        tp = Teleporter("plus")
        self.assertIn("plus", repr(tp))

    def test_repr_contains_alpha(self):
        tp = Teleporter("zero")
        self.assertIn("alpha", repr(tp))

    def test_repr_contains_beta(self):
        tp = Teleporter("zero")
        self.assertIn("beta", repr(tp))

    def test_repr_custom_state_label(self):
        tp = Teleporter([0.6, 0.8])
        self.assertIn("custom", repr(tp))


if __name__ == "__main__":
    unittest.main()