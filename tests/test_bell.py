"""
test_result.py
--------------
Unit tests for every function and method in result.py.

Coverage:
    MeasurementResult    : __init__, probabilities, most_likely,
                           probability_of, outcomes, print, __repr__
    TeleportationResult  : __init__, _extract_sahaj, sahaj_probabilities,
                           fidelity, print, __repr__
"""

import unittest
import math
import io
import sys
from quantrs.result import MeasurementResult, TeleportationResult


# ── MeasurementResult ─────────────────────────────────────────────────────────

class TestMeasurementResultInit(unittest.TestCase):

    def test_stores_counts(self):
        r = MeasurementResult({"0": 500, "1": 500}, 1000)
        self.assertEqual(r.counts["0"], 500)
        self.assertEqual(r.counts["1"], 500)

    def test_stores_shots(self):
        r = MeasurementResult({"0": 100}, 100)
        self.assertEqual(r.shots, 100)

    def test_empty_counts(self):
        r = MeasurementResult({}, 0)
        self.assertEqual(r.counts, {})


class TestMeasurementResultProbabilities(unittest.TestCase):

    def test_equal_split(self):
        r = MeasurementResult({"0": 500, "1": 500}, 1000)
        probs = r.probabilities()
        self.assertAlmostEqual(probs["0"], 0.5)
        self.assertAlmostEqual(probs["1"], 0.5)

    def test_all_zero(self):
        r = MeasurementResult({"00": 1000}, 1000)
        probs = r.probabilities()
        self.assertAlmostEqual(probs["00"], 1.0)

    def test_probs_sum_to_one(self):
        r = MeasurementResult({"00": 400, "11": 600}, 1000)
        probs = r.probabilities()
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_three_outcomes(self):
        r = MeasurementResult({"00": 250, "01": 250, "10": 500}, 1000)
        probs = r.probabilities()
        self.assertAlmostEqual(probs["00"], 0.25)
        self.assertAlmostEqual(probs["10"], 0.5)


class TestMeasurementResultMostLikely(unittest.TestCase):

    def test_clear_winner(self):
        r = MeasurementResult({"0": 900, "1": 100}, 1000)
        self.assertEqual(r.most_likely(), "0")

    def test_only_one_outcome(self):
        r = MeasurementResult({"00": 1000}, 1000)
        self.assertEqual(r.most_likely(), "00")

    def test_picks_highest(self):
        r = MeasurementResult({"00": 200, "01": 600, "11": 200}, 1000)
        self.assertEqual(r.most_likely(), "01")


class TestMeasurementResultProbabilityOf(unittest.TestCase):

    def test_present_outcome(self):
        r = MeasurementResult({"0": 600, "1": 400}, 1000)
        self.assertAlmostEqual(r.probability_of("0"), 0.6)

    def test_absent_outcome_is_zero(self):
        r = MeasurementResult({"0": 1000}, 1000)
        self.assertAlmostEqual(r.probability_of("1"), 0.0)

    def test_full_probability(self):
        r = MeasurementResult({"11": 1000}, 1000)
        self.assertAlmostEqual(r.probability_of("11"), 1.0)


class TestMeasurementResultOutcomes(unittest.TestCase):

    def test_sorted_outcomes(self):
        r = MeasurementResult({"11": 500, "00": 500}, 1000)
        self.assertEqual(r.outcomes(), ["00", "11"])

    def test_single_outcome(self):
        r = MeasurementResult({"0": 100}, 100)
        self.assertEqual(r.outcomes(), ["0"])

    def test_three_outcomes_sorted(self):
        r = MeasurementResult({"10": 1, "01": 1, "00": 1}, 3)
        self.assertEqual(r.outcomes(), ["00", "01", "10"])


class TestMeasurementResultPrint(unittest.TestCase):

    def test_print_produces_output(self):
        r = MeasurementResult({"0": 500, "1": 500}, 1000)
        captured = io.StringIO()
        sys.stdout = captured
        r.print()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("1000", output)
        self.assertIn("|0>", output)
        self.assertIn("|1>", output)

    def test_print_shows_probabilities(self):
        r = MeasurementResult({"0": 1000}, 1000)
        captured = io.StringIO()
        sys.stdout = captured
        r.print()
        sys.stdout = sys.__stdout__
        self.assertIn("1.000", captured.getvalue())


class TestMeasurementResultRepr(unittest.TestCase):

    def test_repr_is_string(self):
        r = MeasurementResult({"0": 100}, 100)
        self.assertIsInstance(repr(r), str)

    def test_repr_contains_shots(self):
        r = MeasurementResult({"0": 100}, 100)
        self.assertIn("100", repr(r))

    def test_repr_contains_MeasurementResult(self):
        r = MeasurementResult({"0": 100}, 100)
        self.assertIn("MeasurementResult", repr(r))


# ── TeleportationResult ───────────────────────────────────────────────────────

def _make_teleport_result(counts, shots, alpha=1.0, beta=0.0, label="zero"):
    return TeleportationResult(
        counts=counts,
        shots=shots,
        state_vec=[complex(alpha), complex(beta)],
        state_label=label,
    )


class TestTeleportationResultInit(unittest.TestCase):

    def test_stores_state_vec(self):
        r = _make_teleport_result({"000": 1000}, 1000)
        self.assertAlmostEqual(r.state_vec[0].real, 1.0)

    def test_stores_state_label(self):
        r = _make_teleport_result({"000": 1000}, 1000, label="plus")
        self.assertEqual(r.state_label, "plus")

    def test_stores_shots(self):
        r = _make_teleport_result({"000": 500}, 500)
        self.assertEqual(r.shots, 500)


class TestExtractSahaj(unittest.TestCase):

    def test_bob_gets_zero_when_all_outcomes_end_in_0(self):
        # All 3-bit strings ending in 0: 000, 010, 100, 110
        counts = {"000": 250, "010": 250, "100": 250, "110": 250}
        r = _make_teleport_result(counts, 1000)
        sahaj = r._extract_sahaj()
        self.assertEqual(sahaj.get("0", 0), 1000)
        self.assertEqual(sahaj.get("1", 0), 0)

    def test_bob_gets_one_when_all_outcomes_end_in_1(self):
        counts = {"001": 250, "011": 250, "101": 250, "111": 250}
        r = _make_teleport_result(counts, 1000, alpha=0.0, beta=1.0, label="one")
        sahaj = r._extract_sahaj()
        self.assertEqual(sahaj.get("1", 0), 1000)
        self.assertEqual(sahaj.get("0", 0), 0)

    def test_mixed_outcomes(self):
        counts = {"000": 500, "001": 500}
        r = _make_teleport_result(counts, 1000)
        sahaj = r._extract_sahaj()
        self.assertEqual(sahaj.get("0", 0), 500)
        self.assertEqual(sahaj.get("1", 0), 500)


class TestSahajProbabilities(unittest.TestCase):

    def test_all_zero_probabilities(self):
        counts = {"000": 500, "010": 500}
        r = _make_teleport_result(counts, 1000)
        probs = r.sahaj_probabilities()
        self.assertAlmostEqual(probs.get("0", 0), 1.0)
        self.assertAlmostEqual(probs.get("1", 0), 0.0)

    def test_half_half(self):
        counts = {"000": 500, "001": 500}
        r = _make_teleport_result(counts, 1000)
        probs = r.sahaj_probabilities()
        self.assertAlmostEqual(probs.get("0", 0), 0.5)
        self.assertAlmostEqual(probs.get("1", 0), 0.5)

    def test_probs_sum_to_one(self):
        counts = {"000": 300, "001": 200, "010": 300, "011": 200}
        r = _make_teleport_result(counts, 1000)
        probs = r.sahaj_probabilities()
        self.assertAlmostEqual(sum(probs.values()), 1.0)


class TestFidelity(unittest.TestCase):

    def test_perfect_fidelity_zero_state(self):
        # Teleporting |0>: Bob should always measure 0
        counts = {"000": 250, "010": 250, "100": 250, "110": 250}
        r = _make_teleport_result(counts, 1000, alpha=1.0, beta=0.0)
        self.assertAlmostEqual(r.fidelity(), 1.0)

    def test_perfect_fidelity_one_state(self):
        # Teleporting |1>: Bob should always measure 1
        counts = {"001": 250, "011": 250, "101": 250, "111": 250}
        r = _make_teleport_result(counts, 1000, alpha=0.0, beta=1.0, label="one")
        self.assertAlmostEqual(r.fidelity(), 1.0)

    def test_zero_fidelity(self):
        # Teleporting |0> but Bob always measures 1
        counts = {"001": 250, "011": 250, "101": 250, "111": 250}
        r = _make_teleport_result(counts, 1000, alpha=1.0, beta=0.0)
        self.assertAlmostEqual(r.fidelity(), 0.0)

    def test_fidelity_is_between_0_and_1(self):
        counts = {"000": 400, "001": 100, "010": 400, "011": 100}
        r = _make_teleport_result(counts, 1000, alpha=1.0, beta=0.0)
        f = r.fidelity()
        self.assertGreaterEqual(f, 0.0)
        self.assertLessEqual(f, 1.0)

    def test_fidelity_plus_state(self):
        # |+> state: ideal P(0)=0.5, P(1)=0.5
        # Perfect result: Bob measures 0 and 1 equally
        a = 1 / math.sqrt(2)
        counts = {"000": 250, "001": 250, "010": 250, "011": 250}
        r = _make_teleport_result(counts, 1000, alpha=a, beta=a, label="plus")
        self.assertAlmostEqual(r.fidelity(), 1.0)

    def test_fidelity_capped_at_one(self):
        # Even with slightly off counts, fidelity should not exceed 1
        counts = {"000": 600, "010": 400}
        r = _make_teleport_result(counts, 1000, alpha=1.0, beta=0.0)
        self.assertLessEqual(r.fidelity(), 1.0)


class TestTeleportationResultPrint(unittest.TestCase):

    def test_print_produces_output(self):
        counts = {"000": 500, "010": 500}
        r = _make_teleport_result(counts, 1000)
        captured = io.StringIO()
        sys.stdout = captured
        r.print()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("Teleportation", output)
        self.assertIn("Fidelity", output)

    def test_print_shows_state_label(self):
        counts = {"000": 1000}
        r = _make_teleport_result(counts, 1000, label="plus")
        captured = io.StringIO()
        sys.stdout = captured
        r.print()
        sys.stdout = sys.__stdout__
        self.assertIn("plus", captured.getvalue())

    def test_print_shows_pass_when_fidelity_high(self):
        counts = {"000": 250, "010": 250, "100": 250, "110": 250}
        r = _make_teleport_result(counts, 1000, alpha=1.0, beta=0.0)
        captured = io.StringIO()
        sys.stdout = captured
        r.print()
        sys.stdout = sys.__stdout__
        self.assertIn("PASS", captured.getvalue())

    def test_print_shows_fail_when_fidelity_low(self):
        # Bob always measures wrong outcome
        counts = {"001": 250, "011": 250, "101": 250, "111": 250}
        r = _make_teleport_result(counts, 1000, alpha=1.0, beta=0.0)
        captured = io.StringIO()
        sys.stdout = captured
        r.print()
        sys.stdout = sys.__stdout__
        self.assertIn("FAIL", captured.getvalue())


class TestTeleportationResultRepr(unittest.TestCase):

    def test_repr_is_string(self):
        r = _make_teleport_result({"000": 1000}, 1000)
        self.assertIsInstance(repr(r), str)

    def test_repr_contains_TeleportationResult(self):
        r = _make_teleport_result({"000": 1000}, 1000)
        self.assertIn("TeleportationResult", repr(r))

    def test_repr_contains_state_label(self):
        r = _make_teleport_result({"000": 1000}, 1000, label="plus")
        self.assertIn("plus", repr(r))

    def test_repr_contains_fidelity(self):
        r = _make_teleport_result({"000": 1000}, 1000)
        self.assertIn("fidelity", repr(r))

    def test_repr_contains_shots(self):
        r = _make_teleport_result({"000": 1000}, 1000)
        self.assertIn("1000", repr(r))


if __name__ == "__main__":
    unittest.main()