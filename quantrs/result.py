"""
result.py
---------
Result classes that wrap simulation output and provide clean analysis.

MeasurementResult     — general shot counts with probabilities
TeleportationResult   — specialised result for the teleportation protocol
"""

from __future__ import annotations
import math
from typing import Dict, List


class MeasurementResult:
    """
    Wraps shot counts from Circuit.run() with convenience accessors.

    Attributes
    ----------
    counts : dict[str, int]
        Bitstring → count.
    shots : int
        Total shots.
    """

    def __init__(self, counts: Dict[str, int], shots: int) -> None:
        self.counts = counts
        self.shots = shots

    def probabilities(self) -> Dict[str, float]:
        """Return {bitstring: probability} for all observed outcomes."""
        return {s: c / self.shots for s, c in self.counts.items()}

    def most_likely(self) -> str:
        """Return the bitstring with the highest count."""
        return max(self.counts, key=self.counts.get)

    def probability_of(self, bitstring: str) -> float:
        """Return the probability of a specific bitstring."""
        return self.counts.get(bitstring, 0) / self.shots

    def outcomes(self) -> List[str]:
        """Return a sorted list of all observed bitstrings."""
        return sorted(self.counts.keys())

    def print(self) -> None:
        """Print a formatted probability table."""
        print(f"\nMeasurement results  ({self.shots} shots)")
        print("─" * 40)
        for state in sorted(self.counts):
            p = self.counts[state] / self.shots
            bar = "█" * int(p * 28)
            print(f"  |{state}>  {bar:<28}  {p:.3f}  ({self.counts[state]})")
        print()

    def __repr__(self) -> str:
        return f"MeasurementResult(shots={self.shots}, outcomes={self.outcomes()})"


class TeleportationResult(MeasurementResult):
    """
    Specialised result for quantum teleportation.

    The 3-bit measurement string has the convention:
        bit 0 = c[0]  Soham's measurement of q[0]
        bit 1 = c[1]  Soham's measurement of q[1]
        bit 2 = c[2]  Sahaj's verification measurement of q[2]

    Fidelity is estimated by comparing Sahaj's marginal output distribution
    to the ideal output for the teleported state.
    """

    def __init__(
        self,
        counts: Dict[str, int],
        shots: int,
        state_vec: List[complex],
        state_label: str,
    ) -> None:
        super().__init__(counts, shots)
        self.state_vec = state_vec
        self.state_label = state_label
        self._sahaj_counts = self._extract_sahaj()

    def _extract_sahaj(self) -> Dict[str, int]:
        """
        Extract Sahaj's qubit (bit index 2, the last character in our
        convention '01c2') from the 3-bit outcome strings.
        """
        sahaj: Dict[str, int] = {}
        for bs, cnt in self.counts.items():
            # Our bitstring is c[0]c[1]c[2], so last char = Sahaj's bit
            sahaj_bit = bs[-1]
            sahaj[sahaj_bit] = sahaj.get(sahaj_bit, 0) + cnt
        return sahaj

    def sahaj_probabilities(self) -> Dict[str, float]:
        """Marginal probability distribution for Sahaj's qubit."""
        return {bit: cnt / self.shots for bit, cnt in self._sahaj_counts.items()}

    def fidelity(self) -> float:
        """
        Estimate teleportation fidelity.

        Compares Sahaj's measured probabilities to the ideal probabilities
        |alpha|² and |beta|² for the teleported state.

        Returns a value in [0, 1].  Perfect teleportation = 1.0.
        """
        alpha, beta = self.state_vec
        p0_ideal = abs(alpha) ** 2
        p1_ideal = abs(beta) ** 2

        sahaj_probs = self.sahaj_probabilities()
        p0_got = sahaj_probs.get("0", 0.0)
        p1_got = sahaj_probs.get("1", 0.0)

        # Bhattacharyya-style fidelity between ideal and measured distributions
        fidelity = (math.sqrt(p0_ideal * p0_got) + math.sqrt(p1_ideal * p1_got)) ** 2
        return min(fidelity, 1.0)

    def print(self) -> None:
        """Print a detailed teleportation summary."""
        alpha, beta = self.state_vec
        p0_ideal = abs(alpha) ** 2
        p1_ideal = abs(beta) ** 2

        print(f"\nTeleportation result  —  state: {self.state_label}")
        print("─" * 44)
        print(f"  alpha = {alpha:.4f}   |alpha|² = {p0_ideal:.4f}")
        print(f"  beta  = {beta:.4f}   |beta|²  = {p1_ideal:.4f}")
        print()
        print("  All outcomes  (c0 c1 c2):")
        for bs in sorted(self.counts):
            print(f"    {bs[0]} {bs[1]} {bs[2]}  →  {self.counts[bs]:5d}")

        print()
        sahaj_probs = self.sahaj_probabilities()
        print("  Sahaj's qubit (marginal):")
        for bit in sorted(sahaj_probs):
            p = sahaj_probs[bit]
            bar = "█" * int(p * 28)
            print(f"    |{bit}>  {bar:<28}  {p:.4f}")

        print()
        f = self.fidelity()
        status = "PASS ✓" if f >= 0.90 else "FAIL ✗"
        print(f"  Fidelity : {f:.4f}  [{status}]")
        print(f"  Ideal    : P(|0>) = {p0_ideal:.4f},  P(|1>) = {p1_ideal:.4f}")
        print()

    def __repr__(self) -> str:
        return (
            f"TeleportationResult(state={self.state_label!r}, "
            f"fidelity={self.fidelity():.4f}, shots={self.shots})"
        )
