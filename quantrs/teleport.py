"""
teleport.py
-----------

The protocol 
------------
Soham wants to send Sahaj an unknown qubit state |psi> = alpha|0> + beta|1>.
He cannot simply copy it (no-cloning theorem) and also he cannot send the qubit
directly (that would require a quantum channel).  Instead:

    Qubit layout:
        q[0]  Soham's input qubit  — holds |psi>
        q[1]  Soham's Bell qubit   — his half of the shared pair
        q[2]  Sahaj's Bell qubit   — his half of the shared pair

    Step 1  Prepare |psi> on q[0].
    Step 2  Create Bell pair Phi+ on (q[1], q[2]).
    Step 3  Soham's Bell measurement:
                CX(0, 1)  — entangle input with Soham's Bell qubit
                H(0)      — rotate into Bell basis
    Step 4  Soham measures q[0] → c[0] and q[1] → c[1].
            He sends c[0] and c[1] to Sahaj over a classical channel.
    Step 5  Sahaj applies corrections:
                if c[1] == 1:  X on q[2]
                if c[0] == 1:  Z on q[2]
    Step 6  Sahaj's qubit q[2] now holds |psi> exactly.

The classical bits c[0] and c[1] carry no information about |psi> on their
own — the protocol is information-theoretically secure.

Usage
-----
    result = Teleporter.run(state="plus")
    result = Teleporter.run(state=[0.6, 0.8])
    result.print()
"""

from __future__ import annotations
import math
import random
from typing import List, Union, Optional, Dict

from .state import QuantumState
from .result import TeleportationResult
from . import gates as G


# Built-in named states
_NAMED_STATES: Dict[str, List[complex]] = {
    "zero":  [1.0,             0.0            ],
    "one":   [0.0,             1.0            ],
    "plus":  [1/math.sqrt(2),  1/math.sqrt(2) ],
    "minus": [1/math.sqrt(2), -1/math.sqrt(2) ],
    "i":     [1/math.sqrt(2),  1j/math.sqrt(2)],
}


class Teleporter:
    """
    Implements the quantum teleportation protocol from scratch.

    Parameters
    ----------
    state : list[complex] or str
        The state to teleport.  Either [alpha, beta] or a named string:
        'zero', 'one', 'plus', 'minus', 'i'.
    """

    def __init__(self, state: Union[List[complex], str] = "zero") -> None:
        self._state_vec = _resolve_state(state)
        self._state_label = state if isinstance(state, str) else "custom"

    # ── public entry point ────────────────────────────────────────────────────

    @classmethod
    def run(
        cls,
        state: Union[List[complex], str] = "zero",
        shots: int = 1024,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> TeleportationResult:
        """
        Run the full teleportation protocol `shots` times.

        Parameters
        ----------
        state   : list[complex] or str
        shots   : int
        seed    : int, optional
        verbose : bool  — if True prints a summary

        Returns
        -------
        TeleportationResult
        """
        tp = cls(state)
        if verbose:
            a, b = tp._state_vec
            print(f"Teleporting: {tp._state_label}")
            print(f"  alpha = {a:.4f}  |alpha|² = {abs(a)**2:.4f}")
            print(f"  beta  = {b:.4f}  |beta|²  = {abs(b)**2:.4f}")

        counts: Dict[str, int] = {}
        rng = random.Random(seed)
        for _ in range(shots):
            shot_seed = rng.randint(0, 2**31)
            outcome = tp._run_once(seed=shot_seed)
            counts[outcome] = counts.get(outcome, 0) + 1

        result = TeleportationResult(
            counts=counts,
            shots=shots,
            state_vec=tp._state_vec,
            state_label=tp._state_label,
        )
        if verbose:
            result.print()
        return result

    # ── single execution ──────────────────────────────────────────────────────

    def _run_once(self, seed: Optional[int] = None) -> str:
        """
        Run the teleportation protocol once and return a 3-bit string:
            bit 0 = c[0]  (Alice's measurement of q[0])
            bit 1 = c[1]  (Alice's measurement of q[1])
            bit 2 = c[2]  (Bob's verification measurement of q[2])
        """
        state = QuantumState(3, seed=seed)

        # ── Step 1: Prepare |psi> on q[0] ────────────────────────────────
        alpha, beta = self._state_vec
        # Set the full 3-qubit statevector:
        # |psi> ⊗ |0> ⊗ |0>  =  alpha|000> + beta|100>
        # Indices (big-endian q0=MSB): |000>=0, |100>=4
        state._vec.data[0] = alpha
        state._vec.data[4] = beta

        # ── Step 2: Bell pair on q[1] and q[2] ───────────────────────────
        # H on q[1]
        state.apply_single(G.H, 1)
        # CX(1, 2)
        state.apply_two(G.CX, 1, 2)

        # ── Step 3: Soham's Bell measurement ─────────────────────────────
        # CX(0, 1)
        state.apply_two(G.CX, 0, 1)
        # H on q[0]
        state.apply_single(G.H, 0)

        # ── Step 4: Soham measures ────────────────────────────────────────
        c0 = state.measure(0)
        c1 = state.measure(1)

        # ── Step 5: Sahaj's corrections ─────────────────────────────────────
        if c1 == 1:
            state.apply_single(G.X, 2)
        if c0 == 1:
            state.apply_single(G.Z, 2)

        # ── Step 6: Sahaj measures to verify ───────────────────────────────
        c2 = state.measure(2)

        return f"{c0}{c1}{c2}"

    # ── utilities ─────────────────────────────────────────────────────────────

    def describe(self) -> None:
        """Print the protocol description with the chosen state."""
        a, b = self._state_vec
        print(f"\nQuantum Teleportation")
        print(f"{'─'*40}")
        print(f"  State to teleport : {self._state_label}")
        print(f"  alpha             : {a:.6f}")
        print(f"  beta              : {b:.6f}")
        print(f"  |alpha|²          : {abs(a)**2:.4f}  (P of Bob measuring |0>)")
        print(f"  |beta|²           : {abs(b)**2:.4f}  (P of Bob measuring |1>)")
        print(f"\n  Protocol steps:")
        print(f"    1. Prepare |psi> on q[0]")
        print(f"    2. Create Bell pair (Phi+) on q[1], q[2]")
        print(f"    3. CX(0,1)  then  H(0)  — Bell basis rotation")
        print(f"    4. Measure q[0] → c[0],  q[1] → c[1]")
        print(f"    5. if c[1]==1: X on q[2]   (bit-flip correction)")
        print(f"       if c[0]==1: Z on q[2]   (phase-flip correction)")
        print(f"    6. Measure q[2] — should match |psi>")

    def __repr__(self) -> str:
        a, b = self._state_vec
        return f"Teleporter(state={self._state_label!r}, alpha={a:.4f}, beta={b:.4f})"


# ── State resolution ──────────────────────────────────────────────────────────

def _resolve_state(state: Union[List[complex], str]) -> List[complex]:
    """Validate and normalise the input state to [alpha, beta]."""
    if isinstance(state, str):
        key = state.lower()
        if key not in _NAMED_STATES:
            raise ValueError(
                f"Unknown state name '{state}'. "
                f"Choose from: {list(_NAMED_STATES.keys())}"
            )
        return list(_NAMED_STATES[key])

    if len(state) != 2:
        raise ValueError(
            f"State vector must have exactly 2 elements [alpha, beta], got {len(state)}."
        )

    alpha, beta = complex(state[0]), complex(state[1])
    norm = math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    if norm < 1e-12:
        raise ValueError("State vector cannot be the zero vector.")

    return [alpha / norm, beta / norm]
