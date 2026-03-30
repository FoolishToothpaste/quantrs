"""
quantrs — quantum computing for a photonic computer

Public API
----------
    Qubit            — single-qubit fluent gate chain
    Circuit          — multi-qubit circuit builder and simulator
    BellState        — all four maximally entangled Bell states
    Teleporter       — full quantum teleportation protocol
    MeasurementResult    — simulation output with probabilities
    TeleportationResult  — teleportation output with fidelity

Quick start
-----------
    from quantrs import Qubit, Circuit, BellState, Teleporter

    # Single qubit
    q = Qubit()
    q.h().x().rz(0.5).measure()
    q.run().print()

    # Bell state
    BellState.phi_plus().draw()
    BellState.phi_plus().run().print()

    # Teleportation
    Teleporter.run(state="plus")
    Teleporter.run(state=[0.6, 0.8])
"""

from .qubit import Qubit
from .circuit import Circuit
from .bell import BellState
from .teleport import Teleporter
from .result import MeasurementResult, TeleportationResult

__all__ = [
    "Qubit",
    "Circuit",
    "BellState",
    "Teleporter",
    "MeasurementResult",
    "TeleportationResult",
]

__version__ = "0.1.0"
