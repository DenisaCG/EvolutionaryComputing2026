"""Body and world factories.

Single place mapping a config's `body` name to the ariel prebuilt-robot
factory that constructs it, and the world (fixed: Olympic Arena) every
experiment spawns robots into. Add a new body here, nowhere else, if the
sweep grows beyond turtle/iguana.
"""

from __future__ import annotations

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import (
    iguana,
    turtle,
)
from ariel.simulation.environments import OlympicArena

from config import BodyName

BODY_FACTORIES: dict[BodyName, type[CoreModule] | object] = {
    "turtle": turtle,
    "iguana": iguana,
}


def build_robot(body: BodyName) -> CoreModule:
    """Construct the robot body named by `body`."""
    return BODY_FACTORIES[body]()


def build_world() -> OlympicArena:
    """Construct the (fixed, for this whole assignment) simulation world.

    `load_precompiled=False` matches the ariel course's own Olympic Arena
    template, avoiding a stale precompiled-XML cache built with different
    parameters.
    """
    return OlympicArena(load_precompiled=False)
