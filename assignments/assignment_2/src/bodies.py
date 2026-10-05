"""Body and world factories.

Single place mapping a config's `body` name to the ariel prebuilt-robot
factory that constructs it, and the world (fixed: Olympic Arena) every
experiment spawns robots into. Add a new body here, nowhere else, if the
sweep grows beyond turtle/iguana.
"""

from __future__ import annotations

from functools import partial
from unittest import mock

import ariel.simulation.environments.olympic_arena as olympic_arena_module
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import (
    iguana,
    turtle,
)
from ariel.simulation.environments import OlympicArena
from ariel.utils.noise_gen import PerlinNoise

from config import BodyName

BODY_FACTORIES: dict[BodyName, type[CoreModule] | object] = {
    "turtle": turtle,
    "iguana": iguana,
}


def build_robot(body: BodyName) -> CoreModule:
    """Construct the robot body named by `body`."""
    return BODY_FACTORIES[body]()


def build_world(terrain_seed: int | None) -> OlympicArena:
    """Construct the (fixed, for this whole assignment) simulation world.

    `OlympicArena` generates its rugged section from Perlin noise created
    without a seed, so every construction -- i.e. every fitness evaluation,
    since each episode builds a fresh world -- would get different bumps.
    While the arena is built, its module's `PerlinNoise` is replaced by one
    seeded with `terrain_seed`, so every evaluation, run and machine sees
    the same terrain. This is a runtime substitution in our code; ariel's
    source is unchanged. `terrain_seed=None` restores ariel's random terrain.

    `load_precompiled=False` matches the ariel course's own Olympic Arena
    template (the repo ships no precompiled arena XML anyway).
    """
    seeded_noise = partial(PerlinNoise, seed=terrain_seed)
    with mock.patch.object(olympic_arena_module, "PerlinNoise", seeded_noise):
        return OlympicArena(load_precompiled=False)
