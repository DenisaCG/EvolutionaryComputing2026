"""NN controller: forward pass and flat-vector <-> weight-matrix conversion.

Architecture matches `A2_template_2026.py`: input -> tanh(hidden) ->
tanh(output), rescaled to the hinge range. CMA-ES and random search both
search over the FLAT weight vector; `unflatten` reshapes it into the two
matrices `nn_controller` expects.
"""

from __future__ import annotations

import mujoco as mj
import numpy as np
import numpy.typing as npt


def genotype_length(input_size: int, hidden_size: int, output_size: int) -> int:
    """Total number of weights: input->hidden plus hidden->output."""
    return input_size * hidden_size + hidden_size * output_size


def unflatten(
    weights: npt.NDArray[np.float64],
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Reshape a flat weight vector into [w1, w2] for `nn_controller`."""
    split = input_size * hidden_size
    w1 = weights[:split].reshape(input_size, hidden_size)
    w2 = weights[split:].reshape(hidden_size, output_size)
    return [w1, w2]


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Map robot state (bare qpos) to hinge commands, scaled to [-pi/2, pi/2]."""
    w1, w2 = weights
    inputs = data.qpos
    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(layer1 @ w2)
    return outputs * (np.pi / 2)
