"""NN controller: input vector, forward pass, flat-vector <-> weight matrices.

Architecture: input -> tanh(hidden) -> tanh(output), rescaled to the hinge
range, with a bias on both layers. Inputs (see `controller_inputs`):

  * hinge angles (qpos after the core's free joint) -- proprioception;
  * sin(2*pi*f*t), cos(2*pi*f*t) -- an external clock, so the otherwise
    reactive network has a rhythm to build a gait on. The network stays
    feedforward with no internal oscillators, so it is not a CPG. The
    sin/cos pair is our choice: with sin alone the network cannot tell the
    rising from the falling half of the cycle, while the pair makes any
    per-joint phase offset a weighted sum
    (sin(wt + phi) = cos(phi) sin(wt) + sin(phi) cos(wt)). The closest
    precedent, Clune et al. (2009, 2011), fed a single sine of ~1.3 Hz to
    HyperNEAT quadruped controllers; we adopt that frequency as default;
  * cos/sin of the bearing to the target in the core's heading frame -- so
    the network knows which way to go relative to where it faces.

CMA-ES and random search both search over the FLAT weight vector; `unflatten`
reshapes it into the two (bias-augmented) matrices `nn_controller` expects.
"""

from __future__ import annotations

import math

import mujoco as mj
import numpy as np
import numpy.typing as npt

# The robot is spawned with a free joint: qpos[0:3] is the core position,
# qpos[3:7] its orientation quaternion (w, x, y, z), the rest hinge angles.
FREE_JOINT_QPOS = 7


def controller_inputs(
    data: mj.MjData,
    target_xy: npt.NDArray[np.float64],
    clock_hz: float,
) -> npt.NDArray[np.float64]:
    """Hinge angles + clock (sin, cos) + target bearing (cos, sin)."""
    hinge_angles = data.qpos[FREE_JOINT_QPOS:]

    phase = 2 * math.pi * clock_hz * data.time
    clock = [math.sin(phase), math.cos(phase)]

    w, x, y, z = data.qpos[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    dx, dy = target_xy - data.qpos[0:2]
    bearing = math.atan2(dy, dx) - yaw
    target = [math.cos(bearing), math.sin(bearing)]

    return np.concatenate([hinge_angles, clock, target])


def genotype_length(input_size: int, hidden_size: int, output_size: int) -> int:
    """Total number of weights, including one bias per hidden and output unit."""
    return (input_size + 1) * hidden_size + (hidden_size + 1) * output_size


def unflatten(
    weights: npt.NDArray[np.float64],
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Reshape a flat weight vector into [w1, w2]; the last row of each is the bias."""
    split = (input_size + 1) * hidden_size
    w1 = weights[:split].reshape(input_size + 1, hidden_size)
    w2 = weights[split:].reshape(hidden_size + 1, output_size)
    return [w1, w2]


def nn_controller(
    inputs: npt.NDArray[np.float64],
    weights: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Map the input vector to hinge commands, scaled to [-pi/2, pi/2]."""
    w1, w2 = weights
    layer1 = np.tanh(np.append(inputs, 1.0) @ w1)
    outputs = np.tanh(np.append(layer1, 1.0) @ w2)
    return outputs * (np.pi / 2)


class FastController:
    """`controller_inputs` + `nn_controller` with preallocated buffers.

    Called once per physics step (7,500 times per 15 s episode), so the
    per-call allocations of the reference functions (`np.append`,
    `np.concatenate`, list building) were ~45% of an evaluation's wall time.
    Computes the same quantities with the same operations in the same order,
    writing into reused bias-augmented buffers instead, so the actions are
    bit-identical to `nn_controller(controller_inputs(...), weights)`
    (checked by `tests/test_fast_controller.py`).
    """

    def __init__(
        self,
        weights: list[npt.NDArray[np.float64]],
        target_xy: npt.NDArray[np.float64],
        clock_hz: float,
    ) -> None:
        self.w1, self.w2 = weights
        self.n_hinges = self.w1.shape[0] - 1 - 4
        self.target_x, self.target_y = float(target_xy[0]), float(target_xy[1])
        self.omega = 2 * math.pi * clock_hz  # same grouping as controller_inputs
        # Last entry of each buffer is the constant bias input 1.0.
        self.x = np.ones(self.w1.shape[0])
        self.h = np.ones(self.w2.shape[0])
        self.scale = np.pi / 2

    def __call__(self, data: mj.MjData, out: npt.NDArray[np.float64]) -> None:
        """Write the hinge commands for `data`'s current state into `out`."""
        x, nh = self.x, self.n_hinges
        qpos = data.qpos
        x[:nh] = qpos[FREE_JOINT_QPOS:]

        phase = self.omega * data.time
        x[nh] = math.sin(phase)
        x[nh + 1] = math.cos(phase)

        w, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        bearing = math.atan2(self.target_y - qpos[1], self.target_x - qpos[0]) - yaw
        x[nh + 2] = math.cos(bearing)
        x[nh + 3] = math.sin(bearing)

        np.tanh(x @ self.w1, out=self.h[:-1])
        np.multiply(np.tanh(self.h @ self.w2), self.scale, out=out)
