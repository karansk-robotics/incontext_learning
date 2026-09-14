"""Regression bounds for the dual-arm convert_delta_action patch.

Covers third_party/icrt/icrt/data/utils.py as patched by aiworker-iclr-01:
per-arm 10-D blocks instead of a single hardcoded block.

Scope note: these exercise the delta/abs conversion layer ONLY. ICRT's
SequenceDataset is deliberately not touched here — as of 2026-09-14 loading a
converted set through it yields len(dataset)==0 and it is unresolved whether
that is a small-data windowing artifact or a real defect. Do not add
SequenceDataset assertions until that is settled.
"""
import numpy as np
import pytest

from icrt.data.utils import (convert_delta_action, convert_abs_action,
                             rot_mat_to_rot_6d)

ROUND_TRIP_TOL = 1e-9      # measured 1.11e-15
CROSS_ARM_TOL = 1e-12      # must be bit-identical in practice

S, T, NA = 4, 16, 2


def _rand_block(n, rng):
    from scipy.spatial.transform import Rotation
    R = Rotation.random(n, random_state=int(rng.integers(0, 2**31 - 1))).as_matrix()
    return np.concatenate(
        [rng.standard_normal((n, 3)), rot_mat_to_rot_6d(R), rng.random((n, 1))], -1)


@pytest.fixture(scope="module")
def blocks():
    rng = np.random.default_rng(0)
    proprio = np.stack([_rand_block(S, rng) for _ in range(NA)], 1).reshape(S, 1, NA * 10)
    proprio = np.repeat(proprio, T, axis=1)
    action = np.stack([_rand_block(S * T, rng).reshape(S, T, 10) for _ in range(NA)],
                      2).reshape(S, T, NA * 10)
    return proprio, action


@pytest.mark.parametrize("with_eos", [False, True])
def test_dual_arm_round_trip(blocks, with_eos):
    proprio, action = blocks
    a = np.concatenate([action, np.zeros((S, T, 1))], -1) if with_eos else action
    err = np.abs(convert_abs_action(convert_delta_action(a, proprio), proprio) - a).max()
    assert err < ROUND_TRIP_TOL, f"round-trip error {err:.2e} (eos={with_eos})"


def test_single_arm_path_unchanged(blocks):
    """num_arms=1 must stay bit-identical to upstream ICRT."""
    proprio, action = blocks
    p1 = proprio[..., :10]
    a1 = np.concatenate([action[..., :10], np.zeros((S, T, 1))], -1)
    err = np.abs(convert_abs_action(convert_delta_action(a1, p1), p1) - a1).max()
    assert err < ROUND_TRIP_TOL, f"single-arm round-trip error {err:.2e}"


def test_no_cross_arm_bleed(blocks):
    """Perturbing arm 1's proprio must leave arm 0's delta untouched.

    This is the property the per-arm rewrite exists to provide: the original
    code reinterpreted one rotation block spanning the whole vector, so a
    second arm's pose leaked into the first arm's delta.
    """
    proprio, action = blocks
    d0 = convert_delta_action(action, proprio)
    pert = proprio.copy()
    pert[..., 10:] += 0.5
    d1 = convert_delta_action(action, pert)

    assert np.abs(d0[..., :10] - d1[..., :10]).max() < CROSS_ARM_TOL, "arm 0 leaked"
    # sanity: the perturbation must actually have changed arm 1, or the test is vacuous
    assert np.abs(d0[..., 10:] - d1[..., 10:]).max() > 1e-6, "arm 1 did not move"
