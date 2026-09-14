"""Regression bounds for the FFW-SG2 kinematics.

Thresholds come from tests/README.md, measured 2026-09-14 by aiworker-iclr-01.
They are set with headroom over the measured values so ordinary numerical
drift does not fail the suite, but a real regression does.

UPDATED 2026-09-14 for the base_link change: EEF poses are now referenced from
base_link rather than arm_base_link, so each arm chain carries 8 actuated joints
([lift_joint] + the 7 arm joints) and EEF z shifts by +1.4316 m. The lift is NOT
commanded - FK reads it, IK holds it fixed and solves only the 7 arm joints.
Raw chain.fk/jacobian/fk_batch therefore take 8-vectors in chain order; use
K._chain_q(side, arm_q7, lift) to assemble one. The public joints_to_cartesian /
cartesian_to_joints path still takes the 22-D vector and is unchanged.

DELIBERATELY NOT TIGHTENED: the random-seed IK stress test allows a small
number of non-convergences. Damped least squares trades exactness for
stability near singularities by design, and the measured misses are all
near-singular. Do not "fix" the solver to make this green — see
test_ik_nonconvergence_is_singular_only, which asserts the failures are
*caused by* conditioning rather than just tolerating them.
"""
import numpy as np
import pytest

# --- measured baselines (README) -> asserted bounds ------------------------ #
JAC_MAX_DIFF = 1e-5      # measured 3.83e-07
IK_WARM_POS_MAX = 2e-4   # measured <100 um; 2x headroom
IK_WARM_MEAN_ITERS = 3.0 # measured 1.2-1.5
IK_STRESS_POS_MAX = 2e-4 # CONVERGED cases only; measured max 0.0999 mm
IK_STRESS_MAX_MISS = 0.02  # <=2% of seeds may miss the iteration budget (measured 2/300)


def _num_jac(chain, q, eps=1e-6):
    """q is a chain-order vector of length chain.n (8: lift + 7 arm joints)."""
    from aiworker_icrt.kinematics import _pose_error
    J = np.zeros((6, chain.n))
    T0 = chain.fk(q)
    for i in range(chain.n):
        qp = q.copy()
        qp[i] += eps
        J[:, i] = _pose_error(T0, chain.fk(qp)) / eps
    return J


def test_chain_structure(K, C):
    """Chains are rooted at base_link, so the prismatic lift is unavoidably in
    them - it sits between base_link and arm_base_link."""
    assert K.left.joint_names == [C.LIFT_CHAIN_JOINT] + C.ARM_L_JOINTS
    assert K.right.joint_names == [C.LIFT_CHAIN_JOINT] + C.ARM_R_JOINTS
    assert K.left.n == K.right.n == 8
    # limits/rest stay arm-only: the lift is never solved for
    assert K.limits["left"].shape == (7, 2)
    assert K.rest["left"].shape == (7,)


def test_zero_pose_mirrored_in_y(K):
    Tl = K.left.fk(K._chain_q("left", np.zeros(7), 0.0))
    Tr = K.right.fk(K._chain_q("right", np.zeros(7), 0.0))
    assert np.allclose(Tl[:3, 3] * [1, -1, 1], Tr[:3, 3], atol=1e-9), (
        f"L={Tl[:3, 3]} R={Tr[:3, 3]}")


def test_poses_are_referenced_from_base_link(K):
    """Guards the base_link change itself. Referenced from arm_base_link the
    zero-pose EEF sat at z = -0.8565; from base_link it is +1.4316 m higher."""
    z = K.left.fk(K._chain_q("left", np.zeros(7), 0.0))[2, 3]
    assert z > 0.0, f"EEF z={z:.4f} - looks rooted at arm_base_link, not base_link"
    assert abs(z - 0.5751) < 1e-3, f"unexpected zero-pose height {z:.4f}"


@pytest.mark.parametrize("which", ["left", "right"])
def test_analytic_jacobian_matches_numerical(K, which):
    rng = np.random.default_rng(0)
    chain = getattr(K, which)
    lim = K.limits[which]
    worst = 0.0
    for _ in range(20):
        # vary the lift too - its column is part of the analytic Jacobian even
        # though IK never solves for it
        q = K._chain_q(which, rng.uniform(lim[:, 0], lim[:, 1]), rng.uniform(-0.3, 0.0))
        worst = max(worst, np.abs(chain.jacobian(q) - _num_jac(chain, q)).max())
    assert worst < JAC_MAX_DIFF, f"max |analytic - numerical| = {worst:.2e}"


def _ik_stress(K, which, n, rng, lift=-0.25):
    """Random-seed IK with a 0.25 rad seed error. Returns per-trial records.

    `lift` is held fixed throughout, which is how the real pipeline uses it -
    IK solves the 7 arm joints only and slices the lift column out of the
    Jacobian.
    """
    from aiworker_icrt.kinematics import _pose_error
    chain = getattr(K, which)
    lim = K.limits[which]
    arm_cols = K.arm_idx[which]
    out = []
    for _ in range(n):
        q_true = rng.uniform(lim[:, 0] * 0.8, lim[:, 1] * 0.8)
        T = chain.fk(K._chain_q(which, q_true, lift))
        seed = np.clip(q_true + rng.normal(0, 0.25, 7), lim[:, 0], lim[:, 1])
        q, info = K._ik_arm(which, T, seed, lift)
        err = _pose_error(chain.fk(K._chain_q(which, q, lift)), T)
        # Conditioning of the matrix IK actually inverts: the ARM columns only,
        # not the lift column, which is sliced out before the solve.
        J_arm = chain.jacobian(K._chain_q(which, q, lift))[:, arm_cols]
        out.append({
            "converged": bool(info["converged"]),
            "pos": float(np.linalg.norm(err[:3])),
            "rot": float(np.linalg.norm(err[3:])),
            "sigma_min": float(np.linalg.svd(J_arm, compute_uv=False)[-1]),
        })
    return out


def test_ik_stress_accuracy(K):
    rng = np.random.default_rng(0)
    recs = _ik_stress(K, "left", 60, rng) + _ik_stress(K, "right", 60, rng)
    miss = sum(not r["converged"] for r in recs)
    # Accuracy is only meaningful for solves that CONVERGED. A non-converged
    # solve returns wherever the iteration budget ran out (measured residuals of
    # 30-72 mm), and asserting a tolerance on it would say nothing.
    conv = [r for r in recs if r["converged"]]
    pos = np.array([r["pos"] for r in conv])
    rot = np.array([r["rot"] for r in conv])

    assert miss / len(recs) <= IK_STRESS_MAX_MISS, (
        f"{miss}/{len(recs)} seeds missed the iteration budget")
    assert conv, "no seeds converged at all"
    assert pos.max() < IK_STRESS_POS_MAX, (
        f"worst CONVERGED position error {pos.max()*1000:.4f} mm")
    assert rot.max() < 5e-3, f"worst CONVERGED rotation error {rot.max()*1000:.3f} mrad"


def test_ik_nonconvergence_is_singular_only(K):
    """Non-convergence must be explained by conditioning, not by a bad solver.

    This is the test that stops someone from silently loosening the budget: if
    misses start happening at well-conditioned poses, that IS a regression even
    though the count is still inside IK_STRESS_MAX_MISS.
    """
    rng = np.random.default_rng(0)
    recs = _ik_stress(K, "left", 60, rng) + _ik_stress(K, "right", 60, rng)
    missed = [r for r in recs if not r["converged"]]
    if not missed:
        pytest.skip("no non-convergences in this sample — nothing to attribute")
    median_sigma = float(np.median([r["sigma_min"] for r in recs]))
    for r in missed:
        assert r["sigma_min"] <= median_sigma, (
            f"non-convergence at a well-conditioned pose "
            f"(sigma_min={r['sigma_min']:.4g} > median {median_sigma:.4g}) — "
            "this is a solver regression, not a singularity")


@pytest.mark.parametrize("which", ["left", "right"])
def test_ik_warm_started_tracking(K, C, which):
    """The regime that actually runs at inference: warm start from the previous
    solution along a smooth trajectory. These bounds are the hard ones."""
    from aiworker_icrt.kinematics import _pose_error
    rng = np.random.default_rng(1)
    chain = getattr(K, which)
    lim = K.limits[which]
    steps = 500

    lift = -0.25
    q0 = np.clip(K.rest[which] + rng.normal(0, 0.2, 7), lim[:, 0], lim[:, 1])
    t = np.linspace(0, 4 * np.pi, steps)
    traj = np.clip(q0[None, :] + 0.3 * np.sin(t[:, None] + np.linspace(0, 2, 7)[None, :]),
                   lim[:, 0], lim[:, 1])
    traj8 = np.stack([K._chain_q(which, a, lift) for a in traj])
    targets = chain.fk_batch(traj8)

    seed = traj[0].copy()
    miss, iters, pos = 0, [], []
    for k in range(steps):
        q, info = K._ik_arm(which, targets[k], seed, lift)
        seed = q
        miss += not info["converged"]
        iters.append(info["iters"])
        pos.append(np.linalg.norm(
            _pose_error(chain.fk(K._chain_q(which, q, lift)), targets[k])[:3]))
    pos = np.array(pos)

    assert miss == 0, f"{miss}/{steps} non-converged when warm-started"
    assert np.mean(iters) < IK_WARM_MEAN_ITERS, f"mean iters {np.mean(iters):.2f}"
    assert pos.max() < IK_WARM_POS_MAX, f"max position error {pos.max()*1e6:.1f} um"


def test_tracking_respects_velocity_limit(K, C):
    """No elbow flips: adjacent IK solutions must stay inside the joint
    velocity limit at the 15 Hz control rate."""
    rng = np.random.default_rng(1)
    chain, lim = K.left, K.limits["left"]
    lift = -0.25
    q0 = np.clip(K.rest["left"], lim[:, 0], lim[:, 1])
    t = np.linspace(0, 2 * np.pi, 200)
    traj = np.clip(q0[None, :] + 0.3 * np.sin(t[:, None] + np.linspace(0, 2, 7)[None, :]),
                   lim[:, 0], lim[:, 1])
    targets = chain.fk_batch(np.stack([K._chain_q("left", a, lift) for a in traj]))

    seed = traj[0].copy()
    jumps = []
    for k in range(200):
        q, _ = K._ik_arm("left", targets[k], seed, lift)
        jumps.append(np.abs(q - seed).max())
        seed = q

    rate_hz = 15.0
    worst = max(jumps) * rate_hz
    assert worst < C.JOINT_VELOCITY_LIMIT, (
        f"{worst:.2f} rad/s at {rate_hz} Hz exceeds {C.JOINT_VELOCITY_LIMIT}")


def test_full_22_to_20_to_22_round_trip(K, C):
    rng = np.random.default_rng(0)
    q22 = np.zeros(C.JOINT_DIM)
    q22[C.SLICE_ARM_L] = rng.uniform(K.limits["left"][:, 0] * 0.7,
                                     K.limits["left"][:, 1] * 0.7)
    q22[C.SLICE_ARM_R] = rng.uniform(K.limits["right"][:, 0] * 0.7,
                                     K.limits["right"][:, 1] * 0.7)
    q22[C.SLICE_GRIP_L] = 0.55
    q22[C.SLICE_GRIP_R] = 0.0
    q22[C.SLICE_HEAD] = [0.1, -0.2]
    q22[C.SLICE_LIFT] = -0.3
    q22[C.SLICE_BASE] = [0.1, 0.0, 0.2]

    cart = K.joints_to_cartesian(q22)
    assert cart.shape == (C.CARTESIAN_DIM,)

    seed = q22.copy()
    seed[C.SLICE_ARM_L] += rng.normal(0, 0.15, 7)
    seed[C.SLICE_ARM_R] += rng.normal(0, 0.15, 7)
    q_rt, _ = K.cartesian_to_joints(cart, seed)

    assert np.abs(K.joints_to_cartesian(q_rt) - cart).max() < 1e-3
    assert abs(q_rt[C.SLICE_GRIP_L] - 0.55) < 1e-6

    # head / lift / base are held, never commanded — they must pass through
    assert np.allclose(q_rt[C.SLICE_HEAD], seed[C.SLICE_HEAD])
    assert q_rt[C.SLICE_LIFT] == seed[C.SLICE_LIFT]
    assert np.allclose(q_rt[C.SLICE_BASE], seed[C.SLICE_BASE])

    # and the solution must be inside the limits
    lo, hi = K.limits["left"][:, 0], K.limits["left"][:, 1]
    assert np.all(q_rt[C.SLICE_ARM_L] >= lo - 1e-9)
    assert np.all(q_rt[C.SLICE_ARM_L] <= hi + 1e-9)


def test_batch_fk_shape(K, C):
    rng = np.random.default_rng(0)
    ep = np.tile(np.zeros(C.JOINT_DIM), (500, 1)) + rng.normal(0, 0.01, (500, C.JOINT_DIM))
    assert K.joints_to_cartesian(ep).shape == (500, C.CARTESIAN_DIM)
