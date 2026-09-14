#!/usr/bin/env python3
"""Cross-validate our forward kinematics against Pinocchio.

    pip install pin
    PYTHONPATH=.:third_party/icrt pytest tests/test_pinocchio_parity.py

Why this matters more than it looks
-----------------------------------
`aiworker_icrt/kinematics.py` parses the URDF and composes transforms by hand.
It is already checked against numerical differentiation (the Jacobian agrees to
3.8e-07), but that only proves it is *self*-consistent — a frame convention error
would be differentiated just as faithfully as a correct one.

Pinocchio is a completely independent implementation, and it is the library
ROBOTIS's `cyclo_control` builds its `KinematicsSolver` on. So this check does
two jobs at once:

1. It rules out a whole class of silent errors — a sign flip in the URDF rpy
   convention, a units mismatch, the lift joint composed in the wrong order.
2. It proves the geometry our *dataset conversion* used is the same geometry
   cyclo_control's QP solver will see at execution time. If those disagreed,
   every Cartesian target the policy produced would be systematically offset
   from where the controller thought it was — and nothing else in the pipeline
   would notice.

Skips cleanly when Pinocchio is absent; it is deliberately not a hard dependency
(nothing in training or conversion needs it).

Measured 2026-09-14: 400 random configurations across both arms and the full
lift range agree to **6.7e-16 m** and **7.8e-16** in rotation — floating-point
limits, i.e. the two implementations are computing the same thing.

Run against two different pinocchio MAJORS: 4.1.0 natively, and 2.7.0 in the
container's sidecar (2.7.0 is the ceiling that resolves against numpy<2). Both
agree with our FK to ~1e-16. That strengthens rather than weakens the result — a
frame-convention error in our parser would have had to be replicated identically
in two implementations built years apart to survive both checks.
"""

import sys

import numpy as np

import pytest

pin = pytest.importorskip(
    "pinocchio",
    reason="pinocchio not importable in this interpreter — expected, and a SKIP "
           "is not a PASS. Every pinocchio binding (the `pin` wheel and "
           "ros-jazzy-pinocchio alike) links an eigenpy compiled against NumPy "
           "1.x, while this image ships NumPy 2.x; adding `pin` here installs "
           "cleanly and still fails at import. The check runs in a dedicated "
           "numpy<2 sidecar instead:  ./container.sh parity")

from aiworker_icrt import constants as C
from aiworker_icrt.kinematics import DEFAULT_URDF, FFWKinematics

TOL = 1e-9  # far looser than the 1e-16 observed; this is a convention check


def test_fk_matches_pinocchio() -> None:
    np.random.seed(0)
    K = FFWKinematics()
    model = pin.buildModelFromUrdf(DEFAULT_URDF)
    data = model.createData()
    print(f"pinocchio model: nq={model.nq} nv={model.nv}")

    base = model.getFrameId(C.FK_ROOT_LINK)
    eef = {'left': model.getFrameId(C.EEF_LINK_L),
           'right': model.getFrameId(C.EEF_LINK_R)}

    def pin_q(side, arm_q, lift):
        """Our chain-order joints -> pinocchio's q vector, by joint name."""
        q = pin.neutral(model)
        names = [C.LIFT_CHAIN_JOINT] + (C.ARM_L_JOINTS if side == 'left'
                                        else C.ARM_R_JOINTS)
        for n, v in zip(names, [lift] + list(arm_q)):
            q[model.joints[model.getJointId(n)].idx_q] = v
        return q

    pos_err, rot_err = [], []
    for side in ('left', 'right'):
        chain = K.left if side == 'left' else K.right
        lim = K.limits[side]
        for _ in range(200):
            arm_q = np.random.uniform(lim[:, 0], lim[:, 1])
            lift = np.random.uniform(-0.5, 0.0)

            T_ours = chain.fk(K._chain_q(side, arm_q, lift))

            pin.framesForwardKinematics(model, data, pin_q(side, arm_q, lift))
            # express the end-effector frame in base_link, matching our root
            M = data.oMf[base].actInv(data.oMf[eef[side]])

            pos_err.append(np.abs(T_ours[:3, 3] - M.translation).max())
            rot_err.append(np.abs(T_ours[:3, :3] - M.rotation).max())

    pos_err, rot_err = np.array(pos_err), np.array(rot_err)
    print(f"\n400 random configurations (both arms, full lift range):")
    print(f"  position  max {pos_err.max():.3e} m   mean {pos_err.mean():.3e}")
    print(f"  rotation  max {rot_err.max():.3e}     mean {rot_err.mean():.3e}")

    ok = pos_err.max() < TOL and rot_err.max() < TOL
    print(f"\n  {'PASS' if ok else 'FAIL'} — independent implementations "
          f"{'agree' if ok else 'DISAGREE: check frame/unit conventions'}")
    assert ok, (
        f'FK disagrees with Pinocchio: position {pos_err.max():.3e} m, '
        f'rotation {rot_err.max():.3e} — check frame/unit conventions')


if __name__ == '__main__':
    test_fk_matches_pinocchio()
    print('PASS')
