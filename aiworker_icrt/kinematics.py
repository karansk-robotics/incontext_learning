"""Dual-arm forward/inverse kinematics for the ROBOTIS AI Worker FFW-SG2.

Why this exists
---------------
ICRT was pretrained on DROID, whose action space is a single arm's end-effector
pose in Cartesian space.  The AI Worker records joint-space actions.  To reuse any
of that pretraining we retarget: joints -> EEF pose for training data, EEF pose ->
joints for execution.

Why the ``base_link`` frame
---------------------------
In ``ffw_sg2_follower.urdf`` the tree is

    base_link --[lift_joint, prismatic]--> arm_base_link --+--> head_joint1 ...
                                                           +--> arm_l_joint1 ... eef_l
                                                           +--> arm_r_joint1 ... eef_r

Poses are expressed in ``base_link``: the robot's canonical frame, shared with
odometry and navigation, and the same reference for both hands.

That puts ``lift_joint`` inside each arm's chain, so a chain carries **8**
actuated joints (one prismatic lift + seven revolute) and EEF height includes
the torso position. The lift is not part of the 20-D action space, so:

* **forward** (dataset conversion) reads the recorded lift and includes it;
* **inverse** (execution) holds the lift at its measured value and solves only
  the seven arm joints.

Holding it matters. With the lift free, an 8-DOF chain solving a 6-DOF pose could
return "raise the torso 5 cm" -- an answer the policy has no way to execute.

This module deliberately parses the URDF itself rather than pulling in ``kinpy``
(unmaintained, and ICRT only uses it in an eval script).  A serial chain of
revolute joints and its analytic Jacobian are short enough to own outright, and
owning them means the IK damping and nullspace behaviour are ours to tune.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from aiworker_icrt import constants as C

# Reuse ICRT's own 6D helpers rather than reimplementing them. ICRT stores the
# first two ROWS of the rotation matrix (icrt/data/utils.py:13), which is the
# transpose of the more common Zhou et al. column convention -- and the model's
# re-orthogonalisation at inference (icrt/models/policy/icrt.py:706) assumes it.
# Importing keeps the two halves from silently drifting apart.
from icrt.data.utils import rot_6d_to_rot_mat, rot_mat_to_rot_6d

DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'assets', 'ffw_sg2', 'ffw_sg2_follower.urdf',
)


# --------------------------------------------------------------------------- #
# URDF parsing
# --------------------------------------------------------------------------- #

@dataclass
class _Joint:
    name: str
    jtype: str
    axis: np.ndarray          # (3,) unit axis in the joint's own frame
    origin_xyz: np.ndarray    # (3,) fixed offset from parent link
    origin_rot: np.ndarray    # (3, 3) fixed rotation from parent link
    lower: Optional[float]
    upper: Optional[float]

    @property
    def actuated(self) -> bool:
        return self.jtype in ('revolute', 'continuous', 'prismatic')


def _parse_origin(elem) -> Tuple[np.ndarray, np.ndarray]:
    origin = elem.find('origin')
    if origin is None:
        return np.zeros(3), np.eye(3)
    xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
    # URDF rpy is extrinsic XYZ, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    # scipy's lowercase 'xyz' is exactly that.
    return xyz, Rotation.from_euler('xyz', rpy).as_matrix()


class SerialChain:
    """A root->tip serial chain extracted from a URDF, with FK and Jacobian."""

    def __init__(self, urdf_path: str, root_link: str, tip_link: str):
        root = ET.parse(urdf_path).getroot()

        by_child = {}
        for j in root.findall('joint'):
            axis_elem = j.find('axis')
            axis = (np.fromstring(axis_elem.get('xyz'), sep=' ')
                    if axis_elem is not None else np.array([1.0, 0.0, 0.0]))
            n = np.linalg.norm(axis)
            if n > 0:
                axis = axis / n
            xyz, rot = _parse_origin(j)
            lim = j.find('limit')
            by_child[j.find('child').get('link')] = (
                j.find('parent').get('link'),
                _Joint(
                    name=j.get('name'),
                    jtype=j.get('type'),
                    axis=axis,
                    origin_xyz=xyz,
                    origin_rot=rot,
                    lower=float(lim.get('lower')) if lim is not None and lim.get('lower') else None,
                    upper=float(lim.get('upper')) if lim is not None and lim.get('upper') else None,
                ),
            )

        # Walk up from the tip until we hit the root, then reverse.
        chain: List[_Joint] = []
        link = tip_link
        while link != root_link:
            if link not in by_child:
                raise ValueError(
                    f'link {link!r} has no parent joint; {tip_link!r} is not a '
                    f'descendant of {root_link!r} in {urdf_path}'
                )
            parent, joint = by_child[link]
            chain.append(joint)
            link = parent
        chain.reverse()

        self.root_link = root_link
        self.tip_link = tip_link
        self.joints = chain
        self.actuated_idx = [i for i, j in enumerate(chain) if j.actuated]
        self.joint_names = [chain[i].name for i in self.actuated_idx]
        self.n = len(self.actuated_idx)

    # -- forward kinematics ------------------------------------------------- #

    def _joint_transforms(self, q: np.ndarray) -> np.ndarray:
        """Per-joint 4x4 transforms for a batch of configurations.

        q: (B, n) -> (B, len(self.joints), 4, 4)
        """
        B = q.shape[0]
        out = np.zeros((B, len(self.joints), 4, 4))
        out[..., 3, 3] = 1.0

        qi = 0
        for k, joint in enumerate(self.joints):
            R = np.broadcast_to(joint.origin_rot, (B, 3, 3))
            t = np.broadcast_to(joint.origin_xyz, (B, 3))
            if joint.actuated:
                theta = q[:, qi]
                qi += 1
                if joint.jtype == 'prismatic':
                    t = t + theta[:, None] * joint.axis[None, :]
                else:
                    # Rodrigues about the joint's own axis, applied after origin.
                    rot = Rotation.from_rotvec(theta[:, None] * joint.axis[None, :])
                    R = R @ rot.as_matrix()
            out[:, k, :3, :3] = R
            out[:, k, :3, 3] = t
        return out

    def fk_batch(self, q: np.ndarray) -> np.ndarray:
        """q: (B, n) -> (B, 4, 4) pose of tip_link in root_link."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        mats = self._joint_transforms(q)
        T = mats[:, 0]
        for k in range(1, mats.shape[1]):
            T = T @ mats[:, k]
        return T

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """q: (n,) -> (4, 4)."""
        return self.fk_batch(np.asarray(q, dtype=np.float64)[None])[0]

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        """Geometric Jacobian in the root frame. q: (n,) -> (6, n).

        Rows 0-2 are linear, rows 3-5 angular.
        """
        q = np.asarray(q, dtype=np.float64)[None]
        mats = self._joint_transforms(q)[0]

        # Cumulative pose at each joint, and the tip.
        cum = np.empty((len(self.joints), 4, 4))
        T = np.eye(4)
        for k in range(len(self.joints)):
            T = T @ mats[k]
            cum[k] = T
        p_ee = cum[-1][:3, 3]

        J = np.zeros((6, self.n))
        for col, k in enumerate(self.actuated_idx):
            joint = self.joints[k]
            axis_world = cum[k][:3, :3] @ joint.axis
            if joint.jtype == 'prismatic':
                J[:3, col] = axis_world
            else:
                J[:3, col] = np.cross(axis_world, p_ee - cum[k][:3, 3])
                J[3:, col] = axis_world
        return J


# --------------------------------------------------------------------------- #
# Dual-arm wrapper
# --------------------------------------------------------------------------- #

def _pose_error(T_cur: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """6-vector twist taking T_cur to T_target, expressed in the root frame."""
    e = np.empty(6)
    e[:3] = T_target[:3, 3] - T_cur[:3, 3]
    R_err = T_target[:3, :3] @ T_cur[:3, :3].T
    e[3:] = Rotation.from_matrix(R_err).as_rotvec()
    return e


class FFWKinematics:
    """Joint <-> Cartesian retargeting for the FFW-SG2's two 7-DOF arms."""

    def __init__(self, urdf_path: str = DEFAULT_URDF):
        self.urdf_path = urdf_path
        self.left = SerialChain(urdf_path, C.FK_ROOT_LINK, C.EEF_LINK_L)
        self.right = SerialChain(urdf_path, C.FK_ROOT_LINK, C.EEF_LINK_R)

        # From base_link each chain is [lift_joint, arm_X_joint1..7].
        self.arm_idx, self.lift_idx = {}, {}
        for side, chain, expected in (
            ('left', self.left, C.ARM_L_JOINTS),
            ('right', self.right, C.ARM_R_JOINTS),
        ):
            if chain.joint_names != [C.LIFT_CHAIN_JOINT] + expected:
                raise ValueError(
                    f'{side} chain actuated joints {chain.joint_names} do not match '
                    f'the expected [{C.LIFT_CHAIN_JOINT}] + {expected}'
                )
            self.lift_idx[side] = chain.joint_names.index(C.LIFT_CHAIN_JOINT)
            self.arm_idx[side] = [i for i, n in enumerate(chain.joint_names)
                                  if n != C.LIFT_CHAIN_JOINT]

        self.limits = {
            'left': np.asarray(C.ARM_JOINT_LIMITS_L, dtype=np.float64),
            'right': np.asarray(C.ARM_JOINT_LIMITS_R, dtype=np.float64),
        }
        # Mid-range posture, used as the IK nullspace attractor.
        self.rest = {k: v.mean(axis=1) for k, v in self.limits.items()}

    def _chain_q(self, side: str, arm_q: np.ndarray, lift: float) -> np.ndarray:
        """Assemble a chain-order vector: the 7 arm joints plus the lift."""
        n = self.left.n if side == 'left' else self.right.n
        q = np.empty(n)
        q[self.lift_idx[side]] = lift
        q[self.arm_idx[side]] = arm_q
        return q

    # -- gripper scaling ---------------------------------------------------- #

    @staticmethod
    def gripper_to_unit(q: np.ndarray) -> np.ndarray:
        """Raw RH-P12-RN joint -> [0, 1], preserving sense: 0 open, 1 closed."""
        lo, hi = C.GRIPPER_LIMITS
        return np.clip((np.asarray(q) - lo) / (hi - lo), 0.0, 1.0)

    @staticmethod
    def gripper_from_unit(u: np.ndarray) -> np.ndarray:
        """[0, 1] (0 open, 1 closed) -> raw RH-P12-RN joint value."""
        lo, hi = C.GRIPPER_LIMITS
        return np.clip(np.asarray(u), 0.0, 1.0) * (hi - lo) + lo

    # -- joints -> Cartesian ------------------------------------------------ #

    def joints_to_cartesian(self, joints: np.ndarray) -> np.ndarray:
        """(..., 22) recorded joint vector -> (..., 20) Cartesian vector.

        Vectorised over leading dimensions so a whole episode converts in one call.
        """
        joints = np.asarray(joints, dtype=np.float64)
        lead = joints.shape[:-1]
        flat = joints.reshape(-1, joints.shape[-1])
        if flat.shape[-1] < C.ARMS_ONLY_DIM:
            raise ValueError(
                f'joint vector is {flat.shape[-1]}-D; need at least '
                f'{C.ARMS_ONLY_DIM} (both arms + grippers). Widths of '
                f'{C.ARMS_ONLY_DIM} (arms only) and {C.JOINT_DIM} (full FFW-SG2) '
                f'are both supported -- the arm/gripper block is a common prefix.'
            )

        # Recordings narrower than the full layout (e.g. the 16-D arms-only sets
        # ROBOTIS publishes) carry no lift column; those robots have no torso to
        # raise, so zero is the correct reading, not a fallback.
        lift = (flat[:, C.SLICE_LIFT] if flat.shape[-1] > C.SLICE_LIFT
                else np.zeros(flat.shape[0]))

        out = np.empty((flat.shape[0], C.CARTESIAN_DIM))
        for side, chain, arm_sl, grip_i, cart_sl in (
            ('left', self.left, C.SLICE_ARM_L, C.SLICE_GRIP_L, C.SLICE_CART_L),
            ('right', self.right, C.SLICE_ARM_R, C.SLICE_GRIP_R, C.SLICE_CART_R),
        ):
            q = np.empty((flat.shape[0], chain.n))
            q[:, self.lift_idx[side]] = lift
            q[:, self.arm_idx[side]] = flat[:, arm_sl]
            T = chain.fk_batch(q)
            block = out[:, cart_sl]
            block[:, C.SLICE_BLOCK_POS] = T[:, :3, 3]
            block[:, C.SLICE_BLOCK_ROT6D] = rot_mat_to_rot_6d(T[:, :3, :3])
            block[:, C.BLOCK_GRIPPER] = self.gripper_to_unit(flat[:, grip_i])
        return out.reshape(*lead, C.CARTESIAN_DIM)

    # -- proprio / action packing ------------------------------------------- #

    def joints_to_proprio(self, joints: np.ndarray) -> np.ndarray:
        """(..., 22) recorded joints -> (..., 23) proprio.

        The 20-D Cartesian arm blocks, then head(2) and lift(1) appended raw.
        Those three are observed, not commanded: the head so the policy can
        interpret what cam_head is showing it, the lift because with poses in
        base_link the torso height changes where the hands are.
        """
        joints = np.asarray(joints, dtype=np.float64)
        cart = self.joints_to_cartesian(joints)
        extra = joints[..., C.PROPRIO_EXTRA_SRC]
        return np.concatenate([cart, extra], axis=-1)

    def joints_to_action(self, joints: np.ndarray) -> np.ndarray:
        """(..., 22) recorded joints -> (..., 21) action: arms + lift.

        The lift is commanded, the head is not. If a demonstrator raised the
        torso to reach something, that is part of the skill; the head only
        changes the view.
        """
        joints = np.asarray(joints, dtype=np.float64)
        cart = self.joints_to_cartesian(joints)
        extra = joints[..., C.ACTION_EXTRA_SRC]
        return np.concatenate([cart, extra], axis=-1)

    def action_to_joints(self, action: np.ndarray, joint_seed: np.ndarray,
                         **ik_kwargs) -> Tuple[np.ndarray, dict]:
        """(21,) action -> (22,) joint command.

        The commanded lift is applied FIRST, then IK solves the arms against that
        torso height. Order matters: solving the arms against the old lift and
        then moving the torso would displace the hands by the lift delta.
        """
        action = np.asarray(action, dtype=np.float64)
        seed = np.asarray(joint_seed, dtype=np.float64).copy()
        seed[C.SLICE_LIFT] = float(action[C.SLICE_ACTION_EXTRA][0])
        out, info = self.cartesian_to_joints(action[:C.CARTESIAN_DIM], seed, **ik_kwargs)
        info['lift_commanded'] = float(seed[C.SLICE_LIFT])
        return out, info

    # -- Cartesian -> joints ------------------------------------------------ #

    def _ik_arm_once(
        self,
        which: str,
        T_target: np.ndarray,
        q_seed: np.ndarray,
        lift: float = 0.0,
        max_iters: int = 100,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        damping: float = 0.01,
        nullspace_gain: float = 0.05,
        nullspace_decay: float = 0.05,
        step_clip: float = 0.2,
    ) -> Tuple[np.ndarray, dict]:
        """Damped least-squares IK with joint-limit clamping and a rest-posture
        nullspace bias.

        The arms are 7-DOF, so one DOF is redundant for a 6-DOF pose target. The
        nullspace term spends it on staying near mid-range, which keeps solutions
        continuous across a trajectory instead of flipping elbow configuration
        between neighbouring timesteps.

        The nullspace projector is built from the *damped* pseudo-inverse, so it
        is only approximate and leaks a little of the rest-posture pull back into
        task space. Left unchecked that leak balances against the task correction
        and the solver parks at a ~1mm offset instead of converging. Fading the
        secondary objective out as the task error shrinks (``nullspace_decay``)
        keeps the posture bias where it is useful -- early, while the arm is far
        from the target -- and hands the last millimetre to the task alone.
        """
        chain = self.left if which == 'left' else self.right
        lim = self.limits[which]
        rest = self.rest[which]
        arm_idx = self.arm_idx[which]

        q = np.clip(np.asarray(q_seed, dtype=np.float64).copy(), lim[:, 0], lim[:, 1])
        err = None
        for it in range(max_iters):
            T_cur = chain.fk(self._chain_q(which, q, lift))
            err = _pose_error(T_cur, T_target)
            if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < rot_tol:
                return q, {'iters': it, 'converged': True,
                           'pos_err': float(np.linalg.norm(err[:3])),
                           'rot_err': float(np.linalg.norm(err[3:]))}

            # Slice out the lift column: the torso is not ours to command, so
            # its DOF must not participate in the solve. Leaving it in would let
            # an 8-DOF chain satisfy a 6-DOF target by raising the torso.
            J = chain.jacobian(self._chain_q(which, q, lift))[:, arm_idx]
            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)

            # Redundancy resolution: project a pull toward rest into the nullspace,
            # faded out as the task error shrinks so the approximate projector's
            # leak cannot stall convergence.
            err_mag = np.linalg.norm(err)
            weight = min(1.0, err_mag / nullspace_decay)
            if weight > 0.0:
                J_pinv = J.T @ np.linalg.inv(JJt)
                dq += (np.eye(len(arm_idx)) - J_pinv @ J) @ (
                    weight * nullspace_gain * (rest - q)
                )

            # Joint-limit handling. Simply clipping q after the step lets the
            # solver jam: a joint sits on its bound, the step keeps pushing
            # outward, and the clip discards it every iteration while the other
            # joints are solved against a correction that assumed it moved. Zero
            # the outward-pushing components first so the remaining DOFs take up
            # the slack, which is the cheap half of the usual clamping loop.
            at_lower = (q <= lim[:, 0] + 1e-9) & (dq < 0)
            at_upper = (q >= lim[:, 1] - 1e-9) & (dq > 0)
            dq[at_lower | at_upper] = 0.0

            norm = np.linalg.norm(dq)
            if norm > step_clip:
                dq *= step_clip / norm
            q = np.clip(q + dq, lim[:, 0], lim[:, 1])

        return q, {'iters': max_iters, 'converged': False,
                   'pos_err': float(np.linalg.norm(err[:3])),
                   'rot_err': float(np.linalg.norm(err[3:]))}

    def _ik_arm(
        self,
        which: str,
        T_target: np.ndarray,
        q_seed: np.ndarray,
        lift: float = 0.0,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        """IK with restarts.

        In normal use the seed is the previous timestep's solution, a few
        millimetres away, and the first attempt converges in a handful of
        iterations. The restarts only matter for the first frame of an episode
        and for targets a regression head puts near a singularity: retrying from
        mid-range escapes a basin the seed was trapped in, at a cost we only pay
        on the rare failure.
        """
        q, info = self._ik_arm_once(which, T_target, q_seed, lift, **kwargs)
        if info['converged']:
            info['restarts'] = 0
            return q, info

        best_q, best_info = q, info
        for attempt, alt_seed in enumerate(self._restart_seeds(which, q_seed), start=1):
            q, info = self._ik_arm_once(which, T_target, alt_seed, lift, **kwargs)
            if info['converged']:
                info['restarts'] = attempt
                return q, info
            if info['pos_err'] < best_info['pos_err']:
                best_q, best_info = q, info

        best_info['restarts'] = len(self._restart_seeds(which, q_seed))
        return best_q, best_info

    def _restart_seeds(self, which: str, q_seed: np.ndarray) -> List[np.ndarray]:
        """Alternate seeds tried when the primary one fails to converge."""
        lim = self.limits[which]
        rng = np.random.default_rng(0)  # deterministic: same input, same output
        return [
            self.rest[which],
            np.clip(np.asarray(q_seed) + rng.normal(0, 0.3, lim.shape[0]),
                    lim[:, 0], lim[:, 1]),
        ]

    def cartesian_to_joints(
        self,
        cart: np.ndarray,
        joint_seed: np.ndarray,
        **ik_kwargs,
    ) -> Tuple[np.ndarray, dict]:
        """(20,) Cartesian -> (22,) joint vector, seeded from the current pose.

        Head, lift and mobile-base entries are copied straight from ``joint_seed``:
        this action space does not control them, so they hold whatever the robot
        was at when the episode started.
        """
        cart = np.asarray(cart, dtype=np.float64)
        out = np.asarray(joint_seed, dtype=np.float64).copy()
        # The lift is held, not solved: whatever the robot's torso is at now is
        # what the IK targets are interpreted against, and out[SLICE_LIFT] is
        # passed through unchanged.
        lift = float(out[C.SLICE_LIFT]) if out.shape[-1] > C.SLICE_LIFT else 0.0
        info = {'lift_held': lift}
        for which, arm_sl, grip_i, cart_sl in (
            ('left', C.SLICE_ARM_L, C.SLICE_GRIP_L, C.SLICE_CART_L),
            ('right', C.SLICE_ARM_R, C.SLICE_GRIP_R, C.SLICE_CART_R),
        ):
            block = cart[cart_sl]
            T = np.eye(4)
            T[:3, 3] = block[C.SLICE_BLOCK_POS]
            # rot_6d_to_rot_mat re-orthogonalises via Gram-Schmidt when needed,
            # which matters because these come from a regression head.
            T[:3, :3] = rot_6d_to_rot_mat(block[C.SLICE_BLOCK_ROT6D][None])[0]
            q, meta = self._ik_arm(which, T, out[arm_sl], lift, **ik_kwargs)
            out[arm_sl] = q
            out[grip_i] = self.gripper_from_unit(block[C.BLOCK_GRIPPER])
            info[which] = meta
        return out, info
