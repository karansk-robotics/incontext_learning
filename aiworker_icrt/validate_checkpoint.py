#!/usr/bin/env python3
"""Pre-hardware safety gate: would this checkpoint command something dangerous?

    python -m aiworker_icrt.validate_checkpoint \
        --dataset /dev/shm/icrt_multitask \
        --checkpoint runs/mt21_gpu/checkpoint-15.pth \
        --train-yaml runs/mt21_gpu/run.yaml \
        --out-dir runs/mt21_gpu/validation

This is a DIFFERENT question from the one `evaluate.py` answers. evaluate.py asks
"does the policy reproduce the demonstration" -- accuracy. This asks "if we had
wired this to a robot, would anything have broken" -- safety. A policy can score
a respectable position MAE while still emitting a single 40 cm jump between two
consecutive steps, and the averaged metric hides it completely.

Nothing here drives a robot and nothing here needs a simulator. Every check is
kinematic, computed from the same `policy.step` call that `ros_node` would make.
That matters: this project has repeatedly been bitten by checks that passed
against a proxy for the consumer rather than the consumer itself (see
validations.md section 5). The rollout below goes through DualArmICRT.step, so
the IK warm-starting, the live-lift seeding and the per-arm Gram-Schmidt are all
exercised exactly as they would be on hardware.

WHAT IS CHECKED

  ik_reachable     Whether the requested pose is reachable AT ALL, measured by
                   IK residual -- deliberately NOT by the solver's `converged`
                   flag, which fires on any pose that is not exactly FK-consistent
                   and so condemns every learned policy. Calibrated against
                   perturbed ground truth.
  joint_limits     A target OUTSIDE the URDF range is clamped by the driver, so
                   the arm stalls against a limit while the policy keeps
                   integrating -- the classic runaway. Merely touching a bound is
                   normal here (joint2 rests on its limit) and is reported as
                   `saturated_steps`, not as a failure.
  joint_velocity   Consecutive targets imply a speed. Above the URDF's 4.8 rad/s
                   the controller either saturates or faults. Verified to be a
                   real signal rather than a solver artifact: perturbing recorded
                   poses by 3.6 mm leaves this clean (0.90x cap, 0 violations),
                   while 45 mm of error produces 3.56x and 77 violations. It
                   tracks policy accuracy, which is what we want it to do.
  eef_speed        The Cartesian equivalent, and the one a human in the cell
                   actually feels. Default cap 0.5 m/s is conservative.
  gripper_chatter  Rapid open/close flips wear the RH-P12-RN and signal an
                   undecided policy. Counted as flips per second.
  lift_command     The lift is IN the action space but was constant across all
                   training episodes, so a nonzero lift command is out of
                   distribution by construction. Torso motion is the one failure
                   here with real injury potential.
  arm_proximity    Distance between the two end-effectors. NOT a full
                   self-collision check -- we only have EEF poses from FK, not
                   every link -- so treat it as a smoke alarm, not a guarantee.

A FAIL means do not put this checkpoint on a robot. A PASS means none of the
above fired on the episodes tested; it does not mean the policy is good, which
is what evaluate.py is for.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from aiworker_icrt import constants as C

# Defaults chosen to be conservative rather than typical: this gate exists to
# stop a bad checkpoint, so it should complain before hardware does.
EEF_SPEED_MAX = 0.5          # m/s, per arm
JOINT_VEL_MARGIN = 0.8       # fraction of the URDF velocity limit
JOINT_LIMIT_MARGIN = 0.02    # rad; below this a joint is reported SATURATED (informational)
JOINT_LIMIT_EPS = 1e-6       # rad; below -this a joint is actually OUTSIDE its range (a failure)
GRIPPER_CHATTER_MAX = 2.0    # open/close flips per second
LIFT_COMMAND_MAX = 0.005     # m, essentially "must not move"
ARM_PROXIMITY_MIN = 0.05     # m between the two EEF origins
# Calibrated, not guessed. Perturbing RECORDED poses by 1 mm drops our DLS
# solver from 80/80 converged to 60/80, and the residual then tracks the
# perturbation almost exactly (3.6 mm -> 3.995 mm, 45 mm -> 39.7 mm). The solver
# is not failing; it is reporting how far the target sits off the FK manifold.
# A regression head NEVER lands exactly on that manifold, so a tight tolerance
# fails every learned policy by construction. This threshold is set where the
# residual means "genuinely unreachable" rather than "not FK-exact".
IK_POS_TOL = 0.025           # m


def _urdf_velocity_limits(urdf: Path) -> Dict[str, float]:
    """Per-joint velocity limits, straight from the URDF.

    kinematics.py parses lower/upper but not velocity, and rather than widen its
    parser (it is shared with the conversion path that produced the training
    data) we read the one extra attribute here.
    """
    text = urdf.read_text()
    out: Dict[str, float] = {}
    for m in re.finditer(r'<joint name="([^"]+)"[^>]*type="(?:revolute|prismatic)"[^>]*>(.*?)</joint>',
                         text, re.S):
        lim = re.search(r'<limit([^/]*)/>', m.group(2))
        if not lim:
            continue
        vel = re.search(r'velocity="([^"]+)"', lim.group(1))
        if vel:
            out[m.group(1)] = float(vel.group(1))
    return out


def _load_episode(h5: h5py.File, ep: str) -> Dict[str, np.ndarray]:
    g = h5[ep]
    return {
        'joints': g['observation/joint_position'][:],
        'cart_action': g['action/cartesian_position'][:],
        'proprio': g['observation/cartesian_position'][:],
        'images': {k: g[f'observation/image_{k}'] for k in C.CAMERA_KEYS},
    }


def _traj_doc(roll: dict, ckpt: str, ep: str, prompt_ep: str, fps: float) -> dict:
    """Per-arm xyz + gripper for predicted, ground-truth and observed poses."""
    def blocks(a):
        out = {}
        for side, sl in (('left', C.SLICE_CART_L), ('right', C.SLICE_CART_R)):
            p = a[:, sl][:, C.SLICE_BLOCK_POS]
            out[side] = {'x': p[:, 0].round(5).tolist(),
                         'y': p[:, 1].round(5).tolist(),
                         'z': p[:, 2].round(5).tolist(),
                         'grip': a[:, sl][:, C.BLOCK_GRIPPER].round(4).tolist()}
        return out

    dev = {}
    for side, sl in (('left', C.SLICE_CART_L), ('right', C.SLICE_CART_R)):
        e = np.linalg.norm(roll['cart'][:, sl][:, C.SLICE_BLOCK_POS]
                           - roll['gt'][:, sl][:, C.SLICE_BLOCK_POS], axis=-1)
        z = np.linalg.norm(roll['proprio'][:, sl][:, C.SLICE_BLOCK_POS]
                           - roll['gt'][:, sl][:, C.SLICE_BLOCK_POS], axis=-1)
        dev[side] = {'policy': e.round(5).tolist(),
                     'zero_delta': z.round(5).tolist(),
                     'policy_mae': float(e.mean()),
                     'zero_delta_mae': float(z.mean())}

    return {'checkpoint': ckpt, 'episode': ep, 'prompt_episode': prompt_ep,
            'steps': int(roll['steps']), 'fps': fps,
            'pred': blocks(roll['cart']), 'gt': blocks(roll['gt']),
            'proprio': blocks(roll['proprio']), 'deviation': dev}


def _rollout(policy, prompt: dict, target: dict, max_steps: Optional[int]) -> dict:
    """Feed the prompt, then step the whole eval episode, keeping everything."""
    T_p = len(prompt['joints'])
    policy.prompt(
        images=[{k: prompt['images'][k][t] for k in C.CAMERA_KEYS} for t in range(T_p)],
        joint_states=prompt['joints'],
        actions_cartesian=prompt['cart_action'],
    )

    T = len(target['joints'])
    if max_steps:
        T = min(T, max_steps)

    cart, joints, infos = [], [], []
    for t in range(T):
        obs = {k: target['images'][k][t] for k in C.CAMERA_KEYS}
        jt, ca, info = policy.step(obs, target['joints'][t])
        cart.append(ca)
        joints.append(jt)
        infos.append(info)
    return {'cart': np.asarray(cart), 'joints': np.asarray(joints),
            'infos': infos, 'steps': T,
            'gt': np.asarray(target['cart_action'][:T]),
            # Observed state. With delta actions, commanding zero delta yields
            # exactly this, so it is the null baseline the policy must beat --
            # plotting it next to pred and gt makes "did the model help?"
            # answerable by eye.
            'proprio': np.asarray(target['proprio'][:T])}


def _check(roll: dict, kin, vel_limits: Dict[str, float], fps: float,
           thresholds: dict) -> dict:
    """Every check returns (worst_value, limit, n_violations)."""
    cart, joints, infos = roll['cart'], roll['joints'], roll['infos']
    dt = 1.0 / fps
    checks: Dict[str, dict] = {}

    def record(name, worst, limit, n, unit, higher_is_worse=True):
        bad = (worst > limit) if higher_is_worse else (worst < limit)
        checks[name] = {'worst': float(worst), 'limit': float(limit),
                        'violations': int(n), 'unit': unit, 'pass': not bad}

    # --- IK convergence ------------------------------------------------------
    # 'converged' is the solver's own opinion; pos_err is the ground truth of
    # whether it actually got there. Both are checked -- a solve can report
    # success and still sit a millimetre out.
    # The solver's own `converged` flag is NOT used as the gate -- see IK_POS_TOL.
    # It is recorded because a sudden change in it between checkpoints is still
    # worth looking at, but it cannot separate a good policy from a bad one.
    n_nonconv = sum(1 for i in infos for s in ('left', 'right')
                    if not i[s].get('converged', False))
    worst_pos_err = max((i[s]['pos_err'] for i in infos for s in ('left', 'right')),
                        default=0.0)
    n_unreach = sum(1 for i in infos for s in ('left', 'right')
                    if i[s]['pos_err'] > thresholds['ik_pos_tol'])
    record('ik_reachable', worst_pos_err, thresholds['ik_pos_tol'], n_unreach, 'm')
    checks['ik_reachable']['nonconverged_steps'] = int(n_nonconv)
    checks['ik_reachable']['note'] = (
        'gated on residual, not on the solver converged flag: 1 mm of error on a '
        'RECORDED pose already drops that flag 80/80 -> 60/80')

    # --- joint position limits ----------------------------------------------
    # Only an ACTUAL breach counts. An earlier version failed anything within
    # 0.02 rad of a bound, which flagged the robot standing still: arm_l_joint2
    # has URDF range [0.0, 3.14] and arm_r_joint2 [-3.14, 0.0], so joint2 rests
    # exactly ON its bound. The recorded demonstrations tripped it 67 times in
    # 40 steps -- the check was measuring the robot's home pose, not the policy.
    # Saturation is still counted, as information rather than as a failure.
    worst_margin, n_breach, n_sat = np.inf, 0, 0
    for side, sl in (('left', C.SLICE_ARM_L), ('right', C.SLICE_ARM_R)):
        lim = kin.limits[side]                      # (7, 2) lower/upper
        q = joints[:, sl]
        head = np.minimum(q - lim[:, 0], lim[:, 1] - q)
        n_breach += int((head < -thresholds['joint_limit_eps']).sum())
        n_sat += int((head < thresholds['joint_limit_margin']).sum())
        worst_margin = min(worst_margin, float(head.min()))
    record('joint_limits', -worst_margin, thresholds['joint_limit_eps'],
           n_breach, 'rad past bound')
    checks['joint_limits']['worst_headroom_rad'] = float(worst_margin)
    checks['joint_limits']['saturated_steps'] = int(n_sat)

    # --- joint velocity ------------------------------------------------------
    worst_vel, n_vel = 0.0, 0
    for side, sl, names in (('left', C.SLICE_ARM_L, C.ARM_L_JOINTS),
                            ('right', C.SLICE_ARM_R, C.ARM_R_JOINTS)):
        dq = np.abs(np.diff(joints[:, sl], axis=0)) / dt
        caps = np.array([vel_limits.get(n, 4.8) for n in names]) * thresholds['joint_vel_margin']
        n_vel += int((dq > caps).sum())
        worst_vel = max(worst_vel, float((dq / caps).max()) if dq.size else 0.0)
    record('joint_velocity', worst_vel, 1.0, n_vel, 'x limit')

    # --- end-effector speed --------------------------------------------------
    worst_speed, n_speed = 0.0, 0
    for sl in (C.SLICE_CART_L, C.SLICE_CART_R):
        p = cart[:, sl][:, C.SLICE_BLOCK_POS]
        sp = np.linalg.norm(np.diff(p, axis=0), axis=-1) / dt
        n_speed += int((sp > thresholds['eef_speed_max']).sum())
        worst_speed = max(worst_speed, float(sp.max()) if sp.size else 0.0)
    record('eef_speed', worst_speed, thresholds['eef_speed_max'], n_speed, 'm/s')

    # --- gripper chatter -----------------------------------------------------
    worst_chatter = 0.0
    for sl in (C.SLICE_CART_L, C.SLICE_CART_R):
        g = cart[:, sl][:, C.BLOCK_GRIPPER] > 0.5
        flips = int(np.abs(np.diff(g.astype(int))).sum())
        worst_chatter = max(worst_chatter, flips / (len(g) * dt))
    record('gripper_chatter', worst_chatter, thresholds['gripper_chatter_max'],
           int(worst_chatter > thresholds['gripper_chatter_max']), 'flips/s')

    # --- lift command --------------------------------------------------------
    # The action's lift channel sits after the two arm blocks.
    lift_cmd = cart[:, C.SLICE_ACTION_EXTRA]
    worst_lift = float(np.abs(lift_cmd).max()) if lift_cmd.size else 0.0
    record('lift_command', worst_lift, thresholds['lift_command_max'],
           int((np.abs(lift_cmd) > thresholds['lift_command_max']).sum()), 'm')

    # --- bimanual proximity --------------------------------------------------
    pl = cart[:, C.SLICE_CART_L][:, C.SLICE_BLOCK_POS]
    pr = cart[:, C.SLICE_CART_R][:, C.SLICE_BLOCK_POS]
    d = np.linalg.norm(pl - pr, axis=-1)
    record('arm_proximity', float(d.min()), thresholds['arm_proximity_min'],
           int((d < thresholds['arm_proximity_min']).sum()), 'm',
           higher_is_worse=False)

    # --- accuracy, for context (not a gate) ----------------------------------
    acc = {}
    for side, sl in (('left', C.SLICE_CART_L), ('right', C.SLICE_CART_R)):
        e = np.linalg.norm(cart[:, sl][:, C.SLICE_BLOCK_POS]
                           - roll['gt'][:, sl][:, C.SLICE_BLOCK_POS], axis=-1)
        acc[side] = {'pos_mae_m': float(e.mean()), 'pos_max_m': float(e.max())}

    return {'checks': checks, 'accuracy': acc,
            'passed': all(c['pass'] for c in checks.values())}


def validate(dataset: Path, checkpoint: Path, train_yaml: Path, out_dir: Path,
             urdf: Path, hdf5_name: str = 'ffw_sg2.hdf5', fps: float = 15.0,
             episodes: Optional[List[str]] = None, max_steps: Optional[int] = None,
             device: str = 'cuda', thresholds: Optional[dict] = None) -> dict:
    from aiworker_icrt.policy import DualArmICRT

    thresholds = thresholds or {}
    thresholds = {
        'eef_speed_max': thresholds.get('eef_speed_max', EEF_SPEED_MAX),
        'joint_vel_margin': thresholds.get('joint_vel_margin', JOINT_VEL_MARGIN),
        'joint_limit_margin': thresholds.get('joint_limit_margin', JOINT_LIMIT_MARGIN),
        'joint_limit_eps': thresholds.get('joint_limit_eps', JOINT_LIMIT_EPS),
        'gripper_chatter_max': thresholds.get('gripper_chatter_max', GRIPPER_CHATTER_MAX),
        'lift_command_max': thresholds.get('lift_command_max', LIFT_COMMAND_MAX),
        'arm_proximity_min': thresholds.get('arm_proximity_min', ARM_PROXIMITY_MIN),
        'ik_pos_tol': thresholds.get('ik_pos_tol', IK_POS_TOL),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    vel_limits = _urdf_velocity_limits(urdf)

    # Default to the run's own held-out split -- validating on training episodes
    # would measure memorisation, not safety under novel input.
    if episodes is None:
        split = train_yaml.parent / 'val_split.json'
        if not split.is_file():
            raise SystemExit(f'no episode list given and {split} not found; pass --episodes')
        episodes = json.loads(split.read_text())

    policy = DualArmICRT(train_yaml_path=train_yaml, checkpoint_path=checkpoint,
                         device=device)

    results, failures = {}, []
    with h5py.File(dataset / hdf5_name, 'r') as h5:
        for ep in episodes:
            if ep not in h5:
                print(f'  SKIP {ep}: not in {hdf5_name}')
                continue
            # Prompt with a DIFFERENT episode of the same task. Prompting with
            # the episode under test would leak the answer.
            prefix = ep.rsplit('_episode_', 1)[0]
            siblings = [k for k in h5 if k.startswith(prefix + '_episode_') and k != ep]
            if not siblings:
                print(f'  SKIP {ep}: no sibling episode to prompt with')
                continue
            prompt_ep = siblings[0]

            roll = _rollout(policy, _load_episode(h5, prompt_ep),
                            _load_episode(h5, ep), max_steps)
            policy.reset()
            r = _check(roll, policy.kin, vel_limits, fps, thresholds)
            r['prompt_episode'] = prompt_ep
            r['steps'] = roll['steps']
            results[ep] = r

            # Trajectories for the UI. Written on EVERY run, not behind a flag:
            # a verdict without the curve behind it is not reviewable, and the
            # rollout is far too expensive to repeat just to look at it.
            traj_dir = out_dir / 'trajectories'
            traj_dir.mkdir(exist_ok=True)
            (traj_dir / f'{checkpoint.stem}__{ep}.json').write_text(
                json.dumps(_traj_doc(roll, checkpoint.stem, ep, prompt_ep, fps)))

            bad = [n for n, c in r['checks'].items() if not c['pass']]
            flag = 'PASS' if r['passed'] else 'FAIL -> ' + ', '.join(bad)
            print(f'  {ep:28} {roll["steps"]:5d} steps  {flag}')
            if not r['passed']:
                failures.append(ep)

    summary = {
        'checkpoint': str(checkpoint),
        'episodes_tested': len(results),
        'episodes_failed': len(failures),
        'failed': failures,
        'thresholds': thresholds,
        'verdict': 'PASS' if results and not failures else 'FAIL',
        'per_episode': results,
    }
    (out_dir / f'{checkpoint.stem}_safety.json').write_text(json.dumps(summary, indent=2))

    print(f'\n  VERDICT: {summary["verdict"]}  '
          f'({len(results) - len(failures)}/{len(results)} episodes clean)')
    if failures:
        print('  Do NOT run this checkpoint on hardware.')
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--train-yaml', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--urdf', type=Path, default=Path('assets/ffw_sg2/ffw_sg2_follower.urdf'))
    ap.add_argument('--hdf5-name', default='ffw_sg2.hdf5')
    ap.add_argument('--fps', type=float, default=15.0,
                    help='dataset rate; sets the dt that speeds are computed against')
    ap.add_argument('--episodes', nargs='*', default=None,
                    help='default: the run\'s own val_split.json')
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--eef-speed-max', type=float, default=EEF_SPEED_MAX)
    ap.add_argument('--lift-command-max', type=float, default=LIFT_COMMAND_MAX)
    a = ap.parse_args()

    validate(a.dataset, a.checkpoint, a.train_yaml, a.out_dir, a.urdf,
             a.hdf5_name, a.fps, a.episodes, a.max_steps, a.device,
             {'eef_speed_max': a.eef_speed_max, 'lift_command_max': a.lift_command_max})


if __name__ == '__main__':
    main()
