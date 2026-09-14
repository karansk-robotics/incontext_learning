#!/usr/bin/env python3
"""Audit a LeRobot dataset's UNCOMMANDED joints, and recommend a configuration.

    python -m aiworker_icrt.audit_dataset --lerobot-root data/ecu_raw

Our 20-D action space covers the two arms and their grippers. Head, torso lift
and the mobile base are recorded but never commanded. That is only sound if those
joints actually hold still — and whether they do is a property of the recording
campaign, not of the robot.

Run this BEFORE converting a new dataset. It answers the one question that
decides the model's input and output dimensions:

  head    changes what cam_head sees. It does not touch end-effector geometry
          (the head is a separate branch from the arms), so its only effect is
          that the policy's main camera shows a viewpoint it may not have been
          trained on.

  lift    sits INSIDE the arm chain. With poses referenced from base_link, the
          same arm configuration at a different torso height yields a different
          z. If the lift moves *within* an episode the model observes hand motion
          it did not cause and cannot reproduce — a causality mismatch that no
          amount of training fixes.

  base    moves the whole robot, so every base_link-referenced pose shifts.

The distinction that matters is WITHIN-episode versus BETWEEN-episode variation.
Between-episode variation is closer to augmentation, and is handled by feeding
these joints into proprioception so the policy can condition on them. Within
-episode variation needs the joint to be commanded, or removed from the frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aiworker_icrt import constants as C

# Thresholds. Deliberately tight: the ECU campaign held the head to 0.0031 rad
# (0.18 deg) of servo jitter around a constant setpoint, so anything an order of
# magnitude above that is deliberate motion, not noise.
WITHIN_RAD = np.radians(2.0)     # within-episode motion above this is real
BETWEEN_RAD = np.radians(2.0)    # spread of per-episode start poses
WITHIN_M = 0.01                  # 1 cm, for the prismatic lift
BETWEEN_M = 0.01


def audit(lerobot_root: Path, limit: int | None = None) -> dict:
    from aiworker_icrt.convert_lerobot_icrt import LeRobotReader

    reader = LeRobotReader(lerobot_root)
    episodes = reader.episodes[:limit] if limit else reader.episodes
    print(f'{len(episodes)} episodes in {lerobot_root} '
          f'(codebase_version={reader.version}, fps={reader.fps})\n')

    states, actions, starts, ranges = [], [], [], []
    for ep_meta in episodes:
        cols = reader.read_frames(ep_meta)
        st, ac = cols['observation.state'], cols['action']
        states.append(st)
        actions.append(ac)
        starts.append(st[0])
        ranges.append(st.max(0) - st.min(0))

    S = np.concatenate(states)
    A = np.concatenate(actions)
    starts, ranges = np.array(starts), np.array(ranges)
    width = S.shape[1]

    groups = [('head', [16, 17], C.HEAD_JOINTS, WITHIN_RAD, BETWEEN_RAD, 'rad'),
              ('lift', [18], [C.LIFT_JOINT], WITHIN_M, BETWEEN_M, 'm'),
              ('base', [19, 20, 21], C.BASE_DIMS, 0.05, 0.05, 'm/s')]

    findings = {}
    for name, idxs, labels, w_tol, b_tol, unit in groups:
        if any(i >= width for i in idxs):
            print(f'{name.upper()}: absent — recording is only {width}-D')
            findings[name] = {'present': False}
            continue

        within = ranges[:, idxs].max()
        between = starts[:, idxs].std(axis=0).max()
        n_moving = int((ranges[:, idxs] > w_tol).any(axis=1).sum())

        print(f'{name.upper()}')
        for i, lbl in zip(idxs, labels):
            print(f'  {lbl:16} state {S[:, i].min():+.4f}..{S[:, i].max():+.4f}   '
                  f'commanded {A[:, i].min():+.4f}..{A[:, i].max():+.4f}')
        print(f'  max WITHIN-episode movement : {within:.4f} {unit}'
              f'   ({"MOVES" if within > w_tol else "static"})')
        print(f'  BETWEEN-episode spread (std): {between:.4f} {unit}'
              f'   ({"VARIES" if between > b_tol else "consistent"})')
        print(f'  episodes that moved         : {n_moving}/{len(episodes)}')
        # The commanded setpoint is the honest preflight target; the measured
        # state carries servo jitter around it.
        print(f'  preflight target (commanded): '
              f'{[round(float(np.median(A[:, i])), 4) for i in idxs]}\n')

        findings[name] = {'present': True, 'within': float(within),
                          'between': float(between), 'moves': within > w_tol,
                          'varies': between > b_tol, 'n_moving': n_moving}
    return findings


def recommend(f: dict) -> None:
    print('=' * 72)
    print('RECOMMENDATION\n')
    actions = []

    head = f.get('head', {})
    if head.get('present'):
        if head['moves']:
            actions.append(
                'HEAD MOVES WITHIN EPISODES.\n'
                '  cam_head\'s viewpoint changes during a demonstration, and the\n'
                '  policy has no way to know the gaze angle. Put head_joint1/2 into\n'
                '  PROPRIOCEPTION (proprio 20 -> 22) so it can condition on them.\n'
                '  Commanding the head is a larger change: it is absent from\n'
                '  cyclo_control\'s model and would need its own controller.')
        elif head['varies']:
            actions.append(
                'HEAD IS FIXED PER EPISODE BUT VARIES BETWEEN THEM.\n'
                '  Each demo is internally consistent, so this behaves as viewpoint\n'
                '  augmentation — useful, if the policy can tell the regimes apart.\n'
                '  Add head to PROPRIOCEPTION (proprio 20 -> 22) so it can.\n'
                '  At inference the head must be set to the pose of the episode you\n'
                '  prompt with, not to a global constant.')
        else:
            actions.append(
                'HEAD IS CONSTANT THROUGHOUT. No change needed.\n'
                '  Set it to the commanded target above before every run — the same\n'
                '  viewpoint the policy was trained on.')

    lift = f.get('lift', {})
    if lift.get('present'):
        if lift['moves']:
            actions.append(
                'LIFT MOVES WITHIN EPISODES — this is the one that breaks things.\n'
                '  With poses in base_link the torso is inside the arm chain, so the\n'
                '  policy observes hand motion it did not cause and cannot reproduce.\n'
                '  Two real fixes:\n'
                '    (a) COMMAND the lift: action 20 -> 21. cyclo_control already\n'
                '        carries lift as a decision variable, so this is releasing a\n'
                '        velocity bound rather than adding a mechanism.\n'
                '    (b) Reference end-effector poses from arm_base_link instead, so\n'
                '        the lift cancels out of the geometry entirely. Cleaner, but\n'
                '        gives up absolute hand height and the frame shared with\n'
                '        navigation.\n'
                '  Adding it to proprioception alone is NOT sufficient here: knowing\n'
                '  the torso moved does not let the policy do anything about it.')
        elif lift['varies']:
            actions.append(
                'LIFT IS FIXED PER EPISODE BUT VARIES BETWEEN THEM.\n'
                '  Every episode is a consistent frame, merely offset in z. Add lift\n'
                '  to PROPRIOCEPTION (proprio 20 -> 21) so the policy can account for\n'
                '  the offset, and set the torso to the prompt episode\'s height at\n'
                '  inference.')
        else:
            actions.append('LIFT IS CONSTANT THROUGHOUT. No change needed.')

    base = f.get('base', {})
    if base.get('present') and base.get('moves'):
        actions.append(
            'THE BASE MOVES. Every base_link-referenced pose shifts with it, and\n'
            '  the mobile base is outside the action space entirely. Either restrict\n'
            '  training to stationary episodes, or treat this as mobile manipulation\n'
            '  and extend the action space to the /cmd_vel twist (action 20 -> 23).')

    for i, a in enumerate(actions, 1):
        print(f'{i}. {a}\n')
    if not any(f.get(k, {}).get('moves') or f.get(k, {}).get('varies')
               for k in ('head', 'lift', 'base')):
        print('All uncommanded joints are static. The 20-D arms-only action space\n'
              'is sound for this dataset as configured.')


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lerobot-root', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    recommend(audit(a.lerobot_root, a.limit))


if __name__ == '__main__':
    main()
