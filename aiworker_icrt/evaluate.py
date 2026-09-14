#!/usr/bin/env python3
"""Offline evaluation: replay a held-out episode through a trained ICRT policy.

    python -m aiworker_icrt.evaluate \
        --dataset data/icrt_ecu --checkpoint runs/smoke/checkpoint-0.pth \
        --train-yaml runs/smoke/args.yaml --prompt episode_000000 --eval episode_000005

This is the gate that matters before hardware (PLAN.md Phase 6). It answers the
question the whole project rests on: does prompting with a demonstration of the
*same* task beat prompting with a different one?

Running it with `--prompt` from another task gives the contrast. If same-task
prompting does not win, the in-context claim does not hold and no amount of
deployment work fixes that.

Nothing here drives a robot. The policy is fed recorded observations and its
predicted actions are compared against what the teleoperator actually did.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from aiworker_icrt import constants as C


def _load_episode(h5: h5py.File, ep: str) -> Dict[str, np.ndarray]:
    g = h5[ep]
    return {
        'images': {k: g[f'observation/image_{k}'][:] for k in C.CAMERA_KEYS},
        'joints': g['observation/joint_position'][:],
        'cart_state': g['observation/cartesian_position'][:],
        'cart_action': g['action/cartesian_position'][:],
    }


def evaluate(
    dataset: Path,
    checkpoint: Path,
    train_yaml: Path,
    prompt_ep: str,
    eval_ep: str,
    out_dir: Path,
    hdf5_name: str = 'ffw_sg2.hdf5',
    max_steps: Optional[int] = None,
    device: str = 'cuda',
) -> dict:
    from aiworker_icrt.policy import DualArmICRT

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset / hdf5_name, 'r') as h5:
        if prompt_ep not in h5:
            raise KeyError(f'prompt episode {prompt_ep!r} not in {dataset}')
        if eval_ep not in h5:
            raise KeyError(f'eval episode {eval_ep!r} not in {dataset}')
        prompt = _load_episode(h5, prompt_ep)
        target = _load_episode(h5, eval_ep)

    policy = DualArmICRT(train_yaml_path=train_yaml, checkpoint_path=checkpoint,
                         device=device)

    # Fill the KV cache with the demonstration.
    T_p = len(prompt['joints'])
    policy.prompt(
        images=[{k: prompt['images'][k][t] for k in C.CAMERA_KEYS} for t in range(T_p)],
        joint_states=prompt['joints'],
        actions_cartesian=prompt['cart_action'],
    )

    T = len(target['joints'])
    if max_steps:
        T = min(T, max_steps)

    pred, gt = [], []
    for t in range(T):
        obs = {k: target['images'][k][t] for k in C.CAMERA_KEYS}
        _joint_target, cart, _info = policy.step(obs, target['joints'][t])
        pred.append(cart)
        gt.append(target['cart_action'][t])

    pred, gt = np.asarray(pred), np.asarray(gt)

    # Per-arm position error, and gripper agreement as a binary decision.
    res = {'prompt_episode': prompt_ep, 'eval_episode': eval_ep, 'steps': int(T)}
    for side, sl in (('left', C.SLICE_CART_L), ('right', C.SLICE_CART_R)):
        pe = np.linalg.norm(pred[:, sl][:, C.SLICE_BLOCK_POS]
                            - gt[:, sl][:, C.SLICE_BLOCK_POS], axis=-1)
        gp = pred[:, sl][:, C.BLOCK_GRIPPER]
        gg = gt[:, sl][:, C.BLOCK_GRIPPER]
        res[side] = {
            'pos_mae_m': float(pe.mean()),
            'pos_rmse_m': float(np.sqrt((pe ** 2).mean())),
            'pos_max_m': float(pe.max()),
            'gripper_mae': float(np.abs(gp - gg).mean()),
            'gripper_agreement': float(((gp > 0.5) == (gg > 0.5)).mean()),
        }

    np.savez_compressed(out_dir / f'{eval_ep}__prompt_{prompt_ep}.npz',
                        pred=pred, gt=gt)
    (out_dir / f'{eval_ep}__prompt_{prompt_ep}.json').write_text(json.dumps(res, indent=2))
    _plot(pred, gt, res, out_dir / f'{eval_ep}__prompt_{prompt_ep}.png')
    return res


def _plot(pred: np.ndarray, gt: np.ndarray, res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    fig.suptitle(f"predicted vs recorded — eval {res['eval_episode']}, "
                 f"prompted with {res['prompt_episode']}", fontsize=12)
    t = np.arange(len(pred))
    for r, (side, sl) in enumerate((('left', C.SLICE_CART_L),
                                    ('right', C.SLICE_CART_R))):
        p = pred[:, sl][:, C.SLICE_BLOCK_POS]
        g = gt[:, sl][:, C.SLICE_BLOCK_POS]
        for i, lbl in enumerate('xyz'):
            ax = axes[r, i]
            ax.plot(t, g[:, i], lw=1.4, label='recorded')
            ax.plot(t, p[:, i], lw=1.1, ls='--', label='predicted')
            ax.set_title(f'{side} {lbl}', fontsize=9)
            ax.grid(alpha=.3)
            if i == 0:
                ax.legend(fontsize=7)
        ax = axes[r, 3]
        ax.plot(t, gt[:, sl][:, C.BLOCK_GRIPPER], lw=1.4, label='recorded')
        ax.plot(t, pred[:, sl][:, C.BLOCK_GRIPPER], lw=1.1, ls='--', label='predicted')
        ax.set_title(f"{side} gripper  (agree "
                     f"{res[side]['gripper_agreement']*100:.0f}%)", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--hdf5-name', default='ffw_sg2.hdf5')
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--train-yaml', type=Path, required=True)
    ap.add_argument('--prompt', required=True, help='episode used as the demo')
    ap.add_argument('--eval', dest='eval_ep', required=True, help='episode to replay')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    res = evaluate(a.dataset, a.checkpoint, a.train_yaml, a.prompt, a.eval_ep,
                   a.out or (a.dataset / 'eval'), a.hdf5_name, a.max_steps, a.device)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
