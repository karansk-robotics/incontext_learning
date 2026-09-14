#!/usr/bin/env python3
"""Compute per-channel action statistics for ICRT's --shared-cfg.scale-action.

    python make_stats.py <dataset_dir> <out.json>

WHY THIS IS NEEDED
------------------
ICRT regresses the delta action with an unweighted MSE, so each channel's share
of the target variance is its share of the gradient. On this dataset the gripper
channel is ~300x larger in scale than the translation channels, which means
translation -- the thing end-effector position error actually measures -- was
receiving a fraction of a percent of the training signal. The model was not
failing to learn position so much as never being asked to.

Standardising each channel (x - mean) / std equalises that. ICRT already has
both halves wired: scale_action at icrt.py:632 on the target, unscale_action at
icrt.py:764 at inference. Only the stats file was missing.

NEAR-CONSTANT CHANNELS
----------------------
A channel that barely moves (the lift: std 0.58 mm, 0/120 episodes above 1 mm)
must not be divided by its own std -- that amplifies pure sensor noise ~1700x
into a full-scale training target.

But the obvious alternative, std = 1.0, is WORSE, and we measured it being worse.
Every other channel gets scaled UP to spread 1 while the lift stays at its raw
0.0006, so it ends up with essentially zero gradient and nothing holds it at
zero any more. Measured on checkpoint-6 of the run that used std = 1.0: commanded
lift drifted to 7.6-15.4 mm on all six held-out episodes, past the 5 mm safety
limit, where the unnormalised run had held 2.2-4.2 mm.

So such channels get a PHYSICALLY MEANINGFUL scale instead: the tolerance beyond
which we consider the channel wrong. For the lift that is 5 mm, the safety limit
in validate_checkpoint.py. The statement this encodes is exactly the one we want:
"5 mm of unwanted torso motion is as serious as a typical error anywhere else."
No noise amplification -- we divide by 5 mm, not by 0.58 mm -- and real pressure
to stay put.

The chunking below mirrors SequenceDataset exactly: convert_multi_step first,
then convert_delta_action, which anchors all num_pred_steps to the proprio at
step 0 of the chunk. Computing stats on step 0 alone understates translation
variance by up to num_pred_steps, because step 15 is 16 frames of travel from
the same anchor.
"""
import json, sys
from pathlib import Path

import h5py
import numpy as np
import torch

from icrt.data.utils import convert_multi_step, convert_delta_action
from aiworker_icrt import constants as C

NUM_PRED_STEPS = 16
# Channels whose measured std falls below STD_FLOOR use these scales instead.
# Keyed by channel index; see constants.SLICE_ACTION_EXTRA (the lift is 20).
SAFETY_SCALE = {20: 0.005}   # lift: 5 mm, the validate_checkpoint.py limit
FALLBACK_SCALE = 1.0         # a near-constant channel with no stated tolerance

STD_FLOOR = 1e-3          # 1 mm / 1 mrad. The lift measures std 0.00058 (0.58 mm)
                          # and is constant by construction (0/120 episodes move >1 mm);
                          # a 1e-4 floor let it through and would have amplified 0.58 mm of
                          # sensor noise into a full-scale target worth 4.76% of the loss.
                          # Smallest live translation channel is R dz at 0.0121, so this
                          # floor separates them with an order of magnitude to spare.


def main() -> None:
    ds_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    split = json.loads((ds_dir.parent / 'train_split.json').read_text()) \
        if (ds_dir.parent / 'train_split.json').is_file() else None

    h5path = ds_dir / 'ffw_sg2.hdf5'
    chunks = []
    with h5py.File(h5path, 'r') as h5:
        eps = split if split else sorted(h5.keys())
        eps = [e for e in eps if e in h5]
        for ep in eps:
            g = h5[ep]
            a = torch.tensor(g['action/cartesian_position'][:], dtype=torch.float64)
            p = torch.tensor(g['observation/cartesian_position'][:], dtype=torch.float64)
            am = convert_multi_step(a, NUM_PRED_STEPS).numpy()
            pm = convert_multi_step(p, NUM_PRED_STEPS).numpy()
            d = convert_delta_action(am, pm, num_arms=2)
            chunks.append(np.asarray(d).reshape(-1, a.shape[1]))
    A = np.concatenate(chunks)
    dim = A.shape[1]

    mean, std = A.mean(0), A.std(0)
    dead = std < STD_FLOOR
    std_out = std.copy()
    for i in np.flatnonzero(dead):
        std_out[i] = SAFETY_SCALE.get(int(i), FALLBACK_SCALE)

    names = (['L dx', 'L dy', 'L dz'] + [f'L r{i}' for i in range(6)] + ['L grip']
             + ['R dx', 'R dy', 'R dz'] + [f'R r{i}' for i in range(6)] + ['R grip']
             + [f'extra{i}' for i in range(dim - 20)])
    print(f'{A.shape[0]:,} chunked timesteps from {len(eps)} episodes, {dim} channels\n')
    print(f'{"channel":8} {"mean":>10} {"std":>10} {"MSE share":>10}  {"after":>8}')
    var = std ** 2
    tot = var.sum()
    for i in range(dim):
        flag = (f'  NEAR-CONSTANT -> scale {std_out[i]:g}' if dead[i] else '')
        print(f'{names[i]:8} {mean[i]:10.5f} {std[i]:10.5f} {100*var[i]/tot:9.2f}% '
              f'{100.0/dim:7.2f}%{flag}')
    tr = [0, 1, 2, 10, 11, 12]
    print(f'\n  BEFORE  translation {100*var[tr].sum()/tot:5.2f}%   '
          f'gripper {100*(var[9]+var[19])/tot:5.2f}%')
    print(f'  AFTER   every channel {100.0/dim:.2f}%  '
          f'({int(dead.sum())} near-constant channel(s) given a fixed scale)')

    out.write_text(json.dumps({
        'mean': mean.tolist(), 'std': std_out.tolist(),
        'min': A.min(0).tolist(), 'max': A.max(0).tolist(),
    }, indent=1))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
