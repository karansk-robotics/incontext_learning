#!/usr/bin/env python3
"""Phase 3 gate: load a converted dataset through ICRT's real SequenceDataset.

    python tests/check_dataset_roundtrip.py /data/icrt/ffw_sg2_mt [seq_length]

Exercises all four third_party/icrt/icrt/data/dataset.py patches at once, and is
the cheapest way to catch a bad conversion before spending GPU hours on it.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from icrt.data.dataset import SequenceDataset
from icrt.util.args import DatasetConfig, SharedConfig

from aiworker_icrt import constants as C

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
seq_length = int(sys.argv[2]) if len(sys.argv) > 2 else 32

ds_cfg = DatasetConfig(
    dataset_json=str(out_dir / 'dataset_config.json'),
    vision_aug=True, proprio_noise=0.0, action_noise=0.0,
    rebalance_tasks=False, num_repeat_traj=1,
)
# Dimensions come from the canonical layout, never hardcoded here: this file
# already went stale once when the action space moved from 20-D to 23/21.
sh_cfg = SharedConfig(seq_length=seq_length, num_pred_steps=16, rot_6d=True,
                      num_cameras=3, num_arms=2,
                      proprio_extra_dim=len(C.PROPRIO_EXTRA),
                      action_extra_dim=len(C.ACTION_EXTRA),
                      use_delta_action=True)
vt = transforms.Compose([
    transforms.Resize(248, antialias=True), transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ds = SequenceDataset(dataset_config=ds_cfg, shared_config=sh_cfg,
                     vision_transform=vt, split='train')
print(f'\ndataset len = {len(ds)}   total_seq_length = {ds.total_seq_length()}')

failures = []


def chk(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        failures.append(name)


if len(ds) == 0:
    # The task_barrier window only admits an episode start when seq_length fits
    # inside that task's summed train-split frames.
    print('\nFAIL  dataset is empty.\n'
          f'      seq_length={seq_length} does not fit inside at least one\n'
          '      task\'s total frames in the train split. Lower --shared-cfg.seq-length,\n'
          '      or check that verb_to_episode.json actually groups your episodes.')
    sys.exit(1)

b = ds[0]
for k, v in b.items():
    print(f'  {k:14} {tuple(v.shape)}  {v.dtype}')

obs, prop, act = b['observation'], b['proprio'], b['action']
print('\nchecks:')
chk('observation is (T, 3 cameras, 3, H, W)', obs.ndim == 5 and obs.shape[1] == 3,
    str(tuple(obs.shape)))
chk(f'proprio is (T, pred, {C.PROPRIO_DIM})  [arms {C.CARTESIAN_DIM} + '
    f'{"+".join(C.PROPRIO_EXTRA)}]',
    prop.shape[-1] == C.PROPRIO_DIM, str(tuple(prop.shape)))
chk(f'action is (T, pred, {C.ACTION_DIM + 1}) = {C.ACTION_DIM} + eos  '
    f'[arms {C.CARTESIAN_DIM} + {"+".join(C.ACTION_EXTRA)}]',
    act.shape[-1] == C.ACTION_DIM + 1, str(tuple(act.shape)))
chk('no NaNs', not (torch.isnan(obs).any() or torch.isnan(prop).any()
                    or torch.isnan(act).any()))
chk('images normalised, not raw 0-255', float(obs.abs().max()) < 20,
    f'max={float(obs.abs().max()):.2f}')
chk('prompt_mask is 0/1',
    set(np.unique(b['prompt_mask'].numpy()).tolist()) <= {0.0, 1.0})
gl, gr = C.BLOCK_GRIPPER, C.ARM_CART_DIM + C.BLOCK_GRIPPER
chk('gripper channels within [0, 1]',
    bool(act[..., gl].min() >= -0.05 and act[..., gl].max() <= 1.05
         and act[..., gr].min() >= -0.05 and act[..., gr].max() <= 1.05),
    f'L={act[...,gl].min():.2f}..{act[...,gl].max():.2f} '
    f'R={act[...,gr].min():.2f}..{act[...,gr].max():.2f}')

# The commanded lift rides in the action tail, just before eos.
lift_i = C.CARTESIAN_DIM
chk('lift channel present in the action tail',
    act.shape[-1] > lift_i,
    f'action[...,{lift_i}] = {act[...,lift_i].min():.4f}..{act[...,lift_i].max():.4f}')
# Delta actions live in the EEF frame and should be much smaller than absolute
# coordinates; if they are comparable, skip_rot_conversion or use_delta_action
# is not doing what we think.
d, a = float(act[..., :3].abs().mean()), float(prop[..., :3].abs().mean())
chk('delta actions smaller than absolute coords', d < a, f'delta={d:.4f} abs={a:.4f}')

print('\nRESULT:', 'ALL PASS' if not failures else f'FAILURES: {failures}')
sys.exit(1 if failures else 0)
