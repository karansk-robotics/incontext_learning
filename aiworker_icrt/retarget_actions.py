#!/usr/bin/env python3
"""Rewrite a converted dataset's action stream from the LEADER's command to the
follower's ACHIEVED pose one step ahead.

    python -m aiworker_icrt.retarget_actions \
        --src /dev/shm/icrt_multitask_split \
        --out /dev/shm/icrt_mt_achieved

WHY
---
`action` is what the leader arm was doing; `observation.state` is what the
follower achieved. They differ by the servo lag -- and that lag is a property of
how the controller was tuned that day, not of the task. Measured over our three
recording sessions:

    box    |action - state|   0.006736 rad    21.1 mm at the hand
    ecu                       0.009399 rad    38.4 mm
    ylw2                      0.000223 rad     0.75 mm     <- 30-40x less

ICRT's regression target is `action[t+j] - proprio[t]`. With the leader source
that target is 21-38 mm of servo lag on two tasks and 0.75 mm on the third, so
"action" means a different thing depending on which session recorded it. The
model cannot tell the regimes apart: it emits a roughly constant ~22 mm delta
everywhere, which is about right for box and 55-65x too large for ylw2. That is
why the policy scored WORSE than predicting no motion at all on ylw2.

Taking the target from the achieved pose makes the label real motion, identical
in meaning across sessions, and independent of controller tuning. It is also
what a policy should emit -- the next pose, leaving the servo to close the gap.

EXACTNESS
---------
This rewrites an existing hdf5 rather than re-converting from video, which is
exact here and not an approximation: `joints_to_action` and `joints_to_proprio`
both call `joints_to_cartesian` and append the same lift channel
(`ACTION_EXTRA == PROPRIO_EXTRA == [lift_joint]`), so the Cartesian action of a
joint vector IS its Cartesian proprio. Shifting the stored proprio forward one
frame gives byte-identical output to re-running the converter with
`--action-source achieved`.

The last frame is repeated, so the final step's target is "stay put" -- correct
at the end of an episode.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

ACT = 'action/cartesian_position'
OBS = 'observation/cartesian_position'
ACT_J = 'action/joint_position'
OBS_J = 'observation/joint_position'


def _shift(a: np.ndarray) -> np.ndarray:
    """x[t] <- x[t+1], last frame repeated."""
    return np.concatenate([a[1:], a[-1:]], axis=0)


def _copy(src: h5py.Group, dst: h5py.Group, replace: dict) -> None:
    for name, obj in src.items():
        if isinstance(obj, h5py.Group):
            _copy(obj, dst.create_group(name), replace)
            continue
        data = replace.get(obj.name.split('/', 2)[-1])
        if data is None:
            data = obj[:]
        chunks = obj.chunks
        if chunks is not None:
            chunks = tuple(min(c, d) for c, d in zip(chunks, data.shape))
        dst.create_dataset(name, data=data, chunks=chunks,
                           compression=obj.compression,
                           compression_opts=obj.compression_opts,
                           shuffle=obj.shuffle)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--hdf5-name', default='ffw_sg2.hdf5')
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    lag_before, lag_after, n = [], [], 0

    with h5py.File(a.src / a.hdf5_name, 'r') as src, \
            h5py.File(a.out / a.hdf5_name, 'w') as dst:
        for ep in src:
            g = src[ep]
            obs = g[OBS][:]
            new_act = _shift(obs)
            rep = {ACT: new_act}
            if ACT_J in g and OBS_J in g:
                rep[ACT_J] = _shift(g[OBS_J][:])

            # working-arm lag, before and after, as a sanity readout
            arm = slice(10, 13) if ep.startswith('ylw') else slice(0, 3)
            lag_before.append(np.linalg.norm(g[ACT][:][:, arm] - obs[:, arm], axis=1).mean())
            lag_after.append(np.linalg.norm(new_act[:, arm] - obs[:, arm], axis=1).mean())

            _copy(g, dst.create_group(ep), rep)
            n += 1
            if n % 25 == 0:
                print(f'  {n} episodes')

    for f in ('hdf5_keys.json', 'epi_len_mapping.json', 'verb_to_episode.json',
              'task_grouping.json'):
        if (a.src / f).is_file():
            shutil.copyfile(a.src / f, a.out / f)

    cfg_p = a.src / 'dataset_config.json'
    if cfg_p.is_file():
        cfg = json.loads(cfg_p.read_text())
        for k in ('dataset_path', 'hdf5_keys'):
            if k in cfg:
                cfg[k] = [str(a.out / Path(p).name) for p in cfg[k]]
        for k, fn in (('epi_len_mapping_json', 'epi_len_mapping.json'),
                      ('verb_to_episode', 'verb_to_episode.json'),
                      ('task_grouping', 'task_grouping.json')):
            if k in cfg:
                cfg[k] = str(a.out / fn)
        (a.out / 'dataset_config.json').write_text(json.dumps(cfg, indent=2))

    print(f'\n{n} episodes -> {a.out}')
    print(f'  working-arm |action - observation|, mean over episodes:')
    print(f'    before (leader command)  {1000*np.mean(lag_before):7.2f} mm  '
          f'[{1000*min(lag_before):.2f} .. {1000*max(lag_before):.2f}]')
    print(f'    after  (achieved pose)   {1000*np.mean(lag_after):7.2f} mm  '
          f'[{1000*min(lag_after):.2f} .. {1000*max(lag_after):.2f}]')
    print('  AFTER is now one step of real motion, consistent across sessions.')
    print('\nNEXT: regenerate action statistics against this directory under a NEW '
          'filename, then retrain. Numbers are NOT comparable to leader-target runs.')


if __name__ == '__main__':
    main()
