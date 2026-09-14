#!/usr/bin/env python3
"""Cut long episodes in two at the grasp, so a demonstration and an attempt fit
in one training window.

    python -m aiworker_icrt.split_episodes \
        --dataset /dev/shm/icrt_multitask \
        --out /dev/shm/icrt_multitask_split \
        --task yellow

WHY, precisely
--------------
ICRT's in-context signal comes from `create_prompt_mask`, which needs a full
demonstration AND a full attempt inside one `seq_length` window. Our ylw2
episodes average 958 frames against a seq_length of 1024, so such a window holds
one episode and a sliver of the next. A boundary still lands inside often enough
that `--no-prompt-loss` computes a loss and nothing looks broken -- but the model
rarely sees demo-then-attempt on that task. ICRT-MT's median episode is 129
frames, so their 512 window holds about four.

This does NOT fix the frame imbalance between tasks; `task_grouping` does that,
and this script rewrites it (see below). The window is the reason to cut.

WHERE THE CUT GOES
------------------
At the grasp -- the first rising edge of the working arm's gripper. It is the
only natural boundary and it sits near the middle (52.7% of the episode on
average). Segment A is a complete *pick*, segment B starts with the part already
held and is a complete *place*. A mid-transport cut would make neither.

The halves get DIFFERENT task labels. Under one label a prompt showing "carry and
drop" could precede a query of "approach and grasp", which is incoherent -- and
more tasks is what the in-context claim wants anyway.

AFTER RUNNING THIS
------------------
1. Regenerate action statistics under a NEW filename -- delta-action means and
   stds change when episode boundaries move, and overwriting corrupts validation
   of any run referencing the old file.
2. Re-run `audit_dataset` on the new directory.
3. Render one `_pick` and one `_place` and WATCH them. A cut one frame late
   starts the place segment with the gripper still open, and nothing numeric
   will tell you.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from aiworker_icrt import constants as C

GRIPPERS = {'left': C.SLICE_CART_L.start + C.BLOCK_GRIPPER,
            'right': C.SLICE_CART_R.start + C.BLOCK_GRIPPER}


def _working_gripper(action: np.ndarray) -> tuple[str, int]:
    """The arm that actually opens and closes. Picking the idle one would find no
    edge at all, or an edge in sensor noise."""
    side = max(GRIPPERS, key=lambda s: np.ptp(action[:, GRIPPERS[s]]))
    return side, GRIPPERS[side]


def _grasp_index(action: np.ndarray) -> int | None:
    """First rising edge of the working gripper, at its own midpoint."""
    _side, idx = _working_gripper(action)
    g = action[:, idx]
    if np.ptp(g) <= 1e-6:
        return None
    closed = (g > (g.min() + g.max()) / 2).astype(np.int8)
    rising = np.flatnonzero(np.diff(closed) == 1)
    return int(rising[0]) + 1 if len(rising) else None


def _copy_group(src: h5py.Group, dst: h5py.Group, sl: slice | None = None) -> None:
    for name, obj in src.items():
        if isinstance(obj, h5py.Group):
            _copy_group(obj, dst.create_group(name), sl)
        else:
            # Preserve compression and chunking. Dropping them inflated an
            # 18.7 GB dataset to 30.8 GB the first time this ran -- the source
            # images are lzf with per-frame chunks.
            chunks = obj.chunks
            data = obj[sl] if sl is not None else obj[:]
            if chunks is not None:
                chunks = tuple(min(c, d) for c, d in zip(chunks, data.shape))
            dst.create_dataset(name, data=data, chunks=chunks,
                               compression=obj.compression,
                               compression_opts=obj.compression_opts,
                               shuffle=obj.shuffle)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--hdf5-name', default='ffw_sg2.hdf5')
    ap.add_argument('--task', required=True,
                    help='substring identifying the task to split, e.g. "yellow"')
    ap.add_argument('--min-segment', type=int, default=30,
                    help='leave an episode whole if either half is shorter')
    ap.add_argument('--suffix', nargs=2, default=('pick', 'place'))
    ap.add_argument('--label', nargs=2,
                    default=('approach and grasp', 'carry and drop'))
    a = ap.parse_args()

    src_dir, out_dir = a.dataset, a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    verb = json.loads((src_dir / 'verb_to_episode.json').read_text())
    epi_len = json.loads((src_dir / 'epi_len_mapping.json').read_text())

    targets = [t for t in verb if a.task.lower() in t.lower()]
    if len(targets) != 1:
        raise SystemExit(f'--task {a.task!r} matched {len(targets)} tasks: {targets}')
    task = targets[0]
    split_eps = set(verb[task])
    print(f'splitting {len(split_eps)} episodes of:\n  {task}\n')

    new_verb: dict[str, list[str]] = {t: list(e) for t, e in verb.items() if t != task}
    new_labels = (f'{task} -- {a.label[0]}', f'{task} -- {a.label[1]}')
    new_verb[new_labels[0]] = []
    new_verb[new_labels[1]] = []
    new_len: dict[str, int] = {}

    n_split = n_whole = 0
    with h5py.File(src_dir / a.hdf5_name, 'r') as src, \
            h5py.File(out_dir / a.hdf5_name, 'w') as dst:
        for ep in src:
            g = src[ep]
            if ep not in split_eps:
                _copy_group(g, dst.create_group(ep))
                new_len[ep] = epi_len[ep]
                continue

            act = g['action/cartesian_position'][:]
            cut = _grasp_index(act)
            T = len(act)
            if cut is None or cut < a.min_segment or (T - cut) < a.min_segment:
                # No usable grasp edge, or a half too short to survive the
                # minimum_length filter. Keep it whole rather than emit a stub.
                _copy_group(g, dst.create_group(ep))
                new_len[ep] = epi_len[ep]
                new_verb[new_labels[0]].append(ep)
                n_whole += 1
                print(f'  {ep}: KEPT WHOLE (cut={cut}, T={T})')
                continue

            for i, (sfx, sl) in enumerate(((a.suffix[0], slice(0, cut)),
                                           (a.suffix[1], slice(cut, T)))):
                name = f'{ep}_{sfx}'
                _copy_group(g, dst.create_group(name), sl)
                new_len[name] = sl.stop - sl.start
                new_verb[new_labels[i]].append(name)
            n_split += 1
            print(f'  {ep}: {T} -> {cut} + {T - cut}')

    keys = sorted(new_len)
    (out_dir / 'hdf5_keys.json').write_text(json.dumps(keys, indent=1))
    (out_dir / 'epi_len_mapping.json').write_text(json.dumps(new_len, indent=1))
    (out_dir / 'verb_to_episode.json').write_text(json.dumps(new_verb, indent=1))

    # task_grouping MUST be rewritten: the task names changed, and
    # upweight_tasks[v] is a bare dict lookup that raises KeyError on any task
    # the grouping omits. Ratios are recomputed to equalise FRAMES per task,
    # because rebalance_tasks equalises episode COUNT and our episodes are not
    # the same length.
    counts = {t: len(e) for t, e in new_verb.items() if e}
    means = {t: float(np.mean([new_len[x] for x in e])) for t, e in new_verb.items() if e}
    rl = int(np.median(list(counts.values())))
    # Equal frames per task. frames_t = rl * ratio_t * mean_len_t, so
    # ratio_t = target / (rl * mean_len_t). Anchoring target at the SMALLEST
    # rl*mean_len keeps every ratio <= 1, so no task is upsampled past the
    # episodes it actually has -- oversampling with replacement is what made the
    # ylw2 imbalance worse in the first place.
    # The two halves of a split episode are still ONE physical task -- the same
    # scene, the same frozen arm. Giving each a full task's budget would put
    # 50% of every epoch back on that task, which is worse than before the
    # split. They share one budget, half each.
    weight = {t: (0.5 if t in new_labels else 1.0) for t in means}
    target = min(rl * m / weight[t] for t, m in means.items())
    ratios, tasks = {}, {}
    for i, (t, m) in enumerate(sorted(means.items())):
        key = f'group{i}'
        ratios[key] = round(weight[t] * target / (rl * m), 4)
        tasks[key] = [t]
    (out_dir / 'task_grouping.json').write_text(
        json.dumps({'ratios': ratios, 'tasks': tasks}, indent=2))

    cfg_src = src_dir / 'dataset_config.json'
    if cfg_src.exists():
        cfg = json.loads(cfg_src.read_text())
        for k in ('dataset_path', 'hdf5_keys'):
            if k in cfg:
                cfg[k] = [str(out_dir / Path(p).name) for p in cfg[k]]
        for k, fn in (('epi_len_mapping_json', 'epi_len_mapping.json'),
                      ('verb_to_episode', 'verb_to_episode.json'),
                      ('task_grouping', 'task_grouping.json')):
            cfg[k] = str(out_dir / fn)
        (out_dir / 'dataset_config.json').write_text(json.dumps(cfg, indent=2))

    print(f'\nsplit {n_split}, kept whole {n_whole}, episodes now {len(keys)}')
    print('\nper-task frames in one epoch (rebalance_length %d, ratios applied):' % rl)
    tot = sum(int(int(rl * ratios[f'group{i}']) * m)
              for i, (_t, m) in enumerate(sorted(means.items())))
    for i, (t, m) in enumerate(sorted(means.items())):
        fr = int(int(rl * ratios[f'group{i}']) * m)
        print('   %-58s %6d  %5.1f%%' % (t[:58], fr, 100 * fr / tot))
    print('\nNEXT: regenerate action stats to a NEW filename, re-audit, and WATCH '
          'one _pick and one _place before training on this.')


if __name__ == '__main__':
    main()
