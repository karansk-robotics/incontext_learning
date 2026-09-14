# Fixing the `ylw2` imbalance — can we cut the long episodes?

> **STATUS: the `task_grouping` fix in §6 is applied and verified on `zeux`.**
> Measured by counting `SequenceDataset.steps` on the train split:
> `ylw2` **58.9% → 28.3%**, `ecu` 21.1% → 36.7%, `box` 20.1% → 35.0%.
> Config versioned at `config/icrt_multitask/task_grouping.json`.
> The cut in §7 is **not** done — it addresses the window, not the balance.

**Short answer: yes, and it is safe if you cut at the grasp — but cutting fixes a
different problem than the one you asked it to fix.**

Cutting barely moves the frame imbalance (58.9% → 41.7%). What actually fixes the
imbalance is `task_grouping`, which needs no data change at all (58.9% → 27.9%).
What cutting *does* fix is the in-context window, which nothing else can.

Every number below was measured on the 114 training episodes of `runs/mt21_gpu`
against `/dev/shm/icrt_multitask` on `zeux`. Companion: `training_tips.md` §C1.

---

## 1. What is actually inside a `ylw2` episode

The first thing to rule out was that these are several picks recorded back to
back, which would make cutting trivial. They are not.

```
grasp events per ylw2 episode:  min 1, max 1, median 1   (all 31 episodes)
```

One grasp each. A `ylw2` episode is a single pick-and-place.

Nor is it idle padding. Measuring frames where the working hand moves less than
0.5 mm:

```
task   eps  frames   idle    moving frames/ep   grasp at
ylw2    31   29,649  17.1%          793.3        52.7% of episode
ecu     39   13,357   6.0%          321.9        39.0%
box     50   16,305   6.1%          306.3        44.7%
```

Trimming idle would take `ylw2` from 956 to ~793 frames — real, but it does not
explain the gap. **`ylw2` genuinely contains ~2.5× more motion than `box` or
`ecu`**: 5.75 m of travel at ~6 mm/frame, against ~3.2 m at ~10 mm/frame. It is
a longer, slower task: reach into a bin, pick, carry, drop.

So there is no free lunch here. Any cut divides one continuous demonstration.

---

## 2. Where to cut, if we cut

The grasp is the one natural boundary, and it sits almost exactly in the middle:

```
ylw2 cut at the grasp:
  segment A (approach + grasp)   mean 501 frames   min 434
  segment B (carry + drop)       mean 457 frames   min  72
```

Segment A is a complete *pick*. Segment B starts with the part already in the
gripper and is a complete *place*. Both are coherent demonstrations on their own,
which a mid-transport cut would not be.

One outlier: a single episode has only **72 frames** after its grasp. Above the
30-frame minimum, so it survives the filter, but it is worth looking at — it is
probably a mis-grasp and re-grasp.

---

## 3. What each option actually does

`rebalance_length` is the **median number of episodes per task**
(`dataset.py:186`), so ICRT equalises episode *count*, never frames. Every option
below is an attempt to work around that.

**Baseline today** — `rebalance_tasks: true`, rebalance_length 37:

```
ylw2    35,456 frames   58.9%    29 eps, sampled WITH replacement
ecu     12,678 frames   21.1%    37 eps
box     12,078 frames   20.1%    48 eps
```

### (a) Cut at the grasp, keep one task label

58 shorter episodes under the same task, so rebalance_length rises to 48:

```
ylw2    22,998 frames   41.7%    58 eps
ecu     16,447 frames   29.8%    37 eps, now sampled WITH replacement
box     15,669 frames   28.4%    48 eps
```

**58.9% → 41.7%.** A real improvement, but not balance, and it pushes `ecu` into
oversampling instead.

### (b) Cut at the grasp, give the halves two task labels

```
ylw2_pick   16,529 frames   30.8%
ylw2_place  15,093 frames   28.1%     <- 58.9% of frames, combined
ecu         11,307 frames   21.1%
box         10,772 frames   20.1%
```

**No improvement at all on balance** — 30.8 + 28.1 = 58.9%, identical to
baseline, because `ylw2` is now two of four tasks and each gets an equal share.
It does raise the task count 3 → 4, which matters for a different reason (§5).

### (c) `task_grouping` ratios — no data change

The ratio multiplies that task's rebalance length (`dataset.py:391`):

```
ratio 0.28  ->  rl_ylw2=10   ylw2 share  27.9%     <- balanced
ratio 0.30  ->  rl_ylw2=11   ylw2 share  29.9%
ratio 0.35  ->  rl_ylw2=12   ylw2 share  31.7%
ratio 0.50  ->  rl_ylw2=18   ylw2 share  41.1%
```

**58.9% → 27.9% by editing one JSON file.** No re-conversion, no new HDF5, no
regenerated statistics. This is the fix for the imbalance.

---

## 4. So why cut at all?

Because of the window, not the balance.

ICRT's in-context mechanism needs **a full demonstration and a full attempt in the
same `seq_length` window**. That is how `create_prompt_mask` works: it picks an
`eos` inside the window and calls everything before it the prompt.

```
ICRT-MT   median episode 129 frames,  seq_length  512  ->  ~4 episodes/window
ours      ylw2    mean 958 frames,    seq_length 1024  ->  ~1 episode/window
          box/ecu mean ~330 frames,   seq_length 1024  ->  ~3 episodes/window
```

For `box` and `ecu` the window works. For `ylw2` it does not: at 958 frames a
1024-frame window holds one episode and a sliver of the next. A boundary still
lands inside often enough that `--no-prompt-loss` computes a loss and nothing
looks broken — but the model rarely sees demo-then-attempt on that task.

**Cutting to ~480-frame segments puts two `ylw2` segments in a 1024 window.** No
flag does this. It is the only reason to touch the data.

(Raising `seq_length` to 2048 would also do it, at roughly 4× the activation
memory and a slower step — worth pricing before committing to a re-conversion.)

---

## 5. Recommendation

Do (c) now, and (a)+(b) only if you are re-converting anyway.

| | fixes imbalance | fixes window | cost |
|---|---|---|---|
| **(c) `task_grouping`** | **yes, 27.9%** | no | one JSON file |
| (a) cut, one label | partly, 41.7% | **yes** | new HDF5 + new stats |
| (b) cut, two labels | no | **yes** | same, plus 3→4 tasks |
| trim idle frames | marginal | marginal | re-convert |
| raise `seq_length` 2048 | no | **yes** | ~4× activation memory |

If you cut, prefer **two labels (b)**: a prompt of "carry and drop" followed by a
query of "approach and grasp" is incoherent under one label, and 4 tasks is
better than 3 for the in-context claim that is the point of the paper. Then apply
(c) on top to bring the combined `ylw2` share down — (b) alone leaves it at 58.9%.

---

## 6. How to do (c) — the cheap fix

Write `/dev/shm/icrt_multitask/task_grouping.json`. **Every task must be listed**
or `upweight_tasks[v]` raises `KeyError` at `dataset.py:391`:

```json
{
  "ratios": { "pick_bin": 0.28, "pick_place_left": 1.0 },
  "tasks": {
    "pick_bin": ["locate the yellow part and pick the part from the bin"],
    "pick_place_left": [
      "locate the ecu part and using the left arm pick the part from the current card board boxes and drop it on other card board box located near to it.",
      "locate the box with blue color hankerchief and use the left arm to pick the box and drop it"
    ]
  }
}
```

Add the path to `dataset_config.json` (read via `dataset_json.get("task_grouping")`):

```json
"task_grouping": "/dev/shm/icrt_multitask/task_grouping.json"
```

Then launch as usual. Confirm it took effect — the loader prints it:

```
Each task is rebalanced to have length:  37
overriding with known ratio:  {'pick_bin': 0.28, 'pick_place_left': 1.0}
```

**Applied 2026-09-14.** `dataset_config.json` was backed up to
`dataset_config.json.bak` first. Verified by instantiating `SequenceDataset`
against `runs/mt21_gpu/run.yaml` and counting frames per task in `ds.steps`:

```
          before    after     (predicted 27.9%)
ylw2      58.9%     28.3%
ecu       21.1%     36.7%
box       20.1%     35.0%
total              34,525 frames/epoch
```

Note the epoch is now ~34.5k frames against ~60k before — the imbalance was
being "fixed" by oversampling `ylw2` with replacement, and removing that removes
duplicated frames rather than adding new ones. Epoch wall-clock should drop
roughly in proportion.

**Change nothing else in that run** (`training_tips.md` §D3). The action head is
the more promising lever, but running both at once makes neither attributable.

---

## 7. How to do (a)/(b) — the split script

Do **not** re-run `convert_lerobot_icrt.py`. Conversion redoes FK and video
decode for all 120 episodes; the cut is a pure restructuring of an HDF5 that
already exists.

New `aiworker_icrt/split_episodes.py`:

```
in:  --dataset /dev/shm/icrt_multitask
     --task "locate the yellow part and pick the part from the bin"
     --at grasp            # cut index = first rising edge of the gripper channel
     --labels pick,place   # omit for one label (option a)
     --out /dev/shm/icrt_multitask_split

for each episode of --task:
    find cut = first index where action[:, GRIPPER] crosses its midpoint upward
    write two groups, <ep>_pick = [0:cut] and <ep>_place = [cut:], copying every
    observation and action dataset with the same slice
copy every other episode unchanged
rewrite the three sidecars: hdf5_keys.json, epi_len_mapping.json, verb_to_episode.json
```

The gripper channel is index **19** for `ylw2` (right arm); the per-arm block is
`[xyz(3), rot_6d(6), gripper(1)]`, so left is 9 and right is 19
(`constants.SLICE_ARM_*`).

### After any data change, in this order

1. **Regenerate action statistics.** Delta-action means and stds change when
   episode boundaries move. Write a **new filename** — overwriting would corrupt
   validation of the runs that reference the old one (`training_tips.md` §B4).
2. **Re-run `audit_dataset`** on the new directory.
3. **Visualise one `_pick` and one `_place`** and watch them. A cut placed one
   frame late starts segment B with the gripper still open and the part still in
   the bin, and nothing numeric will tell you (`training_tips.md` §A5).
4. **Check the per-task FRAME share**, not the episode count.
5. Check the 72-frame outlier survived the `minimum_length` filter, or drop it.

---

## 8. What this does not fix

The left arm is frozen in **every** `ylw2` frame regardless of how they are
sliced or weighted. Rebalancing changes how often the model is told "the left arm
does not move"; it does not add a single frame where it does. And none of this
creates bimanual coordination — `box` and `ecu` have the left arm travelling
10–13× further than the right, which reads as steadying, not manipulation
(`training_tips.md` §C2).

The measured ranking of levers has not changed: the **action head** (`mlp` →
`diffusion`/`gmm`) remains the largest untested one, and normalisation — the last
thing that looked this promising — bought 2%.
