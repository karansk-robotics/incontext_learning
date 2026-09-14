# Session report — 2026-09-14

**One defect found and fixed, one experiment settled, one claim tested and
failed. The policy is not usable, and the reason is not something another
training run fixes.**

| | value |
|---|---|
| best policy | `runs/mt21_warm_ach/checkpoint-8` |
| held-out working-arm MAE | **0.0309 m** |
| bar (predict no motion / repeat previous action) | **0.0074 m** |
| verdict | **4.2x above the bar** |
| in-context (condition 3) | **no effect** — 1.0 mm on a 55.6 mm error |

---

## 1. What was settled

### The warm start works — 59%, controlled

Two runs, identical 151-episode dataset, statistics, `task_grouping`,
hyperparameters and 21 epochs. One variable: ICRT's DROID-pretrained trunk plus
their CrossMAE encoder, versus random init plus the stock ImageNet encoder.

```
epoch    warm      scratch    warm better
    4    0.0513    0.0535        +4.1%
   12    0.0308    0.0618       +50.1%
   20    0.0308    0.0752       +59.1%
```

The control **degrades** from epoch 4 while its training loss falls — overfitting
invisible in the training curve. Warm improves and holds.

Cross-embodiment transfer works despite 10-D one-arm -> 21-D two-arm-plus-lift:
all 24 dimension-dependent tensors were discarded, only the trunk transferred
(125 tensors, 88.6 M of 91.6 M), verified by cosine 0.997-0.999 against source.

**Initialisation is now a closed question.** Use the warm start.

### The action target was three different quantities — fixed

`action` is the leader arm; `observation.state` is the follower. They differ by
the servo lag, which was tuned differently in each recording session:

```
box   21.1 mm      ecu   38.4 mm      ylw2   0.75 mm
```

Controller gains are not in the images or the proprio, so that part of the target
was unpredictable in principle. MSE's optimal answer to an unpredictable target
is its average — and the model emitted a constant ~22 mm delta everywhere. On
`ylw2` that was **65x worse than emitting zero**.

Fixed by taking the target from the achieved pose one step ahead. Verified:

```
|action - observation|   before  18.42 mm  [ 0.54 .. 51.89]
                         after    8.70 mm  [ 4.68 .. 14.00]

ylw2 predicted/true motion ratio   before  54.5x, 65.4x
                                   after    5.0x,  5.3x   (in line with all tasks)
```

Detail: `action_target_fix.md`.

### Condition 3 — the in-context claim — was run for the first time, and fails

```
prompt with the SAME task      0.0556 m
prompt with a DIFFERENT task   0.0565 m      1.0 mm on a 55.6 mm error
self-prompt (shown the answer) no better; worse on 2 of 8
```

Not a plumbing bug — predictions differ between prompt conditions by up to
38.7 mm, so the prompt reaches the model and carries nothing useful.
Detail: `condition3_result.md`.

## 2. The finding that matters most

**The label fix did not move the ceiling.**

```
old run, BROKEN targets   plateaued at 0.0308 m
new run, FIXED targets    plateaued at 0.0309 m   (best, epoch 8; 0.0314 at 10)
```

Two runs, completely different targets, the same wall. And the new run shows the
plateau internally: MAE fell 26% from epoch 4->6, then 1.3% from 6->8, then rose,
while training loss kept dropping (0.4535 -> 0.3084).

So the servo-lag bug was real, was worth fixing, and **was not what was holding
the model back**.

## 3. Why — confirmed by audit, not assumed

Each task was recorded at its own head angle, constant across every episode of
that task:

```
task    head1     head2       pairwise
box     0.466     0.348       box vs ecu    10.1 deg
ecu     0.643     0.348       box vs ylw2   18.2 / 31.4 deg
ylw2    0.784    -0.199       ecu vs ylw2    8.1 / 31.4 deg
```

Within a task the head varies by 0.18 deg. **The task is identifiable from a
single frame.** The model never needed the demonstration, so it never learned to
use one. That is why condition 3 is flat, and why it cannot currently be tested
fairly — "ignores the prompt" and "the prompt was redundant" predict identical
results.

Everything else audited clean: image and state stream lengths match exactly, the
three cameras are genuinely distinct, the head does not drift within episodes,
gripper sense verified, lift constant. `ylw2` carries 15.9% near-stationary
frames against ~8% elsewhere — minor, and ICRT deletes such frames where we keep
them.

## 4. Also done

- **`task_grouping`** — `ylw2` frame share 58.9% -> 28.3%. `rebalance_tasks` was
  already on and was *causing* the imbalance (it equalises episode count, not
  frames). An earlier doc said the opposite; corrected.
- **`ylw2` episodes cut at the grasp** — 120 -> 151 episodes, so a demonstration
  and an attempt fit in one 1024 window. Their median episode is 129 frames and
  puts ~4 in a 512 window; ours were 958.
- **Trajectory artifact** — predicted / commanded / observed traces for all 8
  held-out episodes.

## 5. What is actually needed

1. **Record two tasks in one scene, at one head angle.** Same table, two targets,
   so the prompt is the only way to know the task. ICRT-MT contains exactly this
   pair deliberately. Without it the paper's central claim cannot be tested at
   all — this is an afternoon of recording and it is worth more than any further
   training run.
2. **More tasks.** 4 against their 34.
3. **Genuinely bimanual episodes**, if that claim stays in the paper. Currently
   one arm is frozen or doing ~30 cm of support motion.
4. **A smaller model.** 92.6 M parameters against 151 episodes; the control
   overfit from epoch 4.

Not on this list, deliberately: more epochs (both runs plateau or degrade), more
loss tuning (normalisation bought 2%), a different action head (their working
model uses `mlp` too), or another initialisation (settled).

## 6. Honest summary

The engineering is sound and the instrumentation caught eight silent failures
today, including two introduced during the session. Three real defects were found
and fixed. None of them was the bottleneck.

**The bottleneck is the recordings**, and no amount of model work changes that.
