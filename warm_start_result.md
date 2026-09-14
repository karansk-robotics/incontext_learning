# Warm start vs from scratch — the controlled result

**A DROID-pretrained trunk plus the authors' CrossMAE encoder cuts held-out error
by 59% and stops the model overfitting. Trained from scratch on the same data,
held-out error gets monotonically WORSE after epoch 4.**

Run 2026-09-14 on `zeux`. Both runs: identical 151-episode grasp-split dataset,
identical action statistics, identical `task_grouping`, identical
hyperparameters, 21 epochs each. **One variable**: whether the transformer
started from ICRT's released DROID weights with their CrossMAE vision encoder,
or from random init with the stock ImageNet MAE encoder.

---

## 1. The result

Held-out **working-arm** position MAE, 8 held-out episodes, identical prompts:

```
epoch    mt21_warm    mt21_scratch_split    warm better by
    4     0.0513          0.0535                +4.1%
    8     0.0398          0.0539               +26.2%
   12     0.0308          0.0618               +50.1%
   16     0.0311          0.0706               +56.0%
   20     0.0308          0.0752               +59.1%
```

**Warm improves then holds.** 0.0513 → 0.0398 → 0.0308, flat through epoch 20.

**Scratch degrades from epoch 4 onward.** 0.0535 → 0.0752, a 41% *increase*,
while its training loss fell from 0.4841 to 0.3189. That is 92.6 M parameters
memorising 151 episodes, visible only because we scored held-out data.

`mt21_warm/checkpoint-12` at **0.0308 m** is the best policy this project has
produced — against the previous best of 0.0546 m (`mt21_gpu/checkpoint-15`), a
**44% reduction**. Caveat: the old figure was measured on the unsplit dataset's
6 held-out episodes, this on the split dataset's 8, so it is not a perfectly
clean comparison.

## 2. Training loss understated the effect by half

```
epoch    warm      scratch    train-loss gap   held-out gap
    4    0.4516    0.4841         6.7%             4.1%
   12    0.3026    0.4012        24.6%            50.1%
   20    0.2543    0.3189        20.3%            59.1%
```

Watching training loss alone, the warm start looks like a ~20% effect that is
slowly eroding as the control catches up. On held-out data it is a ~59% effect
that is still **widening**. The two curves say opposite things about the trend.

Worth recording plainly: through most of this run I read the widening
training-loss gap as a warning that the warm model might be memorising faster.
The opposite was true. Training loss could not distinguish the two hypotheses,
and only held-out error did.

## 3. Cross-embodiment transfer works here

This was the open question: their model is a **one-arm Franka, 10-D action,
2 cameras**; ours is **two arms plus a torso lift, 21-D action, 3 cameras**.

Every layer that touches dimensionality had to be discarded and randomly
re-initialised — 24 tensors: proprio encoder, action encoder, output head, camera
pooling. Only the transformer trunk transferred: **125 tensors, 88.6 M of 91.6 M
parameters (96.7%)**, verified loaded by cosine similarity against the source
(0.997-0.999 vs 0.13-0.33 for a from-scratch checkpoint).

What transfers is the trunk's model of *sequence* — how an action follows from a
history of observation tokens. That is apparently embodiment-independent enough
to be worth a 2x reduction in held-out error.

**This matters more than the number.** It means outside data is usable: the 857
BG2 episodes, ICRT-MT, Open X-Embodiment. Our data shortage becomes partly a
compute problem rather than purely a recording problem.

**Not separated:** this run changed the trunk *and* the encoder together. A third
run (CrossMAE encoder, random trunk) would say which did the work. The encoder
probe in section 5 suggests the encoder contributes.

## 4. What the policy actually does now

`mt21_warm/checkpoint-12`, per held-out episode:

```
episode                  task   |pred-cmd|  pred step  human step   obs-cmd   grip
box_episode_000009       box       18.6mm     4.72mm      5.50mm     11.2mm   100%
box_episode_000021       box       21.4mm     9.18mm     10.16mm     22.6mm   100%
box_episode_000036       box       48.1mm    11.24mm     13.91mm     31.1mm   100%
box_episode_000047       box       42.3mm     8.66mm     11.10mm     25.0mm    82%
ecu_episode_000017       ecu       39.5mm     2.87mm      5.79mm     36.1mm   100%
ecu_episode_000037       ecu       28.9mm     2.93mm      4.73mm     32.0mm   100%
ylw2_episode_000006_pick ylw2      21.5mm     2.15mm      4.12mm      0.4mm   100%
ylw2_episode_000013_pick ylw2      26.2mm     2.78mm      4.80mm      0.4mm   100%
```

**The hedging is partly gone.** The old model moved 51% of the operator's rate
(2.96 mm vs 5.79 mm) -- the signature of regression to the mean. Now:

```
box    86%, 90%, 81%, 78%      <- tracks the demonstrator
ecu    50%, 62%
ylw2   52%, 58%                <- still hedging
```

So the gain is real behaviour change, not a better-tuned hedge -- but only on one
of three tasks.

**Gripper agreement is 100% on 7 of 8 episodes.** The discrete decision is solved.

**And it still loses to the trivial baseline.** `obs-cmd` is the servo lag, which
is also the score of "command the pose you already observe":

```
                 ours    that baseline
box_000009      18.6         11.2      baseline
box_000021      21.4         22.6      POLICY
box_000036      48.1         31.1      baseline
box_000047      42.3         25.0      baseline
ecu_000017      39.5         36.1      baseline
ecu_000037      28.9         32.0      POLICY
ylw2_000006     21.5          0.4      baseline by 54x
ylw2_000013     26.2          0.4      baseline by 65x
```

**2 of 8.** And 0.0308 m remains 4.7x the 0.0065 m "repeat the previous action"
baseline -- better than the 8.4x we started at, still not a working policy.

One unexplained asymmetry: `obs-cmd` is **0.4 mm** on `ylw2` against 11-36 mm on
`box`/`ecu`. The follower tracks its command almost perfectly in those recordings.
Different controller gains or recording speed between sessions -- worth
understanding, because it makes `ylw2` episodes trivially easy to score well on by
doing nothing.

## 5. The encoder probe

Same 64 frames of a held-out episode through both frozen encoders, measured on the
197x768 patch-token grid the policy actually consumes (`global_pool=''`):

```
                        spread    adjacent-step   corr(feature dist, hand dist)
ImageNet MAE (default)  0.5476      0.2383              +0.288
CrossMAE rtx (theirs)   0.1796      0.1023              +0.343
```

CrossMAE moves less than half as much between frames yet correlates better with
actual hand travel -- more signal, less churn. But **both correlations are weak**,
on the wrist camera, where the whole scene shifts as the arm moves 5.68 m. A
frozen encoder is the entire visual pathway; what it cannot represent, the policy
can never recover.

## 6. What this does and does not change

**Does:** initialisation is no longer an open question. Use the warm start. And
pretraining on outside data is now a supported strategy rather than a guess.

**Does not:** condition 3 still fails, there is still no genuinely bimanual data,
there are still only 4 tasks, and each was recorded in its own scene at its own
head angle -- so the task is identifiable from one frame and the prompt is
redundant by construction. No initialisation fixes any of that.

The highest-value next action remains **recording two tasks in one scene**, which
is what makes the in-context claim testable at all.

## 7. Reproduce

```bash
./train_warm.sh                      # warm
./train_scratch_split.sh             # control
PYTHONPATH=third_party/icrt:. python sweep_heldout.py 4,8,12,16,20
```

Known rough edges in the harness used here, both worth fixing before reuse:
- `evaluate()` names output by `{eval}__prompt_{prompt}`, so sweeping epochs into
  one directory **overwrites** the per-episode trajectories. Use a per-epoch dir.
- The sweep picks the alphabetically-first train episode as the prompt, which
  after the split can mean a `_place` segment prompting a `_pick` query. Harmless
  here (identical for both runs, and condition 3 showed prompts change the error
  by ~1 mm) but wrong for any experiment where the prompt is meant to matter.
- `OMP_NUM_THREADS` unset means every torch process claims all 24 cores. Four
  parallel jobs drove load average to 68 on a 24-core box and made one checkpoint
  take 413 s instead of 45 s. Cap threads before running sweeps in parallel.
