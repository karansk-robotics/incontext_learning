# Condition 3 — the in-context claim, tested

**Result: the prompt makes no difference. The policy is doing behaviour cloning.**

Run 2026-09-14 on `runs/mt21_gpu/checkpoint-15.pth`, the best checkpoint we have.
Reproduce with `python -m aiworker_icrt.condition3`.

---

## 1. Why this experiment and not the accuracy number

Conditions 1 and 2 — prompt with the same task, score accuracy — **cannot tell
in-context learning apart from ordinary behaviour cloning.** A policy that
ignores the prompt entirely scores identically on both. `PLAN.md §4` calls this
the central risk of the project and the go/no-go for the paper.

Condition 3 separates them: hold the observations fixed and change only the task
the prompt is drawn from. If the model reads the demonstration, that must hurt.

It had never been run. The old blocker — "`evaluate.py` cannot cross datasets" —
was dissolved when the three tasks were merged into one `ffw_sg2.hdf5`;
`--prompt` and `--eval` take arbitrary episode names in one file.

## 2. Method

- 6 held-out episodes (2 per task), from `val_split.json`.
- Each prompted 3 ways: its own task, and each of the 2 other tasks.
- 2 prompt episodes per task, **drawn only from the train split**.
- Metric: position MAE of the **working arm** over 40 steps. Scoring the idle arm
  measures nothing — in `ylw2` the left arm is frozen at exactly 0.000 m.
- 36 runs plus 6 controls.

**Prompt length is a confound and was controlled.** `ylw2` episodes average 958
frames against ~330 for `box`/`ecu`, so an uncontrolled comparison varies prompt
*length* at the same time as prompt *task* and can attribute neither. Every
prompt is truncated to its **last 300 frames** (`--prompt-max-steps`, added to
`evaluate.py` for this). The last frames rather than the first: the window stays
contiguous and in-distribution and ends at the episode boundary, which is exactly
where `create_prompt_mask` puts a prompt edge during training.

## 3. Result

```
eval episode             task    SAME-task    CROSS-task   delta
box_episode_000044       box     0.0688       0.0710       -0.0022
box_episode_000047       box     0.0862       0.0896       -0.0034
ecu_episode_000014       ecu     0.0722       0.0721       +0.0001
ecu_episode_000017       ecu     0.0462       0.0465       -0.0002
ylw2_episode_000025      ylw2    0.0237       0.0237       -0.0000
ylw2_episode_000027      ylw2    0.0363       0.0364       -0.0001
------------------------------------------------------------------
MEAN                             0.0556       0.0565       -0.0010
```

Same-task prompting wins by **1.0 mm on a 55.6 mm error — 1.8%**.

"Better on 5 of 6 episodes" reads well until the magnitudes are checked: four of
those five are between 0.0 and 0.2 mm. Only the two `box` episodes move at all,
2–3 mm against a 70–86 mm error. Swapping in a demonstration of an entirely
different task costs the policy essentially nothing.

## 4. The control: self-prompting does not help either

Prompted with the **exact episode about to be replayed** — the strongest prompt
that exists, since the model has been shown the answer it will be scored against:

```
eval                   task  SELF      same      cross     self vs cross
box_episode_000044     box   0.0716    0.0688    0.0710    +0.0006
box_episode_000047     box   0.0912    0.0862    0.0896    +0.0016
ecu_episode_000014     ecu   0.0701    0.0722    0.0721    -0.0020
ecu_episode_000017     ecu   0.0463    0.0462    0.0465    -0.0002
ylw2_episode_000025    ylw2  0.0236    0.0237    0.0237    -0.0001
ylw2_episode_000027    ylw2  0.0364    0.0363    0.0364    +0.0000
```

No better than a prompt from a different task; **2 of 6 are slightly worse.**
There is no prompt that helps this policy.

## 5. It is not a plumbing bug — checked, not assumed

On `ylw2` the same-task and cross-task errors agreed to **four decimal places**.
A weakly-used prompt should still perturb the output. Identical to 0.1 mm raises
the obvious alternative: the prompt never arrives.

That distinction has been wrong eight times on this project, so it was measured
rather than argued. Comparing the raw predicted action arrays:

```
same-task vs cross-task predictions
   bitwise identical : False
   max abs diff      : 3.870e-02      (38.7 mm)
   mean abs diff     : 1.416e-03
```

**The prompt does reach the model and does change the computation** — by up to
38.7 mm per channel. It flows through `prompt()` → `forward_inference` → the KV
cache at `[0, start_pos)` → `get_action_eval`, and it moves the output. It just
moves it *without structure*. The perturbation is noise, not conditioning.

So this is a real negative result, not the ninth silent failure.

## 6. What it means

**The policy attends to the demonstration and extracts nothing from it.** On this
checkpoint ICRT is behaving as plain behaviour cloning. That is `PLAN.md §4`'s
central risk, measured.

**Two caveats keep this from being a verdict on ICRT:**

1. **This checkpoint is already ~8× worse than repeating the previous action**
   (0.0546 m vs the 0.0065 m null baseline). A model that cannot do the task at
   all cannot be expected to show prompt sensitivity. **This experiment must be
   repeated on any checkpoint that beats the baseline** before the negative can
   be attributed to the method rather than to the run.
2. **It was trained in conditions where the prompt mechanism barely existed.**
   With `seq_length 1024` against ~958-frame `ylw2` episodes, those windows rarely
   contained a demonstration *and* an attempt — the pairing `create_prompt_mask`
   needs. ICRT-MT's median episode is 129 frames, so their 512 window holds ~4.
   And 3 tasks is thin ground for learning that a prompt carries task identity at
   all, against their 34.

There is also the possibility this experiment cannot rule out: that each task is
identifiable from a single frame, because each was recorded in its own scene at
its own head angle — making the prompt redundant rather than unusable. Both
readings predict exactly what was measured. Distinguishing them needs either two
tasks in one scene (ICRT-MT has this deliberately: `context-top-yellow-cup` and
`context-bottom-orange-cup`, 50 episodes each) or a model that works at all.

## 7. What follows

Nothing here changes the priority order, it sharpens it:

1. **Warm start** from the DROID-pretrained trunk and CrossMAE encoder
   (`using_icrt_pretrained.md`) — 95.7% of our parameters transfer.
2. **More tasks**, and ideally two tasks sharing one scene, so the prompt has
   something to disambiguate that the image cannot.
3. **Shorter `ylw2` episodes** (`task_balance_fix.md` §7) so a window can hold a
   demonstration and an attempt.
4. **Re-run this experiment** on the first checkpoint that beats 0.0065 m.

Until (4), treat the accuracy numbers in `documentation.md` as measuring an
**unprompted** policy, because that is what they measure.
