# Evaluation

## What the evaluation is for

**Not "how accurate is the policy". It is "does the policy read its prompt at
all".**

A policy that tracks the recorded trajectory well but ignores which demo it was
prompted with has **falsified ICRT on this data** — it learned the task from the
training set, not from the prompt. Low position error is necessary and nowhere
near sufficient.

And the obvious test is not sufficient either. Comparing a same-task prompt
against a different-task prompt looks like it measures this, **but on this data
it does not** — every task was recorded in its own scene at its own camera
angle, so the model can identify the task from a single frame and ignore the
prompt entirely while still winning that comparison.

The experiment that actually separates the two is **condition 3** below:
prompt with one task, show observations from another, and see which one the
policy obeys. Everything else in this document is measurement; that is the
question.

---

## Before you convert: audit the dataset

```bash
python -m aiworker_icrt.audit_dataset --lerobot-root <dir>
```

Run this **before converting any new recording set**. It measures how much the
head, lift and base actually move and recommends a configuration. The
distinction it draws is the one that matters:

- **Within-episode movement breaks causality.** If a joint moves during a demo
  and the policy cannot command it, the policy is being asked to reproduce an
  effect whose cause it has no access to. That joint must be commanded.
- **Between-episode variation behaves as augmentation.** A joint that is fixed
  within each episode but differs across them is something the policy only needs
  to *observe*.

CPU-only. No GPU, no model, no checkpoint.

---

## Preflight: point the head where the demonstrations had it

> **DORMANT — there is no physical robot (PLAN.md rev 2).** Kept because it
> applies the moment one exists, or in Gazebo. Nothing below runs today.

**Before inference, set the head to the pose the demonstrations used.** For the
ECU set that is commanded `[0.6427, 0.3496]` rad — constant in every frame of
all 39 episodes.

Point it anywhere else and `cam_head` is off-distribution with **no error
raised**. The two wrist cameras still look correct, so the failure presents as a
bad policy rather than a mispointed camera — which is a long way to debug from
the wrong end.

The converter now stores `action/joint_position` (22-D), so this target travels
with the prompt episode instead of living as a hardcoded constant. Read it from
there rather than copying the numbers above.

---

## Running it

```bash
python -m aiworker_icrt.evaluate \
  --dataset data/icrt_ecu \
  --checkpoint output/<run>/checkpoint-N.pth \
  --train-yaml output/<run>/run.yaml \
  --prompt episode_000000 \
  --eval   episode_000005 \
  [--max-steps 60] [--device cuda]
```

Writes into `<dataset>/eval/`:

| File | Contents |
|---|---|
| `.json` | metrics, per arm |
| `.png` | predicted vs recorded traces |
| `.npz` | raw arrays, for your own analysis |

**Requires a GPU.** `--device` defaults to `cuda`, and this is not merely a
speed preference: `third_party/icrt/icrt/models/policy/icrt.py` builds the
transformer under `torch.device("cuda")`, so the model fails at **construction**
without a visible GPU, not at the forward pass. There is no CPU path.

The UI's **Evaluate tab** is the front end for this, with prompt-episode and
eval-episode as separate dropdowns specifically so the comparison below is two
clicks apart.

---

## A note on the 23/21 action space

Proprio is 23-D (both arms, plus head ×2 and lift, observed) and action is 21-D
(both arms, plus lift, commanded). Evaluation metrics below are computed on the
**per-arm Cartesian blocks**, which are unchanged — so the numbers remain
comparable across the dimension change.

**Caveat that belongs next to any result from the ECU set:** head and lift are
constant there, so those channels carry no information in that data. The extra
dimensions are verified to work *mechanically* — shapes, gradients, inference
width. Whether they help is a separate claim with no evidence yet, and it cannot
get any until a dataset arrives in which they vary.

---

## Metrics

| Metric | Meaning |
|---|---|
| `pos_mae`, `pos_rmse`, `pos_max` | metres, per arm, EEF position error |
| `gripper_mae` | on the normalised 0–1 signal |
| `gripper_agreement` | fraction of steps where the binary open/closed decision matches (threshold 0.5) |

Two conventions that have each already caused a real bug:

- **Frame is `base_link`**, not `arm_base_link`. `z` is height above the floor
  (~0.58–1.30 m on the ECU set), not a shoulder-relative number.
- **Gripper: `0 = open`, `1 = closed`** (RH-P12-RN). Verified against
  wrist-camera frames, not assumed — it was backwards until the plot
  contradicted the task description.

---

## First real run — a PLUMBING result, not a performance result

1-epoch checkpoint, 39 single-task episodes, 60 steps, prompt `ep0` / eval `ep5`:

| Arm | pos_mae | rmse | max | gripper_agreement |
|---|---|---|---|---|
| left | 0.126 m | 0.165 | 0.305 | 1.00 |
| right | 0.017 m | 0.027 | 0.144 | 1.00 |

**Read this as "the pipeline is closed", nothing more.** 12.6 cm of error on an
arm with roughly 0.9 m of reach is not a usable policy. ICRT's own recipe is
**125 epochs**; this was one. The number worth taking from this run is that it
ran at all — inference, checkpoint loading and the bimanual action path all
work end to end.

Measured GPU during training (seq_length 512):

| | util | power | temp |
|---|---|---|---|
| idle | 1% | 5.2 W | 44 °C |
| training | **96%** | 62 W | 71 °C sustained |

torch peak 9812 MiB. `cuda available: True` is a capability check; 96% sustained
is evidence of use. The UI header shows device, utilisation and torch-reported
memory live, refreshed while a GPU job is in flight.

> GPU memory comes from **torch**, never `nvidia-smi`, on both machines. GB10 is
> unified-memory and reports `[N/A]` for `memory.used`; the RTX PRO 6000 has
> discrete VRAM and would report a real number. Reading whichever works would
> show different quantities under the same label on the two boxes.

---

## The gate — three conditions, and the third is the experiment

**Run the same eval episode three times, changing only the prompt.**

| | prompt | observations | what it tells you |
|---|---|---|---|
| **1** | same task | task A | baseline |
| **2** | different task | task A | control |
| **3** | **task B** | **task A** | **does the policy follow the PROMPT or the SCENE?** |

### Why conditions 1 and 2 alone are not evidence

The obvious reading — "same-task prompting scores better, therefore the model
reads the prompt" — **does not follow on this data**, because a second
explanation fits the same result exactly.

Each task was recorded in its own physical setup *and* at its own camera angle:

```
ECU        head [+0.6427, +0.3496]
pick_box   head [+0.4663, +0.3482]
yellow     head [+0.7839, -0.1994]
```

Up to 18° and 31° apart. **Head angle alone predicts the task perfectly, and so
does the scene.** A model that recognises "this is the ECU bin" from a single
frame never needs to read the prompt at all — and it will still beat a
different-task prompt, because that prompt is noise it can ignore.

So a pass on 1 + 2 is consistent with in-context learning *and* with pure scene
recognition. It cannot distinguish them. That is not a robustness caveat; it
means the headline experiment has not been run.

### Condition 3 is the one that separates them

Prompt with task **B** while showing observations from task **A**, and see which
one the policy obeys.

- **Follows the prompt** (attempts task B's behaviour) → in-context learning.
  This is ICRT's claim, demonstrated.
- **Follows the scene** (carries on doing task A) → recognition. The prompt is
  decorative. The model has learned "ECU bin ⇒ ECU motions" and would behave
  identically with the prompt removed.

It costs nothing — no new data, no retraining, only a different pair of episodes
already on disk. **Run it first**, not last: if the answer is "scene", conditions
1 and 2 were never measuring what their names suggest.

> Practical requirement: prompt and eval episodes must live in the **same
> converted dataset**, because `evaluate.py` opens one hdf5 and looks both
> episodes up in it. Convert all tasks together into one multi-task set — which
> ICRT needs anyway for `verb_to_episode` prompt masking. The UI's Evaluate tab
> groups episodes by task and names the condition you have just built.

### If the model follows the scene

Two honest options, and choosing between them is the user's call:

1. **Report the negative result about this data.** "ICRT's in-context mechanism
   is not demonstrable on a dataset where task identity is visible in every
   frame" is a real finding, and a more useful one than a number that cannot be
   interpreted.
2. **Record same-scene different-task episodes.** Same bin, same camera pose:
   *"pick the yellow part"* vs *"pick the blue part"*. This is the only setup
   where the observation is genuinely ambiguous and the prompt is the sole
   disambiguator — and it is the single highest-value addition anyone could make
   to this project. Everything else is plumbing that already works.

**Nothing should be built on a positive result from conditions 1 and 2 alone.**

## Before you trust any of it: look at the episode

The **Visualize tab** renders video and per-arm EEF traces on one shared
playhead — state in red, action in blue, overlaid on the same axis so a tracking
error is visible at a glance. Scrub the video and the chart cursors follow;
click a chart and the video seeks.

This screen caught an inverted gripper that 28 passing tests did not. Run it on
a couple of episodes before spending GPU hours on them.
