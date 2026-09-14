# Making the model actually learn — tips from five sessions

Every tip below was given at some point across the five Claude sessions on this
project, most of them after something had already gone wrong. This file collects
them in one place so the next run does not rediscover them.

**Each entry states the evidence.** Where a tip is an untested hypothesis it says
so. Where a tip was tried and did *not* work, it stays in the list with the
measured result, because "we tried that and it bought 2%" is worth as much as a
success.

Companions: `documentation.md` (the measured record), `PLAN.md` (forward-looking),
`EXECUTION.md` (what was built), `validations.md` (what is proven), `eval.md`
(how to score it).

---

## The one-paragraph summary

The model trains, the loss falls, and the policy still cannot place its hand.
The loss was measuring the wrong thing (fixed — bought 2%), the data is half
"the left arm does not move", and the action head is plain MSE, which under
uncertainty is rewarded for predicting *nothing*. The untested lever with the
largest expected effect is **one flag**: `--model-cfg.policy-cfg.decoder-pred-head
diffusion`. Everything else in this file is either a trap that silently prevents
learning or a way to see the failure sooner.

---

## A · Traps that silently stop the model learning at all

None of these raise an error. Each cost real debugging time.

### A1 · `seq_length` must EXCEED the longest episode, in frames

With `--no-prompt-loss`, loss is computed only *after* an episode boundary falls
inside the window. No boundary in the window → no loss contribution.

```
measured on 280-412 frame episodes:
  seq_length = 32    only 10% of windows contain a boundary
  seq_length = 512           95%
```

At 15 fps a 60 s episode is 900 frames, so **use 1024**. Training runs, reports no
error, and the loss line reads a flat `0.0000`.

### A2 · `maximum_length` silently discards every episode

`SequenceDataset.maximum_length` was a hidden class attribute of **450 frames**,
tuned for ~15 fps. A 30 fps recording doubles the frame count and every episode
vanished — `len(dataset) == 0`, no error. Now exposed as
`DatasetConfig.maximum_length` and it raises naming the actual lengths.

### A3 · `min_demos = 4` deletes small tasks

A task with fewer than 4 episodes does not survive the train/val split and
disappears from `verb_to_episode`. Smoke tests hit this constantly and it looks
exactly like a converter bug.

### A4 · Audit the dataset BEFORE converting it

```bash
python -m aiworker_icrt.audit_dataset --lerobot-root <dir>
```

It measures whether head, lift and base actually hold still, and recommends the
configuration. Running it after conversion means re-converting.

### A5 · Look at an episode. Numbers do not catch sign errors.

```bash
python -m aiworker_icrt.visualize --dataset data/icrt_multitask --episode ecu_episode_000000
```

The gripper sense was inverted in our labelling — RH-P12-RN is raw **0 = OPEN,
1.1 = CLOSED** — and no assertion caught it. Plotting did: the gripper appeared to
open during transport and close at the drop, which is nonsense for pick-and-place.
The mapping is monotonic so the model was unaffected, but the sense matters
wherever a threshold is applied (ICRT's `binary_gripper` uses `> 0.5`).

### A6 · A green training run says nothing about inference

ICRT's rot_6d re-orthogonalisation rebuilt the action as `cat([xyz, b1, b2,
gripper])` — one 10-D arm. Given a 20-D bimanual action it truncated to 10 and
**silently dropped the right arm**. Training never calls `forward_inference`, so
**28 passing tests, a clean forward/backward and a COMPLETE TRAINING RUN were all
green** while this was broken. On the robot the right arm would never have moved.
`tests/test_inference_path.py` now guards it — keep it in the loop.

### A7 · `wandb` permission-denied kills the training process outright

The original run inherited `WANDB_MODE` from an interactive shell; `train.log`
showed no wandb lines and gave no hint it was a dependency. Scripts now set
`WANDB_MODE=disabled` explicitly. Do not remove it.

### A8 · `torch.load` fails on every checkpoint under torch ≥ 2.6

`weights_only` flipped to `True` and ICRT pickles an `ExperimentConfig` into the
`.pth`. Breaks inference, `--resume`, and any checkpoint released with the paper.
Loading also needs `PYTHONPATH=third_party/icrt` or you get
`ModuleNotFoundError: No module named 'icrt'`.

---

## B · Make the loss measure what you care about

### B1 · Unweighted MSE lets the units decide what matters

ICRT regresses the delta action with an unweighted MSE, so **each channel's share
of the target variance is its share of the gradient.** Measured over 898,192
chunked timesteps from all 114 training episodes:

```
translation    5.22%     <- what end-effector position error measures
rotation      33.0%
gripper       61.80%
```

The gripper is ~25× larger in scale than the hand's vertical motion — not because
it matters more, but because hands move centimetres (small decimals in metres)
and grippers swing across their whole 0-to-1 range. A falling loss was entirely
consistent with 12 cm of hand error.

Concretely, before normalisation:

```
position error 0.01 m  -> squared 0.0001
gripper  error 0.05    -> squared 0.0025      the gripper mistake counts 25x more
```

### B2 · Turn `scale_action` on — it already exists

Both halves were wired and the stats file was simply missing: target at
`icrt.py:632`, inverse at inference `icrt.py:764`. Generate the stats with
`make_action_stats.py`, pass via `--shared-cfg.scale-action`. Translation goes
from 5.2% to 28.6% of the gradient.

**Measured effect: 2%.** (`0.0546 → 0.0536` mean working-arm MAE; ~4.5%
like-for-like on epochs.) The diagnosis was right about the mechanism and **wrong
about it being the bottleneck**. Keep the fix — it is correct — but do not expect
the next loss-weighting idea to do better.

### B3 · A near-constant channel needs a tolerance, not `std` and not `1.0`

The lift measures std 0.58 mm and is constant by construction (0/120 episodes move
more than 1 mm). Two wrong answers:

- **Divide by its own std** — amplifies pure sensor noise ~1700× into a
  full-scale training target.
- **Set std = 1.0** — worse, and measured being worse. Every other channel scales
  *up* to spread 1 while the lift stays at its raw 0.0006, so it gets essentially
  zero gradient and nothing holds it at zero. On checkpoint-6 of the `std = 1.0`
  run the commanded lift drifted to **7.6–15.4 mm on all six held-out episodes**,
  past the 5 mm safety limit, where the unnormalised run had held 2.2–4.2 mm.

The right answer is a **physically meaningful scale: the tolerance beyond which
the channel is wrong.** For the lift that is 5 mm, the limit in
`validate_checkpoint.py`. `SAFETY_SCALE = {20: 0.005}` in `make_action_stats.py`.

### B4 · Never overwrite `action_stats.json` in place

It would have corrupted validation of the run that referenced it — lift unscaled
by 0.005 instead of 1.0, 200× off. Write `action_stats_v2.json`.

### B5 · Measure channel variance over ALL prediction steps

A first measurement sampled only prediction step 0 and understated translation
variance ~30× (0.17% vs the true 5.22%). The dataset anchors all 16 steps to the
same proprio. Reproduce the dataset's exact chunking — `convert_multi_step` then
`convert_delta_action` — or the number is fiction.

### B6 · Loss is not comparable across runs once you normalise

Normalised targets have unit variance, so `mt21_scaled`'s 0.208 and `mt21_gpu`'s
0.0097 measure different things. **Only held-out position MAE compares.**

---

## C · What the data can and cannot teach

### C1 · Half the frames teach "the left arm does not move"

```
task    eps   frames   % data   working arm   L travel   R travel
ylw2     30   29,143    49.1%   right           0.000 m    5.752 m
box      50   16,305    27.5%   left            3.244 m    0.326 m
ecu      39   13,357    22.5%   left            4.067 m    0.301 m
ylw1      1      506     0.9%   right           0.000 m    3.041 m
```

`ylw2` is 25% of the episodes but **49% of the frames**, because those recordings
are long — ~958 frames per episode against ~330 for `box` and `ecu`.

**`rebalance_tasks` does NOT fix this, and it is already on.** All six runs
(`mt21`, `mt21_gpu`, `mt21_scaled`, `mt21_tuned`, `mt21_v2`, `mt21_w6`) have
`rebalance_tasks: true` in their `run.yaml` — the upstream default, never
disabled. An earlier version of this file said it was off and recommended
enabling it. That was wrong.

**It makes the imbalance worse.** `rebalance_length` is the *median number of
episodes per task* (`dataset.py:186`), so it equalises episode **count**, not
frames — and our episodes are not the same length. Measured on the 114 training
episodes of `mt21_gpu`:

```
task   train eps   mean len   raw share   after rebalance (37 eps each)
ecu           37      342.6       22.6%        12,678 frames   21.1%
box           48      326.4       27.9%        12,078 frames   20.1%
ylw2          29      958.3       49.5%        35,456 frames   58.9%   <- replace=True
```

`ylw2` has 29 episodes against a rebalance length of 37, so it is sampled **with
replacement** — the one task we want less of is the only one being oversampled.
The left-arm-frozen share goes **49.5% → 58.9%**.

ICRT has no frame-level balancing flag. The options are to cut the long `ylw2`
recordings into shorter episodes, drop some of them, or record the other tasks
longer. `task_grouping` (§C6) can also down-weight `ylw2` by a hand-set ratio,
which is the cheapest lever — but it scales the *episode count* too, so the
ratio has to absorb the ~3× length difference.

### C2 · No training setting creates behaviour that was never demonstrated

Where both arms move (`box`, `ecu`) the left travels **10–13× further**; the
right's ~30 cm reads as steadying, not manipulation. There is no episode where
two arms cooperate. Normalisation already gives each arm 47.6% of the gradient —
10 channels each — so the *loss* balance is fixed. The *behaviour* is not in the
recordings. **A bimanual claim in the paper needs two-arm recordings.**

### C3 · Decide fps BEFORE converting a second dataset

The 857 BG2 episodes (`ROBOTIS/Task_0002_OrderPicking_lerobot`) are natively
10 fps and cannot be upsampled. Converting SG2 at 15 fps and BG2 at 10 fps would
hand one model **two delta scales for identical motion**. Pick one rate for both.

### C4 · 120 episodes against 92.6 M trainable parameters is severely overparameterised

More data is the second-biggest lever after the action head. Pretraining on the
public ROBOTIS AI Worker sets (718 and 268 episodes) plus ICRT-MT and finetuning
on ours restores the in-context story if our own set stays thin on task variety.

### C5 · State constant channels precisely in the paper

The lift's global range is 1.4 mm, max within-episode motion 0.01 mm. Harmless
for training (a constant input is a bias; a constant delta target costs ~0 loss)
and it is the *safe* configuration at inference. Write it as: "21-D (20 actuated
in this campaign; the torso lift is in the space but held fixed throughout these
recordings)."

### C6 · `task_grouping` — the composition lever we have never used

The authors group their 34 tasks into 6 action primitives and up-weight each by
hand (`config/data_config/icrt_mt/task_grouping.json`):

```json
"ratios": { "drawer": 0.8, "push": 0.8, "poke": 1.5,
            "pick_up": 0.8, "pick_place": 2.5, "stacking": 2.5 }
```

The ratio multiplies that group's rebalance length (`dataset.py:391`), so it is a
direct dial on how often each task is sampled per epoch. Our
`dataset_config.json` has no `task_grouping` key at all. It is the only
per-task weighting ICRT exposes.

---

## D · The biggest untested lever: the action head

### D1 · MSE predicts the average, and the average is "don't move"

Measured on `checkpoint-15`:

```
policy moves per step:   2.96 mm
teleoperator moves:      5.79 mm      <- the policy moves about HALF
policy absolute error:  44.50 mm      <- yet ends up 8x its own motion away
```

Smooth, timid motion plus a large standing error is the signature of **regression
to the mean**. When the model is uncertain which way the arm should go, MSE's
optimal answer is the average of all plausible moves — roughly *nothing*. It is
not confused; it is hedging, and hedging is what MSE rewards.

### D2 · ICRT ships two heads that do not average, both unused

`decoder_pred_head: Literal["mlp", "gmm", "diffusion"]`. The project uses `"mlp"`,
the default. `GMMHead` and `DiffusionHead` are both fully implemented in
`icrt/models/policy/pred_head.py` and have **never been run**.

```bash
--model-cfg.policy-cfg.decoder-pred-head diffusion \
--model-cfg.policy-cfg.num-train-diffusion-steps 100 \
--model-cfg.policy-cfg.num-inference-diffusion-steps 10
```

A diffusion head samples from the action distribution instead of collapsing it,
which is why diffusion policies replaced MSE regression for this class of problem.
Try `gmm` too — cheaper at inference and often sufficient. ~2 h per run.

**This is the strongest untested hypothesis and the cheapest to test.**

### D3 · Change one thing per run

Two changes at once means you cannot attribute the result. Each run is ~2.2 h;
three clean data points beat one ambiguous one. This is why no second flag was
added to the normalisation run mid-flight.

---

## E · Seeing overfitting before it costs a run

### E1 · It bottomed at epoch 15 and got worse

```
mt21_gpu:   epoch 5    0.0561
            epoch 15   0.0546   <- best
            epoch 35   0.0700   <- worse, while train loss fell 2.5x
```

Held-out error degraded on **5 of 6 episodes** from epoch 15 to 35 while training
loss improved 2.5×. Three independent measures — MAE, IK residual, lift command —
moved the same way on 6 of 6.

### E2 · Score validation DURING training

`log.txt` has no `val_loss` column. Add one before the next run, or overfitting
shows up only in a post-hoc sweep over checkpoints — which is how it was found.

### E3 · Shrink the model

12 layers / 768 dim / 92.6 M trainable is large for 120 episodes. Halving it
reduces memorisation.

### E4 · Unfreeze the vision encoder only with evidence

`vit_base_patch16_224.mae` is frozen. Unfreeze (or use ICRT's cross-MAE
checkpoint) **only** if a run shows visual features are the bottleneck — ICRT's
own README warns it costs > 2 days on 8×A100.

---

## F · What will NOT help

- **More epochs.** Held-out error bottomed at epoch 15 and got worse.
- **More loss tuning.** Normalisation was the clean version of that experiment
  and it bought 2%.
- **Resuming the bad run.** Normalised and unnormalised targets are different
  objectives; the checkpoint's decoder is fitted to the old scale.

---

## G · Calibrate what "exact" means

The commanded action **leads** the measured state by **2–4 cm** on the working arm
(`|action − observation|` mean 0.0207 m left, 0.0012 m right). That is servo lag,
baked into the recordings. The model is trained to predict the *commands*, so this
is not unfair scoring — but zero error is not the target.

Score against the null baselines, always:

| | position error, 40 steps |
|---|---|
| "repeat the previous action" | **0.0065 m** |
| "command the currently observed pose" | 0.0417 m |
| best checkpoint (`mt21_gpu/checkpoint-15`) | 0.0546 m |

The policy is currently **6–15× worse than repeating the previous action**, a
function it could represent in one layer. **The useful goal is beating 0.0065 m,
not reaching zero.**

And position MAE says "close to what the human did", not "picked up the ECU
part". There is still **no task-completion measure** — that needs a rollout.

---

## H · The head, the lift, and the camera

- The **head is observed but never commanded.** It carries `cam_head`, so the
  policy needs to know its own gaze to interpret the image, not to change it. It
  stays wherever the operator sets it — and **must be set to the pose the
  demonstrations used**, or the main camera is off-distribution *with no error
  anywhere*. Nothing currently verifies this; a preflight head check is ~20 lines
  and has been deferred twice.
- The **lift is both observed and commanded.** It sits inside the arm chain, and
  if a demonstrator raised the torso to reach something, that is part of the
  skill. Referencing from `arm_base_link` would have made the lift vanish from
  the maths and made the policy unable to reproduce that behaviour at all.
- EEF poses are referenced from **`base_link`**, which sits at floor level, so
  EEF z is height above the ground.

---

## I · Preflight checklist for the next run

```
[ ] audit_dataset run on any new data, BEFORE converting
[ ] one fps for every dataset in the merge
[ ] seq_length (1024) > longest episode in frames
[ ] maximum_length > longest episode in frames
[ ] every task has >= 4 episodes
[ ] visualize one episode per task and watch it
[ ] action_stats written to a NEW filename, near-constant channels on a tolerance
[ ] WANDB_MODE=disabled set in the launch script
[ ] val_loss column added to log.txt
[ ] per-task FRAME share checked, not episode count -- rebalance_tasks equalises counts
[ ] exactly ONE variable changed vs the previous run, and it is named in the run dir
[ ] test_inference_path.py green — training green proves nothing about inference
[ ] baseline numbers (0.0065 / 0.0417) in front of you before reading any result
```

---

## J · The recurring lesson

Eight-plus instances now, all the same shape: **a green signal from a check that
never exercised the failing path.** The training loss fell for 46 epochs while the
model could not place its hand. The safety gate's own `ik_converged` check failed
on ground-truth data. A `joint_limits` check flagged the robot standing still.
`import cv_bridge` succeeded while its conversion segfaulted. A complete training
run was green while the right arm was structurally discarded.

The discipline that caught every one of them was the same: **run a control.** Feed
the recorded human demonstration through the identical code path. Anything that
fails on both is measuring the robot, the solver, or your threshold — not the
model.

> *Convergence measures agreement with the question asked, never whether it was
> the right question.*
