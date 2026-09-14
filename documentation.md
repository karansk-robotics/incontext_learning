# ICRT × ROBOTIS AI Worker — Training & Validation Record

**Date:** 2026-09-14 · **Machines:** DGX Spark GB10 (aarch64, local) · `zeux` RTX PRO 6000 Blackwell 96 GB (x86_64)

This document records what was built, what was measured, and what was learned in
the session that took the project from "training runs, loss goes down" to a
measured understanding of why the policy does not work yet. It is deliberately
written around **numbers that were measured**, not intentions.

Companion documents: `PLAN.md` (forward-looking), `EXECUTION.md` (what was built),
`validations.md` (silent-failure catalogue), `eval.md`, `changes.md`.

---

## 1. Headline result

**No checkpoint is fit for hardware, and the best one is beaten by a trivial baseline.**

| | value |
|---|---|
| best checkpoint | **`runs/mt21_warm/checkpoint-12.pth`** (was `mt21_gpu/checkpoint-15`) |
| working-arm position error (40 steps) | **0.0308 m** (was 0.0546 m -- see `warm_start_result.md`) |
| same, at 120 steps | **0.1244 m** |
| "repeat the previous action" baseline | **0.0065 m** |
| "command the currently observed pose" baseline | 0.0417 m |
| safety gate | FAIL — 2/6 held-out episodes clean |

The policy is **6–15× worse than repeating the previous action**, a function it
could represent in one layer. Error grows with horizon (0.045 → 0.124 m from 40
to 120 steps), which indicates drift rather than constant noise.

---

## 2. What was built

### `aiworker_icrt/validate_checkpoint.py` — pre-hardware safety gate

Answers a different question from `evaluate.py`. That one asks *"does the policy
reproduce the demonstration"* (accuracy); this asks *"if this were wired to a
robot, would anything break"* (safety). A policy can score a respectable MAE while
emitting one 40 cm jump between consecutive steps, and the average hides it.

Runs entirely on recorded data — no robot, no simulator. Every check is kinematic,
computed from the same `DualArmICRT.step` call `ros_node` would make, so IK
warm-starting, live-lift seeding and per-arm Gram-Schmidt are exercised as they
would be on hardware. This matters: `policy.step` had **never been executed
end-to-end** before this gate ran. Training never calls `forward_inference`, which
is how a silently discarded right arm survived 28 green tests and a full training
run earlier in the project.

| check | what it catches |
|---|---|
| `ik_reachable` | requested pose is not reachable at all |
| `joint_limits` | target outside the URDF range → driver clamps, policy keeps integrating |
| `joint_velocity` | consecutive targets implying > 4.8 rad/s |
| `eef_speed` | the Cartesian equivalent — what a human in the cell feels |
| `gripper_chatter` | rapid open/close flips |
| `lift_command` | torso motion — the one with real injury potential |
| `arm_proximity` | the two hands converging (EEF-distance proxy, not full self-collision) |

Writes a per-episode JSON verdict **and** per-episode trajectory files
(`validation/trajectories/<ckpt>__<episode>.json`) containing predicted,
ground-truth and observed poses. Emitted on every run, not behind a flag: a
verdict without the curve behind it is not reviewable, and the rollout is far too
expensive to repeat just to look at it.

### `aiworker_icrt/make_action_stats.py` — per-channel action statistics

Computes mean/std/min/max over all training-split delta actions, reproducing the
dataset's exact chunking (`convert_multi_step` → `convert_delta_action`). Writes
the JSON consumed by `--shared-cfg.scale-action`.

### `aiworker_icrt/cyclo_bindings/` — pybind11 wrapper for cyclo_control

`cyclo_py.cpp` + `CMakeLists.txt` + `build.sh`. Exposes `KinematicsSolver` and
`AIWorkerBimanualMoveLController` to Python. **Written but never compiled** — see
§7.

### Gazebo camera sensors

`third_party/ai_worker/ffw_description/gazebo/ffw_sg2_rev1_follower/ffw_sg2_follower.gazebo.xacro`
gained three `<sensor type="camera">` blocks (the shipped model had only two
`gpu_lidar` sensors). **Written but never expanded with `xacro`** — see §7.

### Two published pages

- Validation report — all six held-out episodes, predicted vs truth, deviation,
  per-check verdicts. Fed automatically by `validate_checkpoint.py`.
- Normalisation explainer — 10 slides on the loss-weighting diagnosis.

---

## 3. The central finding: the loss was measuring the wrong thing

The training loss fell from 0.2152 to 0.0097 over 46 epochs while held-out hand
error sat at 12 cm. Decomposing the loss by channel explains why.

ICRT regresses the delta action with an **unweighted MSE**, so each channel's
share of the target variance is its share of the gradient. Measured over 898,192
chunked timesteps from all 114 training episodes:

```
translation   5.22%    <- what end-effector position error measures
rotation     33.0%
gripper      61.80%
```

The gripper is ~25× larger in scale than the hand's vertical motion — not because
it matters more, but because hands move centimetres (small decimals in metres)
and grippers swing across their whole 0-to-1 range. **The units silently decided
what the model considered important.** A falling loss was entirely consistent with
12 cm of position error.

### The fix, and its measured effect

`scale_action` (standardise each channel by its own std) already existed in ICRT
with both halves wired — target at `icrt.py:632`, inverse at inference
`icrt.py:764`. It was set to `null`. Enabling it moves translation from 5.2% to
28.6% of the gradient.

**Result: 2%.**

```
                    OLD ckpt-15   NEW ckpt-6   change
mean working-arm MAE     0.0546       0.0536      -2%
```

Like-for-like on epochs (old ckpt-5 = 0.0561 vs new ckpt-6 = 0.0536) it is ~4.5%.
Either way, noise-level. **The diagnosis was right about the mechanism and wrong
about it being the bottleneck.**

---

## 4. Why the model is inaccurate — current best explanation

```
policy moves per step:   2.96 mm
teleoperator moves:      5.79 mm      <- policy moves about HALF
policy absolute error:  44.50 mm      <- yet ends up 8x the motion away
```

Smooth, timid motion plus a large standing error is the signature of **regression
to the mean**. Under MSE, when the model is uncertain which way the arm should go,
the optimal answer is the average of all plausible moves — roughly *nothing*. It is
not confused; it is hedging, and hedging is what MSE rewards.

The project uses `decoder_pred_head: "mlp"` (the default, plain MSE). ICRT ships
two alternatives that do not average, both fully implemented in
`icrt/models/policy/pred_head.py` and **never used**:

```
decoder_pred_head : Literal["mlp", "gmm", "diffusion"]
```

This is the strongest untested hypothesis and the cheapest to test — one flag.

---

## 5. Dataset facts

```
120 episodes, 3 tasks, 59,311 frames @ 15 fps (stride 2 from 30 fps source)
proprio 21-D / action 21-D, verified at every layer:
  hdf5 (T,21) · SequenceDataset (1024,16,21) · checkpoint fc1 (21,21) · decoder fc2 (336,128)=16x21
action tensor is 22 wide: 21 real + eos, values exactly {0.0, 1.0}
```

| task | eps | frames | % of data | working arm | left-arm travel | right-arm travel |
|---|---|---|---|---|---|---|
| `ylw2` | 30 | 29,143 | **49.1%** | right | 0.000 m | 5.752 m |
| `box` | 50 | 16,305 | 27.5% | left | 3.244 m | 0.326 m |
| `ecu` | 39 | 13,357 | 22.5% | left | 4.067 m | 0.301 m |
| `ylw1` | 1 | 506 | 0.9% | right | 0.000 m | 3.041 m |

**Two consequences:**

1. **50.0% of training frames have the left arm frozen at exactly 0.000 m.**
   `ylw2` is 25% of episodes but 49% of frames because those recordings are long.
   **`rebalance_tasks` was on in all six runs and makes this worse, not better**
   — it equalises episode *count* (median = 37), and `ylw2` averages 958 frames
   against ~330, so its frame share goes 49.5% → 58.9% and it is the only task
   sampled with replacement. There is no frame-level balancing flag in ICRT; see
   `training_tips.md` §C1.

2. **There is no genuinely bimanual data.** Where both arms move (`box`, `ecu`),
   the left travels 10–13× further; the right's ~30 cm reads as steadying, not
   manipulation. No loss weighting creates coordination that was never
   demonstrated. A bimanual claim in the paper needs two-arm recordings.

**The lift is constant.** Global range 1.4 mm, max within-episode motion 0.01 mm,
0/120 episodes move more than 1 mm. So the 21st dimension carries no information —
literally 21/21, informationally 20/20 with a constant channel. Harmless for
training (a constant input is a bias; a constant delta target costs ~0 loss) and
it is the *safe* configuration at inference. Worth stating precisely in the paper:
"21-D (20 actuated in this campaign; the torso lift is in the space but held fixed
throughout these recordings)."

**Servo lag — and it is NOT uniform across sessions.** See
`action_target_fix.md`: the leader-follower gap is 21.1 mm on `box`, 38.4 mm on
`ecu` and **0.75 mm** on `ylw2`, a property of controller tuning on the day. That
made the training target mean three different things and is now fixed by taking
the target from the achieved pose one step ahead. The note below describes the
OLD leader-command target.

The commanded action leads the measured state by **2–4 cm** on the
working arm (`|action − observation|` mean 0.0207 m left, 0.0012 m right). The
model is trained to predict this, so it is not unfair scoring — but "exact" is not
the target. A perfect policy reproduces the *commands*. The useful goal is beating
the 0.0065 m null baseline, not zero error.

---

## 6. Runs

| run | scale_action | epochs | ckpts | final train loss | best held-out MAE |
|---|---|---|---|---|---|
| `mt21_gpu` | `null` | stopped at 46/60 | 10 | 0.00966 | **0.0546** (ckpt-15) |
| `mt21_scaled` | v1 (lift std = 1.0) | stopped at 14/25 | 8 | 0.20773 | 0.0536 (ckpt-6) |
| `mt21_v2` | v2 (lift scale 5 mm) | stopped at 1/25 | 1 | 0.44254 | not validated |

All checkpoints preserved: 11 G + 8.3 G + 1.1 G.

**Training loss is not comparable across runs.** Normalised targets have unit
variance, so `mt21_scaled`'s 0.208 and `mt21_gpu`'s 0.0097 measure different
things. Only held-out position MAE compares.

### Held-out error over training — the overfitting curve

```
mt21_gpu:   epoch 5    0.0561
            epoch 15   0.0546   <- best
            epoch 35   0.0700   <- worse, while train loss fell 2.5x
```

Held-out error degraded on 5 of 6 episodes from epoch 15 to 35 while training
loss improved 2.5×. Three independent measures (MAE, IK residual, lift command)
moved the same way on 6 of 6. **114 training episodes against 92.6 M trainable
parameters, with no validation scored during training.** `log.txt` has no
`val_loss` column — add one before the next run.

---

## 7. Not verified — stated plainly

- **`cyclo_py` has never been compiled.** No docker access from this session
  (`docker ps` → permission denied) and no ROS on the host. cyclo_control was built
  2026-09-13 21:37 inside the **ROS 2 Jazzy container**; on the bare host its `.so`
  has 6 unresolved dependencies (pinocchio 4.1.0, coal 3.0.3, urdfdom 4.0, osqp,
  OsqpEigen 0.10.3). Build with:
  `./container.sh bash -lc 'aiworker_icrt/cyclo_bindings/build.sh'`
  All 25 bound symbols were checked by name against the real headers; namespaces
  and paths verified. Expect one or two compile errors regardless.
- **Gazebo cameras** — XML parses and all 7 macro params resolve, but `xacro`
  could not expand them (no ROS on host). Verify with
  `xacro ... | grep -c '<sensor'`. FOVs are datasheet figures, not calibrated;
  take `horizontal_fov` from real `camera_info` if available.
- **The gate's IK uses the wrong solver.** `kinematics.py`'s DLS solver, not
  cyclo_control's QP, which is what runs on the robot. cyclo_control carries
  velocity limits as *hard QP constraints*, so it physically cannot emit the
  2.45× overspeed the gate reports — it would clamp and lag instead. Right symptom,
  wrong mechanism. Its `getCollisionPairDistances()` would also replace the crude
  EEF-distance proximity check.
- **Condition 3 HAS NOW BEEN RUN — and it fails.** Prompting with a different
  task costs the policy 1.0 mm on a 55.6 mm error (1.8%, noise), and prompting
  with the episode about to be replayed does not help either. The prompt does
  reach the model (predictions differ by up to 38.7 mm), so this is a real
  negative, not a plumbing bug — the policy reads the demonstration and extracts
  nothing. Caveat: this checkpoint is already 8x worse than the null baseline, so
  the negative cannot yet be attributed to the method. Full result and method:
  `condition3_result.md`. ORIGINAL NOTE: prompt from task A, observations from
  task B. This is the core in-context claim and remains the central risk (`PLAN.md §4`):
  each task was recorded in its own scene at its own head angle, so the task may be
  identifiable from a single frame and the prompt redundant.
- **No task-completion measure.** Position MAE says "close to what the human did",
  not "picked up the ECU part". That needs a rollout.

---

## 8. Bugs found and fixed

| # | bug | how it was caught |
|---|---|---|
| 1 | `joint_limits` check failed the robot's **home pose** — `arm_l_joint2` has URDF range `[0.0, 3.14]` and rests exactly on its bound; 67 "violations" in 40 steps of *ground truth* | ground-truth control |
| 2 | `ik_converged` condemned **any** learned policy — perturbing a recorded pose by **1 mm** drops the solver 80/80 → 60/80 converged. It tested FK-exactness, which no regression head achieves. Recalibrated to a 25 mm residual and renamed `ik_reachable` | perturbation sweep on ground truth |
| 3 | `scale_action` result was **overwritten** on the next line in `icrt.py`; worked only because the function mutates in place. Rebound explicitly | code read |
| 4 | Setting the lift's std to `1.0` gave it **zero gradient** (every other channel scaled up to 1, lift left at 0.0006). Commanded torso drifted to 7.6–15.4 mm, past the 5 mm limit, on all 6 episodes — where the unnormalised run held 2.2–4.2 mm. Fixed with a physically meaningful 5 mm scale | validated ckpt-6 |
| 5 | My first channel-variance measurement sampled only prediction step 0, understating translation variance ~30× (0.17% vs the true 5.22%). The dataset anchors all 16 steps to the same proprio | disagreement with an independently measured 2 cm gap |
| 6 | `dump_traj.py` wrote `prompt`/no `deviation`; the UI expected `prompt_episode`/`deviation` — would have rendered a blank page. UI now derives deviation from raw series | schema check before publishing |

### Process mistakes worth not repeating

- **`pgrep -f <pattern>` self-matches the ssh command containing that pattern.**
  Reported a dead validator as "running" for over an hour, and a dead training run
  as alive. Three times. Verify via `nvidia-smi --query-compute-apps` or
  `ps | grep scripts/train.py` instead.
- **Piping a long remote job through `| tail` buffers all output until exit**, so
  working jobs look dead. Redirect to a file.
- **`pkill -f "<pattern>"` can match the shell running it** and kill your own
  command. One launch silently did nothing because of this.
- `wandb` **permission denied kills the training process outright.** The original
  run inherited `WANDB_MODE` from an interactive shell, so `train.log` showed no
  wandb lines and gave no hint it was a dependency. Scripts now set
  `WANDB_MODE=disabled` explicitly.
- Overwriting `action_stats.json` in place would have corrupted validation of the
  run that referenced it (lift unscaled by 0.005 instead of 1.0 → 200× off).
  Wrote `action_stats_v2.json` instead.

---

## 9. Next steps, in order of expected impact

0. **DONE — warm start from the DROID-pretrained trunk + CrossMAE encoder.**
   Held-out error 0.0752 -> 0.0308 against an identical from-scratch control
   (59% better at epoch 20), and it stops the overfitting the control shows from
   epoch 4 onward. `warm_start_result.md`.

1. **Switch the action head** — one flag, now demoted (their working model uses
   `mlp` too):
   ```
   --model-cfg.policy-cfg.decoder-pred-head diffusion \
   --model-cfg.policy-cfg.num-train-diffusion-steps 100 \
   --model-cfg.policy-cfg.num-inference-diffusion-steps 10
   ```
   Try `gmm` too — cheaper at inference, often sufficient. ~2 h per run.
2. **More data** — 857 BG2 episodes (`ROBOTIS/Task_0002_OrderPicking_lerobot`,
   19-D, 10 fps, `ffw_bg2_follower.urdf`). Decide fps **before** converting: BG2 is
   natively 10 fps and cannot be upsampled, so converting SG2 at 15 fps and BG2 at
   10 fps would give one model two delta scales for identical motion.
3. **Fix the frame-level task imbalance** — 58.9% of frames teach "left arm
   frozen". `rebalance_tasks` is already on and causes it; cut the long `ylw2`
   episodes or down-weight them via `task_grouping`.
4. **Shrink the model** — 12 layers / 768 dim / 92.6 M trainable is large for 120
   episodes.
5. **Score validation during training** — add `val_loss` to `log.txt` so
   overfitting shows up live instead of in a post-hoc sweep.
6. **Compile `cyclo_py`** so the gate tests the deployment solver.
7. **Re-run condition 3** on the first checkpoint that beats the 0.0065 m
   baseline — it has now been run once (`condition3_result.md`) and fails, but
   on a policy too weak for the result to be attributable.

### What will not help

- **More epochs.** Held-out error bottomed at epoch 15 and got worse.
- **More loss tuning.** Normalisation was the clean version of that experiment and
  it bought 2%.

---

## 10. Operational reference

```bash
# training (from /home/zeux/aiworker_iclr on zeux)
./train_v2.sh                      # WANDB_MODE=disabled + PYTHONPATH already set

# validate a checkpoint (writes JSON verdict + trajectory files)
PYTHONPATH=third_party/icrt:. ./.venv/bin/python -m aiworker_icrt.validate_checkpoint \
  --dataset /dev/shm/icrt_multitask \
  --checkpoint runs/<run>/checkpoint-N.pth \
  --train-yaml runs/<run>/run.yaml \
  --out-dir runs/<run>/validation --max-steps 40

# regenerate action statistics
PYTHONPATH=third_party/icrt:. ./.venv/bin/python make_stats.py \
  /dev/shm/icrt_multitask /dev/shm/icrt_multitask/action_stats_v3.json

# dataset UI (pure stdlib; jobs it launches need the container)
python3 -m aiworker_icrt.ui --port 8770        # http://127.0.0.1:8770

# render an episode as video — catches sign/frame errors numbers miss
python -m aiworker_icrt.visualize --dataset data/icrt_multitask --episode ecu_episode_000000
```

**Checkpoints pickle ICRT's `args` object**, so loading needs
`PYTHONPATH=third_party/icrt` or you get `ModuleNotFoundError: No module named
'icrt'`. Portability under stock torch ≥ 2.6 remains an open item.

---

## 11. The recurring lesson

Eight-plus instances now, all the same shape: **a green signal from a check that
never exercised the failing path.** The training loss fell for 46 epochs while the
model could not place its hand. The safety gate's own `ik_converged` check failed
ground-truth data. A `joint_limits` check flagged the robot standing still.

The discipline that caught each one was the same: **run a control.** Feed the
recorded human demonstration through the identical code path. Anything that fails
on both is measuring the robot, the solver, or your threshold — not the model.

> *Convergence measures agreement with the question asked, never whether it was
> the right question.*
