# ICRT × ROBOTIS AI Worker — Integration Plan

Retarget **ICRT** (In-Context Robot Transformer, Fu et al. 2024, [arXiv:2408.15980](https://arxiv.org/abs/2408.15980))
onto the **ROBOTIS AI Worker FFW-SG2**, training on ~700 multi-task demonstrations
and deploying as a ROS 2 node over Zenoh.

Last updated 2026-09-14. Two Claude sessions contributed: `aiworker-iclr-01`
(kinematics, ICRT patches, converter, policy, ROS node) and `aiworker-iclr-61`
(Docker, compose, Zenoh, launch, tests).


> **Architecture diagram:** [`architecture.html`](architecture.html) — open locally, or view published at https://claude.ai/code/artifact/0174f2ec-4e75-4e1a-82a2-093e875a849f

---

## 0. Requirements, and where each is met

Stated by the user 2026-09-14. Nothing below is inferred.

| # | Requirement | Status | Where |
|---|---|---|---|
| R1 | Convert the LeRobot format into what ICRT supports | **Built + verified** (v3.0 and v2.1) | §8 Phase 3, `aiworker_icrt/convert_lerobot_icrt.py` |
| R2 | The model must work in the AI Worker's modality — code changes required | **Built + verified** (bimanual, 3 cameras, 20-D Cartesian) | §3, §7, §8 Phase 5 |
| R3 | Use cyclo_control for inverse kinematics | **Decided**, not yet integrated | §3, §7a, §8 Phase 8 |
| R4 | 700 episodes live on another device; samples to be shared first | **Waiting on samples** | §8 Phase 2 |
| R5 | The Docker image must build, and the model must run inside Docker | **In flight** | §8 Phases 1 and 1b — 1b is a hard gate |

---

## 1. The problem in one paragraph

ICRT learns from demonstrations *in context*: prepend a few teleop demos to the
transformer's KV cache and it performs the task without finetuning. It was
pretrained on DROID — one Franka arm, two cameras, a 10-D Cartesian end-effector
action space. The AI Worker is a bimanual mobile robot recording 22-D joint-space
actions from three cameras. The integration is a retargeting problem at three
layers (data, model, runtime), plus one genuine research risk (§10).

---

## 2. Target hardware

| Role | Machine | Notes |
|---|---|---|
| Training | x86_64 Blackwell, and this DGX Spark | Spark: NVIDIA GB10, compute_cap **12.1** (sm_121), driver 580.173.02, aarch64, Ubuntu 24.04 |
| Inference | Jetson Thor (sm_110) | SBSA stack, **not** L4T/JetPack-Orin |
| Robot | FFW-SG2 | 25 DOF: 2×7-DOF arms, 2 grippers, 2-DOF head, 1-DOF lift, 3-wheel swerve base |

Base image `nvcr.io/nvidia/pytorch:25.11-py3` is multi-arch and covers all three.
Chosen because its build driver (580.95.05) is the newest at or below the Spark's
580.173.02. sm_121 is not natively compiled in any NGC tag — GB10 runs via PTX JIT
from `compute_120`, which costs first-kernel latency but is otherwise correct.

> **Never `pip install torch` in these images.** The base ships a CUDA build for
> these GPUs; a PyPI wheel silently replaces it with a CPU one. The Dockerfiles
> enforce this with a generated `PIP_CONSTRAINT` file and a hard
> `assert torch.version.cuda` after the pip layers.

---

## 3. Design decisions (locked — do not relitigate)

| Decision | Value | Why |
|---|---|---|
| Action space | **20-D**, arms + grippers only | Per arm `[xyz(3), rot_6d(6), gripper(1)]`, left block then right |
| Head / lift / base | **Not commanded** | Held at episode-start values. Keeps the space invertible and IK simple |
| FK root | `arm_base_link` | EEF pose there depends *only* on that arm's 7 joints — lift/head/base fall out of the kinematics entirely |
| Rotation | `rot_6d=True` | ICRT's convention: first two **rows** of the rotation matrix (`icrt/data/utils.py:13`), not the columns |
| Cameras | 3, order `[cam_head, cam_wrist_left, cam_wrist_right]` | Order must stay stable — each camera gets its own attention-pooling adapter and they are not interchangeable |
| Delta actions | Per-arm, in each arm's own EEF frame | Gripper stays absolute |
| **Inverse kinematics** | **`cyclo_control` QP controller** (R3) | ROBOTIS's `ai_worker_bimanual_movel_controller`: Pinocchio + QP, with self-collision (CBF) and bimanual coordination that our DLS solver does not have. See §7a |
| Forward kinematics | ours (`kinematics.py`) | Conversion only — offline batch numpy, no collision or coordination concerns apply |

### The 22-D recorded layout (both `action` and `observation.state`)

```
 0-6   arm_l_joint1..7        7   gripper_l_joint1
 8-14  arm_r_joint1..7       15   gripper_r_joint1
16-17  head_joint1..2        18   lift_joint
19-21  base linear_x, linear_y, angular_z
```

`physical_ai_server` orders both vectors by the per-group `joint_order` lists and
concatenates in `joint_list` order, so they share this layout exactly.

---

## 4. Repository layout and ownership

```
aiworker_iclr/
├── PLAN.md                       this file
├── assets/ffw_sg2/               FFW-SG2 URDF                        [01]
├── aiworker_icrt/                                                    [01]
│   ├── constants.py              canonical 22-D / 20-D layouts
│   ├── kinematics.py             dual-arm FK + Jacobian + DLS IK
│   ├── convert_lerobot_icrt.py   LeRobot → ICRT dataset converter
│   ├── policy.py                 DualArmICRT (replaces ICRTWrapper)
│   └── ros_node.py               ROS 2 inference node
├── third_party/icrt/             vendored ICRT, patched               [01]
├── tests/                        verification scripts + measurements  [01→61]
├── docker/                                                           [61]
├── docker-compose.yml            zenohd + icrt services               [61]
├── container.sh                  build/start/enter/status             [61]
├── launch/icrt_policy.launch.py                                      [61]
└── config/zenoh/README.md                                            [61]
```

**Ownership is strict.** `docker/`, `docker-compose.yml`, `container.sh`,
`launch/`, `config/` belong to `-61`. Everything else to `-01`. This exists
because both sessions clobbered each other's files on 2026-09-14 — twice, in one
night, in the same directory that had no owner.

> **Action: `git init` the project root.** It is not a repo. The clobbered files
> were unrecoverable. `third_party/icrt` *is* a repo, which is the only reason
> the ICRT patches survived.

---

## 5. Current status

### Verified by execution

| Component | Evidence |
|---|---|
| `kinematics.py` FK/Jacobian | analytic vs numerical Jacobian max diff **3.83e-07** |
| `kinematics.py` IK, warm-started | **0/500** non-converged per arm, mean **1.2–1.5 iters**, error **<100 µm**, ~1700 Hz/arm vs a 15 Hz budget |
| `kinematics.py` IK, cold seeds | 119/120 converged; worst residual 0.12 mm, all near-singular (σ_min 0.0009–0.09 vs median 0.082) |
| Motion continuity | max single-step joint jump 0.031 rad = 0.47 rad/s at 15 Hz (limit 4.8) |
| Dual-arm `convert_delta_action` | round-trip **1.11e-15**, no cross-arm bleed, single-arm path unchanged |
| `convert_lerobot_icrt.py` | 24 synthetic episodes → hdf5 + 4 sidecars, correct shapes |
| **Full ICRT `SequenceDataset` load** | `len=1247`; `observation (32,3,3,224,224)`, `proprio (32,16,20)`, `action (32,16,21)`, masks 0/1, no NaNs — **ALL PASS** |
| **LeRobot v3.0 conversion** | 24 episodes; camera features auto-discovered; **video alignment frame-exact** vs an independent full decode of the concatenated mp4 |
| **Bimanual model construction + forward** | `num_arms=2` → 20/21; 3-camera padding == 0; finite loss at widths 192 and 768 (verified by `-61`) |
| **Full pytest suite** | **27 passed, 0 skipped** on real GPU torch (2.14.0+cu130, GB10), dataset tier included |

### Written but NOT verified

- `policy.py`, `ros_node.py` — syntax-clean only. Never imported against real
  torch/ROS, never run against hardware.
- All Docker images — **never built**.
- Launch file, Zenoh config — never run.

### Not started

- Training run (§8). Offline eval. Anything on the physical robot.

---

## 6. Blockers

1. ~~**Docker is inaccessible.**~~ **RESOLVED.** The earlier diagnosis ("the
   user is not in the docker group") was **factually wrong**. `getent group
   docker` shows `docker:x:988:dexstro` — the membership exists. It is absent
   from `groups`/`id -nG` in our shells only because those process credentials
   predate the grant. No `usermod` was needed; `sg docker -c '<cmd>'` picks the
   group up immediately, and a fresh login clears it permanently.

   **GPU access is via CDI, not the nvidia runtime.** `docker info` registers
   only `runc` / `io.containerd.runc.v2` — there is **no `nvidia` runtime on this
   machine**. nvidia-container-toolkit 1.20.0 is installed with
   `/var/run/cdi/nvidia.yaml`. Verified working:

   ```
   docker run --rm --entrypoint /usr/bin/nvidia-smi --gpus all <img>                -> NVIDIA GB10, 12.1
   docker run --rm --entrypoint /usr/bin/nvidia-smi --device nvidia.com/gpu=all <img> -> NVIDIA GB10, 12.1
   ```

   `--gpus all` works (Docker 29 resolves it through CDI). Anything specifying
   `--runtime nvidia` will fail here.

   > **A green build does not prove the GPU works.** Live counterexample on this
   > box: `robotis/cyclo-intelligence:1.4.0` reports `torch.version.cuda == 12.8`
   > — and `torch.cuda.is_available() == False`, while nvidia-smi works in that
   > same container. The Dockerfile's `assert torch.version.cuda` cannot catch
   > this (build has no GPU). The real gate is `torch.cuda.is_available()` at
   > runtime, which `container.sh status` checks. Treat *that* as Phase 1's gate.

2. **The 700 episodes are not on this machine.** Confirmed by the user to be
   **LeRobot v3.0** (cyclo_intelligence layout). The converter now reads v3.0
   and v2.1, auto-detected from `codebase_version`, and is validated on
   synthetic data of both.
3. **Project root is not a git repo.** See §4.

---

## 7. Changes made to vendored ICRT (R2)

ICRT ships as a single-arm, two-camera, 10-D Cartesian model. Making it speak the
AI Worker's modality — **two arms, three cameras, 20-D** — required changing its
code, not just its config. Every change is backward compatible: `num_arms=1`
reproduces upstream exactly, which is what lets us keep pulling from upstream.


`third_party/icrt` — 5 files. `git diff` in that directory is the authoritative
record. All are backward compatible: `num_arms=1` reproduces upstream exactly.

| File | Change |
|---|---|
| `util/args.py` | `SharedConfig.num_arms: int = 1` |
| `util/model_constructor.py` | `proprio_dim = num_arms*10`, `action_dim = num_arms*10 + 1` (eos only on the `rot_6d` path, preserving upstream's 10/11 and 8/8) |
| `data/utils.py` | `convert_delta_action` / `convert_abs_action` generalised to per-arm 10-D blocks |
| `models/policy/icrt.py` | replaced the deprecated `torch.set_default_tensor_type(torch.cuda.HalfTensor)` (deprecated since torch 2.1; warns on both 2.14 and NGC's 2.10.0a0, and would hard-fail when removed) with `set_default_dtype(float16)` + `with torch.device("cuda")`. Verified under `-W error::UserWarning` |
| `data/dataset.py` | ×4: new `skip_rot_conversion` flag; bypass in `helper_load_proprio`; **guard `helper_load_action`**, which called `euler_to_rot_6d(ret[:, 3:])` unconditionally and throws on 17 dims; `get_key_from_demo` passes 4-D image arrays through instead of byte-reshaping to a hardcoded 180×320 and raising on unknown episode names |

Deliberately **not** installed: `kinpy` (nothing imports it — `kinematics.py` owns
the URDF parsing and Jacobian), `flash-attn` (`handle_flash_attn` at
`models/backbones/utils.py:11` has **zero call sites**, and v2.2.3 won't compile
for sm_110/sm_121).

---

## 7a. The ROBOTIS `cyclo` stack — what to adopt

Discovered 2026-09-14 after the user pointed at `cyclo_control`. Two repos, both
newer than `physical_ai_tools`, and they overlap our work substantially.

### `cyclo_intelligence` (already checked out at `~/robotis/cyclo_intelligence`)

A full Physical AI platform: record → convert → train → infer → execute.
Relevant pieces:

| Path | What it is | Bearing on us |
|---|---|---|
| `cyclo_data/converter/to_lerobot_v30.py` | writes the **v3.0** datasets | **Authoritative schema for the user's 700 episodes.** Our converter was written from it |
| `cyclo_brain/policy/{common,lerobot,groot}` | **pluggable policy engines** — `common/runtime/engine.py` is the base, lerobot and groot are backends | ICRT could become a third backend rather than a standalone ROS node |
| `cyclo_brain/sdk/zenoh_ros2_sdk` | a Zenoh ROS 2 SDK | overlaps `-61`'s hand-built Zenoh config |
| `cyclo_brain/sdk/action_chunk_processing` + `interfaces/msg/ActionChunk.msg` | action-chunk transport | ICRT predicts 16-step chunks natively — a direct fit |

### `cyclo_control` (GitHub, not checked out)

`ai_worker_bimanual_movel_controller` is a **bimanual Cartesian controller built
on Pinocchio + a QP solver**. Compared to `aiworker_icrt/kinematics.py`:

| Capability | ours (DLS) | cyclo (QP) |
|---|---|---|
| Joint limits | clamp + zeroed outward steps | hard constraints with slack |
| Singularities | damping only | explicit slack variable |
| **Self-collision** | **none** | CBF constraints (`cbf_alpha`, `safe_distance`), SRDF pairs |
| **Bimanual coordination** | **none** | `setRigidGraspPoseConstraint` — holds a fixed relative transform between hands |

### Decision (R3 — settled by the user)

**`cyclo_control` does inverse kinematics at execution time. Ours does forward
kinematics for dataset conversion.** Both directions are needed; they are
different problems and get different tools.

- *Execution:* the QP controller is strictly better where it matters. Two arms
  in a shared workspace can collide, and our IK has no notion of that — it will
  happily solve a pose that drives one arm through the other. Note its interface
  is velocity-level (`setDesiredTaskVel` takes a twist, `getOptJointVel` returns
  joint velocities), so the node converts ICRT's predicted pose into a twist via
  pose error × gain rather than calling a pose-level IK.
- *Conversion:* keep `kinematics.py`. It is the **FK** direction only, offline,
  batch, pure numpy at ~217k steps/s, and verified to 3.8e-07 against numerical
  differentiation. Pulling a C++/Pinocchio/colcon dependency into an offline
  batch job buys nothing. Worth cross-validating our FK against Pinocchio once
  cyclo is available — an independent check of the same quantity.
- *Platform:* whether ICRT becomes a `cyclo_brain` backend instead of a
  standalone ROS node is a **strategic decision for the user**, not a technical
  one. It would give us the recording UI, dataset management, Zenoh SDK and
  orchestration for free, at the cost of restructuring `ros_node.py` and
  `policy.py` around their engine interface. **Not** a blocker for Phases 0-5.

> Cost note: `cyclo_control` is C++ in the `ros2_control` stack and needs
> Pinocchio + an SRDF. It is a real integration, not a drop-in import.

---

## 8. Phased plan

Each phase has a **gate** — do not proceed until it passes.

### Phase 0 — Unblock
```bash
sudo usermod -aG docker $USER && newgrp docker
cd /home/dexstro/development/aiworker_iclr && git init && git add -A && git commit -m "initial"
```
**Gate:** `docker info` succeeds; `git log` shows one commit.

### Phase 1 — Build the image (R5)
```bash
./container.sh build --data      # --data adds lerobot/datasets/pyarrow
./container.sh start
./container.sh status
```
**Gate:** `status` reports torch with a non-empty `cuda` version, `torch.cuda.is_available()` True, `icrt importable`. If the build's `assert torch.version.cuda` fires, **do not relax `PIP_CONSTRAINT`** — pin the offending package properly.


### Phase 1b — The model must RUN in Docker (R5, hard gate)

Building is not running. Both must be demonstrated, and they fail differently.

```bash
ICRT_DATA_DIR=$PWD/data ./container.sh start
./container.sh status                       # reports torch.cuda.is_available()

./container.sh exec "ICRT_DATASET_DIR=/data/icrt_synth_v30 \
    PYTHONPATH=/workspace:/workspace/third_party/icrt \
    python -m pytest /workspace/tests/ -q"
```

**Gate — all four, no partial credit:**

1. `torch.cuda.is_available()` is **True** inside the container.
2. `torch.cuda.get_arch_list()` contains `sm_120`. Measured natively on this box:
   `['sm_80','sm_90','sm_100','sm_110','sm_120']`, device capability `(12, 1)`.
   **`sm_121` is absent and must not be required** — GB10 reaches sm_121 by PTX
   JIT from `compute_120`, so a guard demanding `sm_121` fails a working install.
3. `import icrt` succeeds.
4. The suite reports **27 passed, 0 skipped**.

> **A green build proves nothing about the GPU.** Live counterexample on this
> box: `robotis/cyclo-intelligence:1.4.0` reports `torch.version.cuda == 12.8`
> while `torch.cuda.is_available()` is `False`, with `nvidia-smi` working in that
> same container. The Dockerfile's build-time assert cannot catch this — `docker
> build` has no GPU. Only the runtime check above closes R5.

> **GPU passthrough here is CDI, not the nvidia runtime.** `docker info`
> registers only `runc`; there is no `nvidia` runtime on this machine.
> `--gpus all` works (Docker 29 resolves it via CDI); `--runtime nvidia` fails.

`data/icrt_synth_v30/` is a 45 MB **synthetic** 24-episode set staged for exactly
this gate. It proves the plumbing, not that real recordings work. Delete it once
Phase 3 has run.

### Phase 2 — Sample episodes (R4)

The 700 episodes live on another device. The user will share **samples first**.
Do not wait for the full set to start — the samples are what de-risk everything.

```bash
# 1. What are we actually looking at?
python - <<'EOF'
import json; from pathlib import Path
r = Path('<samples>')
i = json.loads((r/'meta/info.json').read_text())
print(i.get('codebase_version'), i.get('robot_type'), i.get('fps'))
for k, v in i['features'].items():
    print(f"  {k:48} {v.get('dtype'):8} {v.get('shape')}")
EOF

# 2. Convert a couple of episodes
./container.sh exec "python -m aiworker_icrt.convert_lerobot_icrt \
    --lerobot-root /data/<samples> --out-dir /data/icrt/sample --limit 2"

# 3. Load them
./container.sh exec "ICRT_DATASET_DIR=/data/icrt/sample \
    python /workspace/tests/check_dataset_roundtrip.py /data/icrt/sample 8"
```

**Gate:** `observation.state` and `action` are both **22-D**; three camera
features resolve; every sample episode carries a non-empty task string.

**What to expect to be wrong**, since the converter has only ever seen data this
project generated:

- *Camera feature names.* We match on the last dotted segment, so
  `observation.images.rgb.cam_head` and `observation.images.cam_head` both work.
  If the real names differ (`head`, `cam_1`, `zed_left`), the converter fails
  loudly with the list it found — fix with `--camera-map cam_head=<actual>`.
- *A different state width.* If it is not 22, the recording used a different
  `joint_order` and §3's layout must be re-derived from that dataset's
  `info.json`, not assumed.
- *Missing task strings.* Fatal for in-context learning, and silent if unchecked
  — the converter counts and reports them.



### Phase 3 — Convert the full dataset (R1)
```bash
./container.sh exec "python -m aiworker_icrt.convert_lerobot_icrt \
    --lerobot-root /data/<dataset> --out-dir /data/icrt/ffw_sg2_mt \
    --prompt-dir /data/icrt/prompts --limit 5"      # smoke test first
```
Then rerun without `--limit`.

**Gate:** no `SKIPPED` entries; the task histogram matches your recording log; **zero** "episodes have no task string" warnings. Task grouping is what makes ICRT's prompt masking meaningful — without it you are training plain behaviour cloning with extra steps.

**Storage:** ~518 KB per timestep across three cameras at 180×320. 700 episodes × ~500 steps ≈ **180 GB**. Put it on NVMe or `/dev/shm` — ICRT's own `TRAIN.md` names disk read speed as the training bottleneck.

> **`/data` is a bind mount and defaults to `./data` in the repo, which is the wrong place for 180 GB.**
> Point it at the NVMe before Phase 2 or the converter will fill the project directory:
>
> ```bash
> ICRT_DATA_DIR=/mnt/nvme/icrt ./container.sh start
> ```
>
> Export it in the same shell for every later `container.sh` call, or the mount
> silently reverts to `./data` on the next `start`. Never a network mount.

### Phase 4 — Data gate
```bash
./container.sh exec "python tests/check_dataset_roundtrip.py /data/icrt/ffw_sg2_mt"
```
**Gate:** `len(dataset) > 0` and shapes are `proprio (T,16,20)` / `action (T,16,21)`.

> `len(dataset)==0` means the `task_barrier` window doesn't fit. `usable_indices`
> only admits an episode start when `seq_length <= task_length`, where
> `task_length` is the **summed frames of that task's episodes in the train
> split**. With 700 episodes across several tasks this is comfortable, but if you
> ever filter to a small subset, lower `--shared-cfg.seq-length` first. This cost
> us an hour of debugging on 6 synthetic episodes.

### Phase 5 — Train (R2)

Baseline, from scratch, bimanual:
```bash
torchrun --nproc_per_node=<N> --master_port=2450 third_party/icrt/scripts/train.py \
  --dataset-cfg.dataset-json /data/icrt/ffw_sg2_mt/dataset_config.json \
  --logging-cfg.output-dir /data/runs --logging-cfg.log-name ffw_sg2_dualarm \
  --shared-cfg.num-arms 2 \
  --shared-cfg.num-cameras 3 \
  --shared-cfg.rot-6d True \
  --shared-cfg.seq-length 256 \
  --shared-cfg.num-pred-steps 16 \
  --shared-cfg.batch-size 2 \
  --shared-cfg.save-every 5 \
  --model-cfg.policy-cfg.scratch-llama-config third_party/icrt/config/model_config/custom_transformer.json \
  --model-cfg.policy-cfg.phase pretrain \
  --model-cfg.policy-cfg.no-prompt-loss \
  --model-cfg.vision-encoder-cfg.vision-encoder vit_base_patch16_224.mae \
  --dataset-cfg.non-overlapping 32 \
  --dataset-cfg.num-repeat-traj 2 \
  --dataset-cfg.shuffle-repeat-traj \
  --dataset-cfg.rebalance-tasks \
  --dataset-cfg.proprio-noise 0.0 \
  --optimizer-cfg.lr 5e-4 --optimizer-cfg.warmup-epochs 1.25 \
  --trainer-cfg.epochs 125 --trainer-cfg.accum-iter 4
```

Parameter notes:

- **`--no-prompt-loss` is the whole point.** It restricts the loss to the target
  portion after a randomly chosen episode boundary, so the model is explicitly
  optimised to *use* the preceding demos. Without it you get behaviour cloning.
- **`seq-length 256`** not ICRT's 512: 256 (s,a) pairs = 512 tokens. Sized against
  both the per-task frame budget above and the KV cache — at 15 Hz with 2 tokens
  per step, the cache covers prompt + rollout, and overflow triggers a doubling
  reallocation mid-episode (`llama.py:203`).
- **`proprio-noise 0.0`**: upstream's 0.005 default was tuned for a 10-D vector
  where dims 3: are one rotation. On our 20-D vector it perturbs two rot_6d
  blocks and both grippers. Revisit only as a deliberate augmentation experiment.
- **`rebalance-tasks`** matters if the 700 episodes are unevenly distributed.
  Check the Phase 2 histogram.

**Gate:** `action_loss` decreasing; a held-out episode's predicted actions track
ground truth when replayed with teacher forcing.

### Phase 6 — Offline evaluation *before* touching hardware
Replay held-out episodes through `DualArmICRT.step()`, prompted with a *different*
episode of the same task. Compare predicted vs recorded EEF trajectories.

**Gate:** prompting with same-task demos beats prompting with a different task's
demos. **This is the paper's central claim.** If it fails, no amount of deployment
work helps — go back to §10.


### Phase 7 — Integrate `cyclo_control` for IK (R3)

Only needed for execution on hardware; training does not depend on it.

1. Clone and build `cyclo_control` in the image (needs Pinocchio + an SRDF).
2. **Cross-validate our FK against theirs.** Feed identical joint vectors to
   `kinematics.py` and to `KinematicsSolver`, and compare EEF poses. Ours is
   verified to 3.83e-07 against numerical differentiation, so a disagreement
   means a frame or unit convention differs — find out here, not on the robot.
3. Replace the IK call in `ros_node.py`. Note the interface is **velocity-level**:
   `setDesiredTaskVel` takes a twist and `getOptJointVel` returns joint
   velocities, so convert ICRT's predicted pose to a twist via pose-error × gain.
4. Enable self-collision (`setConstraintLinks`, `cbf_alpha`, `safe_distance`) and,
   if a task needs two-handed carrying, `setRigidGraspPoseConstraint`.

**Gate:** replay a converted episode's Cartesian actions through the QP
controller in simulation and confirm the arms track without a self-collision
flag. **This is the step that stops one arm being driven through the other** —
our DLS solver has no concept of that and will happily solve such a pose.

### Phase 8 — Deploy

> **Rebuild the image before deploying.** The Dockerfile `COPY`s
> `third_party/icrt` into `/opt/icrt` at build time, and `docker-compose.yml`
> bind-mounts the host copy over it at runtime. On this dev box the mount always
> wins, so the baked copy can drift from the host indefinitely without anyone
> noticing. **On the robot there is no repo to bind-mount** — the Thor runs the
> image standalone, and it will execute whatever ICRT was baked in. Any image
> built before the last `third_party/icrt` change ships stale patches, silently.
> Rebuild, and verify inside the container *without* the mount that
> `icrt/models/policy/icrt.py` contains the `set_default_dtype` form.

```bash
# one host only:
./container.sh start --router
# every other host:
ZENOH_ROUTER_ENDPOINT=tcp/<router-ip>:7447 ./container.sh start

ros2 launch launch/icrt_policy.launch.py \
    checkpoint:=/ckpt/... train_yaml:=/ckpt/...yaml prompt_npz:=/data/icrt/prompts/episode_000003.npz
ros2 service call /icrt_policy/start std_srvs/srv/Trigger
```
The node **idles until `~/start`** — ICRT's KV cache must be prompted with a demo
before the first action means anything. `autostart` defaults false; keep it there.

**Gate:** e-stop within reach; first run with the arms clear of obstacles.

---

## 9. How the runtime fits together

```
3× CompressedImage + /joint_states
        │  ApproximateTimeSynchronizer (slop 0.05s)
        ▼
  ros_node.py ── timer @15 Hz (NOT the callback: ICRT's KV cache
        │                      advances 2 tokens/step and needs regular cadence)
        ▼
  FK: 22-D joints ──────────► 20-D Cartesian proprio
        ▼
  ICRT.get_action_eval() ──► 20-D Cartesian action (temporal ensembling)
        ▼
  IK (warm-started from previous solution) ──► 22-D joint target
        ▼
  JointTrajectory → /leader/joint_trajectory_command_broadcaster_{left,right}/joint_trajectory
```

**Publishing to the *leader* topics is deliberate.** `ffw_sg2_follower_ai.launch.py`
remaps those onto `arm_{l,r}_controller` at spawn time, so the policy is a drop-in
replacement for the physical leader — the robot is brought up exactly as it is for
teleoperation, nothing relaunched, no controller reconfigured. Launch
`ffw_sg2_follower_ai.launch.py`, **not** `ffw_sg2_ai.launch.py`, which also starts
the LG2 leader and would fight the policy for the same topic.

A single trajectory point one control period out lets the 100 Hz
`JointTrajectoryController` interpolate, so a 15 Hz policy still produces smooth
motion on the DYNAMIXEL bus.

---

## 10. Risks

### R1 — DROID pretraining may not transfer (highest)
ICRT's published checkpoints are single-arm, 10-D, two-camera. Ours is dual-arm,
20-D, three-camera. The `icrt_*` adapters, action encoder and decoder all change
shape; only the LLaMA backbone (768-dim, 12 layers) is shape-compatible. Even
then DROID is single-arm, so it is at best a warm start for half the action space.

*Mitigation:* Phase 4 trains from scratch and does not depend on transfer. Treat
partial backbone loading as a separate experiment with its own baseline, and
verify what actually loads rather than trusting `strict=False` to be meaningful.

### R2 — 700 episodes may be too few for in-context ability to emerge
In-context learning emerged in ICRT from DROID-scale pretraining plus ICRT-MT
finetuning. 700 episodes is modest.

*Mitigation:* Phase 5's gate detects this before hardware time is spent. If
same-task prompting doesn't beat cross-task prompting, the honest options are
more data, fewer tasks, or reporting the negative result.

### R3 — Singularities
IK is DLS: near-singular targets get sub-millimetre residuals rather than exact
tracking, by design. 0.75% of *cold* random targets; **0/500** warm-started.
`ros_node.py` logs non-convergence and clamps per-step joint motion
(`max_joint_step`, default 0.25 rad).

### R4 — Nothing has been built or executed
No image built, no ROS node run, no hardware touched. Every number in §5 comes
from numpy/scipy/torch running natively outside a container. **Do not describe
this stack as working.**

---

## 11. Open questions for the user

1. Where are the 700 episodes, and are they LeRobot v2.1 from `physical_ai_server`?
2. Distribution across tasks — how many tasks, how many episodes each? Drives
   `rebalance-tasks` and `seq-length`.
3. Jetson Thor's JetPack version (for the inference image base).
4. Which vision encoder? Default is `vit_base_patch16_224.mae`; ICRT's own runs
   used a custom cross-MAE checkpoint passed by path.
5. Should `aiworker_icrt` become a proper `ament_python` package? Currently a
   plain package on `PYTHONPATH`, launched via `ExecuteProcess`.

---

## 12. Sources

- ICRT: [arXiv:2408.15980](https://arxiv.org/abs/2408.15980) · [Max-Fu/icrt](https://github.com/Max-Fu/icrt) (vendored at `e0c9588`, Apache 2.0)
- AI Worker: [ROBOTIS-GIT/ai_worker](https://github.com/ROBOTIS-GIT/ai_worker) · [physical_ai_tools](https://github.com/ROBOTIS-GIT/physical_ai_tools) · [docs](https://ai.robotis.com/ai_worker/hardware_ai_worker)
