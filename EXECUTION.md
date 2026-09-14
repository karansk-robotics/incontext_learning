# ICRT × AI Worker — Execution Record

What was built, what was changed, what was measured, and what bit us.
Companion to `PLAN.md` (forward-looking). This file is the backward-looking record.

Session of 2026-09-14. Three Claude sessions: **-01** (model/data — this record),
**-61** (docker, compose, zenoh, launch, tests), **-91** (build execution, reporting).


> **Architecture diagram:** [`architecture.html`](architecture.html) — open locally, or view published at https://claude.ai/code/artifact/0174f2ec-4e75-4e1a-82a2-093e875a849f

---

## 1. Where it stands

| | Status |
|---|---|
| LeRobot → ICRT conversion | **Working on real data.** v2.1 and v3.0, auto-detected |
| Bimanual 20-D Cartesian retargeting | **Working**, FK verified to 3.83e-07 |
| Model adapted to AI Worker modality | **Working** — 2 arms, 3 cameras, 20-D |
| Training | **Learning on real ECU data** — loss 0.3550 → 0.2399 in one epoch at seq_length 512 |
| Container | **Built and verified** — GPU True, cv2 4.14.0, ros2 responding |
| Test suite | **28 passed** natively |
| Deployment (cyclo_control IK, robot) | Not started |

**Nothing has run on the physical robot.**

---

## 2. What was built (`aiworker_icrt/`)

| Module | Purpose |
|---|---|
| `constants.py` | The two canonical layouts: 22-D joint (as recorded) and 20-D Cartesian (as fed to ICRT) |
| `kinematics.py` | Self-contained URDF parser, FK, analytic Jacobian, damped-least-squares IK with restarts |
| `convert_lerobot_icrt.py` | LeRobot (v2.1 + v3.0) → ICRT hdf5 + sidecars, with FK retargeting |
| `policy.py` | `DualArmICRT` — replaces ICRT's single-arm `ICRTWrapper` |
| `ros_node.py` | ROS 2 inference node, publishes as a drop-in for the leader |
| `visualize.py` | Renders converted episodes to video + trajectory plots |

### The 20-D action space

Per arm `[xyz(3), rot_6d(6), gripper(1)]`, left block then right.
**Arms and grippers only** — head, lift and mobile base are not commanded.

### Reference frame: `base_link`

EEF poses are in `base_link` (between the wheels), not `arm_base_link` (shoulders).
This puts `lift_joint` inside each chain, so a chain has **8** actuated joints.

- **Forward** (conversion): reads the recorded lift, includes it.
- **Inverse** (execution): holds the lift fixed, solves the 7 arm joints by
  slicing the lift column out of the Jacobian.

Holding it is not cosmetic. With the lift free, an 8-DOF chain solving a 6-DOF
pose can answer "raise the torso 5 cm" — which the policy cannot execute.
Verified it does not: asked for +10 cm of EEF height, lift stayed at −0.250 and
the arm moved 0.203 rad.

---

## 3. Changes to vendored ICRT (7 files, 24 hunks)

`third_party/icrt` is a git checkout; **`git diff` there is authoritative** — an
earlier draft of this table said 8 files and undercounted `dataset.py`; the diff
said 7 files / 24 hunks. Read the diff, not this summary. All
changes are backward compatible — `num_arms=1` reproduces upstream exactly.

| File | Change | Why |
|---|---|---|
| `util/args.py` | `SharedConfig.num_arms`; `DatasetConfig.minimum_length` / `maximum_length` | Arm count was implicit; length filters were unreachable class attributes |
| `util/model_constructor.py` | `proprio_dim = num_arms*10`, `action_dim = num_arms*10 + 1` | eos only on the `rot_6d` path, preserving upstream's 10/11 and 8/8 |
| `data/utils.py` | `convert_delta_action` / `convert_abs_action` generalised to per-arm 10-D blocks | Upstream assumed one arm; each arm's delta must be in its own EEF frame |
| `data/dataset.py` | ×7, incl. **the length filter now raises with the actual episode range and its own remediation** ("raise --dataset-cfg.maximum-length, or downsample when converting") — turning the silent `len(dataset)==0` into a message that names its fix. Also: `skip_rot_conversion` flag; proprio bypass; **action guard**; 4-D image passthrough; length filter honoured + loud failure | See §4 — three of these were silent-failure bugs |
| `models/policy/icrt.py` | `set_default_tensor_type` → `set_default_dtype` + `torch.device("cuda")` | Deprecated since torch 2.1; would hard-fail when removed, and the failure would read as a container problem |
| `util/misc.py` | `torch.load(..., weights_only=False)` ×2 | torch 2.6 default flip breaks every checkpoint load — see §4.4c |
| `models/policy/icrt.py` (2nd, 3rd) | per-arm Gram-Schmidt at inference; `torch.load` guard | upstream truncated a 20-D action to 10, silently dropping the right arm — see §4.4b |
| `models/backbones/encoders.py` | `view` → `reshape`; `torch.load` guard | The camera axis makes the tensor non-contiguous; only manifests with >1 camera |

**Deliberately not installed:** `kinpy` (nothing imports it), `flash-attn`
(`handle_flash_attn` has zero call sites and v2.2.3 won't compile for sm_110/121).

---

## 4. Findings that cost real time

These are the ones worth carrying forward. Most were **silent** failures.

### 4.1 `maximum_length = 450` silently discards long episodes
A hidden class attribute tuned for ~15 fps data. The ECU recordings are 30 fps, so
episodes are 691–802 frames — **all 39 were dropped**, yielding an empty dataset
with no error. Now configurable, and raises a message naming the actual lengths.

### 4.2 `min_demos = 4` per task
A task with fewer than 4 episodes does not survive the train/val split and vanishes
from `verb_to_episode` entirely. A 3-episode smoke test produces `len(dataset) == 0`
that looks exactly like a converter bug.

### 4.3 `--no-prompt-loss` needs `seq_length` > episode length
Measured on the ECU data (episodes 280–412 frames after subsampling):

| seq_length | windows containing an episode boundary |
|---|---|
| 32 | **10%** |
| 128 | 48% |
| 256 | 78% |
| **512** | **95%** |
| 768 | 98% |

With `--no-prompt-loss`, loss is computed only *after* a boundary. At
`seq_length=32` the loss is exactly `0.0000` on ~90% of iterations — training
appears to run, converges to nothing, and reports no error. **Use seq_length ≥ 512
for these episode lengths.**

### 4.4 The gripper sense was inverted in our labelling
Caught by `visualize.py`, not by any assertion. The plot showed the gripper
"opening" during transport and "closing" at the drop — nonsense for pick-and-place.
Wrist-camera frames settled it: at normalised 0.09 the fingers are wide apart, at
0.72 they are clamped on the part.

**RH-P12-RN: raw joint 0 = OPEN, 1.1 = CLOSED.** We normalise preserving that
sense, so **0 = open, 1 = closed**. The mapping is monotonic and round-trips, so
the model was never affected — but the sense matters wherever a threshold is
applied, e.g. ICRT's `binary_gripper` (`> 0.5`).

### 4.4b The right arm was silently discarded at inference
**The most serious bug found, and only running `evaluate.py` exposed it.**

`icrt.py`'s rot_6d re-orthogonalisation rebuilt the action as
`cat([xyz, b1, b2, gripper])` — a single 10-D arm. Given a 20-D bimanual action
it **truncated to 10 and dropped the right arm entirely**, with no error. The
symptom surfaced two functions away, as `convert_abs_action` receiving a 10-wide
action against a 20-wide proprio.

Nothing caught it earlier because training never calls this path — it lives in
`forward_inference`, so the full test suite, the forward/backward checks and a
complete training run were all green while inference was structurally broken.
Now applies Gram-Schmidt per arm block and asserts the width divides by 10.

### 4.4c `torch.load` fails on every checkpoint under torch >= 2.6
PyTorch 2.6 flipped `weights_only` to default `True`. ICRT checkpoints embed an
`ExperimentConfig` dataclass, so the safe unpickler rejects them: inference,
resume and deployment all fail. Four call sites fixed with
`weights_only=False` — these are our own training artefacts.

### 4.5 Two IK bugs, found by testing
- **Nullspace leak.** The rest-posture pull used the *damped* pseudo-inverse, so
  it leaked into task space and parked the solver at ~1 mm. Fixed by fading the
  secondary objective as task error shrinks.
- **Joint-limit deadlock.** Naive `np.clip` let the solver jam against a bound;
  worst-case error **23.8 mm**. Fixed by zeroing outward-pushing `dq` components.
  Worst case now **0.11 mm**.

### 4.6 Recording width varies by robot_type
ROBOTIS's public datasets (`ffw_bg2_rev4_custom`, `ffw_arm_only`) record **16-D**,
not 22-D — arms and grippers only. The arm/gripper block is a stable *prefix* of
the full layout, so both are supported. The 22-D assumption was not universal.

### 4.7 Environment traps (from -61 / -91)
- **No `nvidia` runtime on this box** — GPU is via CDI. `--gpus all` works;
  `--runtime nvidia` fails.
- **`torch.version.cuda` is not sufficient.** `robotis/cyclo-intelligence:1.4.0`
  reports cuda 12.8 but is compiled **sm_87 only (Orin)** and cannot drive GB10.
- **`sm_121` is absent from the arch list yet CUDA works** — GB10 JITs from
  `compute_120`. A guard requiring `sm_121` would reject a working install.
- **`get_arch_list()` gating is version-dependent** — empty without a GPU on
  torch 2.10.0a0, populated on 2.14.0. Build-time guards must use
  `_cuda_getArchFlags()`.
- **cv_bridge segfaults** in this image (C++ extension built against NumPy 1.x,
  base ships 2.1.0). Our code never uses it — `ros_node.py` does
  `np.frombuffer` + `cv2.imdecode` directly.
- **`token.txt` was in the repo root at mode 664 with no `.gitignore`**, and the
  repo root bind-mounts into every container. Now 600 and ignored.

---

## 5. Verification evidence

All measured natively (torch 2.14.0+cu130, GB10, sm_121) unless noted.

**Kinematics**
- analytic vs numerical Jacobian: **3.83e-07**
- warm-started tracking (the regime that runs): **0/500 non-converged**, mean
  **44.9 µm**, max **98.9 µm**, ~1700 Hz/arm against a 15 Hz budget
- cold random seeds: 2/300 miss, all near-singular; converged cases mean 0.05 mm
- max single-step joint jump 0.031 rad = 0.47 rad/s at 15 Hz (limit 4.8)

**Dual-arm delta actions**
- round trip **1.11e-15**, no cross-arm bleed, single-arm path unchanged

**Conversion — real ECU data**
- 39 episodes, 13,357 frames, 30→15 fps, 280–412 frames/episode, 7.2 GB
- v3.0 video alignment verified **frame-exact** against an independent full decode

**Model — real ECU data**
- dataset len **12,565**; `observation (32,3,3,224,224)`, `proprio (32,16,20)`,
  `action (32,16,21)`
- LLaMA architecture **from scratch** (768-dim, 12 layers): 92.6M trainable / 178.4M
- forward `action_loss=0.3417`; backward 162 grad tensors, mean |grad| 2.8e-04
- `scripts/train.py` at **seq_length 32**: 392 it, 0.19 s/it — but loss `0.0000`
  on ~90% of iterations. Runs, reports success, learns nothing. See §4.3.
- `scripts/train.py` at **seq_length 512**: 47 it, **2.32 s/it**, 9.8 GB peak,
  loss **0.3550 → 0.2399** over one epoch. This is the real number.

**Container** (reported by -91)
- torch 2.10.0a0 / cuda 13.0, `cuda.is_available()` **True** (12,1), status 2.95 s
- cv2 4.14.0, numpy 2.1.0, ros2 responding, `import icrt` / `import aiworker_icrt` OK

---

## 6. Commands

```bash
# convert (30 fps source → ICRT's ~15 fps design rate)
python -m aiworker_icrt.convert_lerobot_icrt \
    --lerobot-root data/ecu_raw --out-dir data/icrt_ecu \
    --target-fps 15 --prompt-dir data/icrt_ecu/prompts

# look at it — catches what assertions do not
python -m aiworker_icrt.visualize --dataset data/icrt_ecu --episode episode_000000

# verify it loads
python tests/check_dataset_roundtrip.py data/icrt_ecu 512

# train
torchrun --nproc_per_node=<N> third_party/icrt/scripts/train.py \
  --dataset-cfg.dataset-json data/icrt_ecu/dataset_config.json \
  --dataset-cfg.maximum-length 1200 \
  --dataset-cfg.non-overlapping 32 \
  --shared-cfg.num-arms 2 --shared-cfg.num-cameras 3 \
  --shared-cfg.seq-length 512 \
  --model-cfg.policy-cfg.scratch-llama-config third_party/icrt/config/model_config/custom_transformer.json \
  --model-cfg.policy-cfg.no-prompt-loss \
  --model-cfg.vision-encoder-cfg.vision-encoder vit_base_patch16_224.mae
```

**`--shared-cfg.seq-length 512` is not optional** with `--no-prompt-loss`. See §4.3.

---

## 7. Open

1. **Multi-task data.** The ECU set is 39 episodes of **one task**, supplied as a
   sample. ICRT's premise is learning a *new* task from in-context demos; one task
   cannot demonstrate it. The real multi-task set is on another device.
2. **`cyclo_control` for IK** — decided, not integrated. Its QP controller has
   self-collision (CBF) and bimanual coordination our DLS solver lacks. Two arms
   in a shared workspace can collide and our solver has no concept of that.
   SRDFs confirmed present at `third_party/cyclo_control/.../ffw_sg2_follower_*.srdf`.
3. **ICRT as a `cyclo_brain` backend?** Strategic, not technical. Would restructure
   `ros_node.py` and `policy.py`. Does not block training.
4. **`git init`** — still not a repo. Two sessions overwrote each other's files
   irrecoverably before ownership was assigned.
5. **Nothing has touched the robot.**
