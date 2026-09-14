# Change register

Every file created or modified, what changed, why, and who owns it.
Audience: someone picking this up in three months with no memory of the work.
`PLAN.md` is the plan, `EXECUTION.md` is the narrative — this is the index you grep.

Verified against `git -C third_party/icrt diff` on 2026-09-14 — **7 files,
31 hunks, 243 insertions**, counted from the diff rather than from a summary.
Where this disagrees with a summary elsewhere, the diff wins.

---

## The action space: 20-D → proprio 23 / action 21

Supersedes the 20-D layout everywhere. User-approved.

```
proprio 23 = [ L arm 10 | R arm 10 | head_joint1, head_joint2, lift ]   OBSERVED
action  21 = [ L arm 10 | R arm 10 |                          lift ]   COMMANDED
```

The **head is observed but never commanded**: it carries `cam_head`, so the
policy needs to know where it is looking to interpret the image, not to move it.
The **lift is both** — it sits inside the arm chain, and a demonstrator raising
the torso to reach something is part of the skill.

Consequences, in the order they will bite:

| | |
|---|---|
| Training needs two new flags | `--shared-cfg.proprio-extra-dim 3` and `--shared-cfg.action-extra-dim 1`. Without them the model builds at the old 20-D width and dies on a shape mismatch. The UI defaults them to 3/1 and refuses to launch otherwise. |
| **Every dataset converted before this is 20-D and will not load** | `dataset_config.json` now records `num_arms`, `proprio_extra`, `action_extra`. Their *absence* is the tell. `data/icrt_ecu` and `data/icrt_synth_v30` are both legacy and need re-converting. |
| `num_arms` is now threaded, not inferred | The delta-action splitter used to guess arm count from vector width. Once non-arm channels are appended that guess is unsound — it would split the tail into a phantom arm, silently. |
| Inference Gram-Schmidt passes the tail through | Re-orthogonalisation applies per arm block; the non-arm channels are carried untouched. |

**Verified for shape and gradient flow only.** Dataset len 1757, proprio
`(32,16,23)`, action `(32,16,22)`, forward loss 0.3665, 162 grad tensors,
inference action width 21.

> **"23/21 works" and "23/21 is useful" are different claims, and only the first
> has evidence.** On the ECU set the head and lift are *constant*, so training
> there teaches the model those channels never change. Whether the extra
> dimensions help is untestable until data arrives in which they vary.

---

## third_party/icrt — vendored ICRT, patched

**7 files, 203 insertions, 71 deletions.** Owner: `-01`.

Every change is backward compatible: **`num_arms=1` reproduces upstream exactly.**
That property is what lets us keep pulling from upstream, so preserve it.

| File | Change | Why |
|---|---|---|
| `util/args.py` | `SharedConfig.num_arms` (default 1); `DatasetConfig.minimum_length` / `maximum_length` | num_arms plumbs bimanual through. The length filters existed but as unreachable class attributes — the config never reached the dataset. |
| `util/model_constructor.py` | `proprio_dim = num_arms*10`, `action_dim = num_arms*10 + 1` | Per arm: xyz(3) + rot_6d(6) + gripper(1). The eos channel stays global and only on the rot_6d path, so 1-arm still yields upstream's 10/11. |
| `util/misc.py` | `torch.load(..., weights_only=False)` ×2 | PyTorch 2.6 flipped `weights_only` to default `True`. ICRT checkpoints embed an `ExperimentConfig` dataclass, so the safe unpickler refuses them. Affects resume and deployment, not just eval. |
| `data/utils.py` | `convert_delta_action` / `convert_abs_action` rewritten to per-arm 10-D blocks; `_split_arm_blocks`, `_infer_num_arms` helpers | Upstream reinterpreted a single rotation spanning the whole vector, so a second arm's pose leaked into the first arm's delta. |
| `data/dataset.py` | 7 hunks: `skip_rot_conversion` flag; proprio bypass; action guard; 4-D image passthrough; length filter honoured; **hard error when the filter empties the dataset** | `euler_to_rot_6d` was called unconditionally on `ret[:,3:]`, which gets 17 dims from a 20-D vector and throws. Images were byte-reshaped to a hardcoded 180×320 and any episode name that wasn't `episode`/`demo` raised. The empty-dataset error reports the actual length range and what to change — it replaces a silent `len(dataset)==0` that cost hours. |
| `models/policy/icrt.py` | 3 changes: `set_default_tensor_type` → `set_default_dtype` + `torch.device("cuda")`; `torch.load` guard; **per-arm Gram-Schmidt at inference** | `set_default_tensor_type` is deprecated since torch 2.1 and warns on both our environments; when removed the model would stop constructing and the failure would read as a container problem. The Gram-Schmidt fix is the serious one — see below. |
| `models/backbones/encoders.py` | `view` → `reshape` ×2; `torch.load` guard | `view` requires contiguous tensors and fails on some batch shapes. |

**Corrections to earlier summaries:** 7 files, not 8. `icrt.py` has 3 changes, not 2
(its `torch.load` call site was omitted). Total `torch.load` sites patched: 4 —
`misc.py` ×2, `encoders.py` ×1, `icrt.py` ×1.

### The bug worth remembering

ICRT's rot_6d re-orthogonalisation rebuilt the predicted action as
`cat([xyz, b1, b2, gripper])` — a single 10-D arm. Given a 20-D bimanual action it
**silently truncated to 10 and discarded the right arm**, with no exception at the
point of failure.

That path is **inference-only**. `forward()` never touches it, so a full training
run and 23 green tests coexisted with structurally broken inference. On the robot
the right arm would never have moved. `tests/test_inference_path.py` exists
specifically to close that gap.

---

## aiworker_icrt/ — first-party

Owner: `-01` except `ui/`.

| File | Purpose | GPU |
|---|---|---|
| `constants.py` | Canonical layouts: JOINT (22-D), CARTESIAN (20-D), camera key order | CPU |
| `kinematics.py` | URDF parser, FK, analytic Jacobian, DLS IK. Chains rooted at `base_link`, so each carries 8 joints (`lift_joint` + 7 arm); the lift is read, never commanded | CPU |
| `convert_lerobot_icrt.py` | LeRobot v2.1 **and** v3.0 → ICRT hdf5 + 4 sidecars | CPU |
| `policy.py` | `DualArmICRT` — replaces ICRT's single-arm wrapper | **GPU** |
| `ros_node.py` | ROS 2 inference node; publishes to the LEADER topics so it drops in for the physical leader | **GPU** |
| `visualize.py` | Per-episode trace PNG + MP4 | CPU |
| `evaluate.py` | Offline replay: prompt with one episode, replay another, compare | **GPU** |
| `audit_dataset.py` | Measures head/lift/base variation in a LeRobot set and recommends a configuration. **Run before converting anything new** | CPU |
| `ui/` (4 files) | Local web UI — owner `-61` | CPU (launches GPU jobs) |

**The GPU split is practically useful.** `icrt.py` builds the transformer under
`torch.device("cuda")`, so anything constructing a model fails **at construction**
without a visible GPU, not merely at the forward pass. But conversion — the
4-minute, 7.2 GB step — needs no GPU at all and can run on any machine, including
alongside a training job.

Measured during a real training run (seq_length 512): idle `util 1% / 5.2 W / 44 °C`,
training `util 96% / 62 W / 71 °C` sustained, torch peak 9812 MiB.

---

## Scope cut — PLAN.md revision 2

**There is no physical robot.** Deployment is out of scope. The following are
complete, documented and building, but nothing consumes them:

| Component | Why it is dormant |
|---|---|
| `aiworker_icrt/ros_node.py` | nothing to command |
| `launch/icrt_policy.launch.py` | no robot to launch against |
| `config/zenoh/` | no robot network |
| cyclo_control build (`--control`) | **offline evaluation needs no inverse kinematics** — it compares predicted vs recorded actions entirely in 21-D Cartesian space. IK existed only to turn those into joint commands. |
| preflight head pose | no robot to preflight |

**Do not delete any of it.** Phase E (simulation via
`ffw_sg2_follower_ai_gazebo.launch.py`) would bring it straight back, and it
costs nothing sitting there. It should simply consume no further effort.

**Forward kinematics is NOT in this list** and still matters: it is what makes
the training data Cartesian in the first place, and it is verified against
Pinocchio to 6.661e-16 m.

**What this promotes:** the container (training runs in it) and **the UI**, which
is now the primary interface for the entire project. Convert → view → train →
evaluate is the whole workflow.

---

## Infrastructure — owner `-61`

| File | What | Why |
|---|---|---|
| `docker/DockerFile.arm64`, `.x86` | NGC `pytorch:25.11-py3` + ROS 2 Jazzy + `rmw_zenoh_cpp` | Newest tag whose build driver (580.95.05) is ≤ the Spark's 580.173.02; covers sm_110 (Thor) natively and sm_120+PTX (GB10 sm_121 JITs from it). Floor is 25.08. |
| `docker/requirements.txt` | Runtime pins | See non-changes below |
| `docker/requirements-data.txt` | LeRobot stack, opt-in `--build-arg INSTALL_DATA_DEPS=1` | lerobot declares its own torch requirement; the robot image shouldn't carry that risk |
| `docker/README.md` | Why a green build ≠ a working GPU; the check-behaviour-not-presence table | Five green checks hid real failures |
| `docker-compose.yml` | `zenohd` (profiled) + `icrt`; `/data` bind mount | One router per network, not per host. `/data` defaults to `./data` — **point `ICRT_DATA_DIR` at NVMe before converting 180 GB** |
| `container.sh` | build / start / enter / stop / status / logs / exec | Re-execs under `sg docker` when the login session predates the group grant; `status` compares host driver against the image's build driver |
| `launch/icrt_policy.launch.py` | Includes `ffw_bringup/launch/ffw_sg2_follower_ai.launch.py` | NOT `ffw_sg2_ai.launch.py`, which also starts the LG2 leader and would fight for the same topics. `autostart` stays false — the KV cache needs prompting first |
| `config/zenoh/README.md` | Topology | One `rmw_zenohd` per network; SHM transport on; `mode="client"` when remote — aligned with ROBOTIS' own `cyclo_intelligence` |
| `tests/` | conftest, kinematics, delta_action, dataset_roundtrip, inference_path, pytest.ini | 27 passed / 5 skipped |
| `tests/check_pinocchio_parity.py` (`-01`) | FK vs Pinocchio, 400 configs: **6.661e-16 m** max position error | Skips without pinocchio |
| `aiworker_icrt/ui/` | stdlib HTTP job launcher + single-file front end | Four actions all spawn local processes; an Artifact cannot |

---

## Deliberate NON-changes

These are decisions. They will otherwise get "fixed" by someone later.

| Not done | Why |
|---|---|
| **kinpy not installed** | Nothing in the pipeline imports it. `kinematics.py` does its own URDF parse and Jacobian. Only ICRT's unused `scripts/eval_fk.py` wants it. Produces a harmless pip resolver warning. |
| **flash-attn not installed** | `handle_flash_attn` at `backbones/utils.py:11` has **zero call sites**. Its pinned v2.2.3 (2023) will not compile for sm_110/sm_121 anyway. torch SDPA on the cuDNN Blackwell backend is the faster path. |
| **cv-bridge REMOVED from the image** | Nothing imports it — `ros_node.py:225` already does `frombuffer` + `imdecode`. Keeping it is actively harmful: its C++ links apt libopencv 4.6 and `cv_bridge_boost.so` is compiled against NumPy 1.x. With pip cv2 4.14 it **segfaults**; with 5.0 it raises `KeyError: 16`. Re-adding it needs a rebuild against a matching OpenCV **and** NumPy 2 — and NumPy 2 is non-negotiable because NGC's torch is ABI-coupled to it. Dropping it also removed the opencv→hdf5→openmpi→libucx0 chain that broke the first build. |
| **opencv pinned `4.14.0.94`, must not float to 5.x** | OpenCV 5 changed the `CV_MAKETYPE` channel shift (`CV_8UC3` = 64 vs 16). An unpinned entry silently resolved to 5.0.0 on the first build. |
| **`_cuda_getArchFlags()`, never `get_arch_list()`** | Whether `get_arch_list()` works without a GPU is a property of the torch *build*: NGC 2.10.0a0 returns `[]`, PyPI 2.14.0 returns the full list. A build-time guard on it fails 100% of builds after a base bump. |
| **pinocchio is not a hard dependency** | Nothing in training or conversion needs it; only the parity check and the cyclo controller do. Both skip cleanly without it. |
| **UI binds loopback only** | It launches GPU jobs and this box is on the robot network. `--host 0.0.0.0` is a deliberate operator decision. |
