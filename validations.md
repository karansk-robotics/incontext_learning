# validations.md — evidence register

**Revision 2, 2026-09-14.** What has actually been proven about this system, by
what measurement, and what has **not** been proven. This is not a test list. A
claim appears here only with the measurement supporting it and the name of whoever
took it; claims resting on nothing appear in §3 with equal prominence.

Host: DGX Spark, NVIDIA GB10 (sm_121, capability 12.1), driver 580.173.02,
Docker 29.2.1, aarch64 SBSA. Companion documents: `PLAN.md` (revision 2),
`changes.md`, `eval.md`. Revision 1 of the plan is archived at
`docs_PLAN_v1_archive.md`.

| Tag | Meaning |
|---|---|
| **[C]** | Measured first-hand in the container by the container session |
| **[M]** | Measured natively by the model/data session (`aiworker-iclr-01`), reported |
| **[D]** | Measured by the docker session (`aiworker-iclr-61`), reported |

**[M]** and **[D]** rows were not independently reproduced by the author of this
file. They are recorded as reported, not as verified twice. Where a **[C]**
measurement contradicts a reported one, both are shown and the discrepancy is
called out rather than silently resolved.

---

## 1. The central risk: the task may be identifiable without the prompt

This now outranks every other finding in the project.

Head angle is **constant within** each dataset but varies **substantially between**
them — [M]:

```
ECU        [+0.6427, +0.3496]
pick_box   [+0.4663, +0.3482]
yellow     [+0.7839, -0.1994]
```

Up to **18°** and **31°** apart. Each task was also recorded in its own physical
setup, so the scene differs too.

**Consequence.** The task is identifiable from a single observation. ICRT's claim
is that the task is inferred *from the demonstration*; a model that never reads the
prompt can still score well on a same-task-vs-different-task comparison purely by
recognising the head pose or the backdrop. **The Phase 6 gate as previously written
can therefore pass for the wrong reason.**

With no physical robot (§4), offline evaluation is the only evidence available, so
this risk cannot be retired by falling back to hardware. Any Phase 6 result must be
accompanied by a control that breaks the shortcut — otherwise a passing number
carries no information about in-context learning.

**Status: unresolved. No experiment has yet been run that distinguishes the two
explanations.**

---

## 2. What is proven

### 2.1 Container and GPU — [C]

Read out of the running `icrt` container:

```
python          3.12.3
torch           2.10.0a0+b558c986e8.nv25.11 / cuda 13.0
archFlags       sm_80 sm_86 sm_90 sm_100 sm_110 sm_120 compute_120
is_available    True
capability      (12, 1)
device          NVIDIA GB10
numpy           2.1.0     cv2 4.14.0     scipy 1.16.3     yaml 6.0.3
ros2            /opt/ros/jazzy/bin/ros2
icrt, aiworker_icrt   importable
cv_bridge       ModuleNotFoundError  (deliberately absent — §5.1)
image           31.5 GB      status 2.95 s wall clock
```

- **`sm_121` is absent from the arch list and the GPU still works.** The device
  JITs from `compute_120`. A guard requiring `sm_121` would reject a working
  install; check `sm_120`/`compute_120`.
- The anticipated PTX-JIT first-call delay **did not occur**.
- GPU passthrough is via **CDI** (`/var/run/cdi/nvidia.yaml`); no `nvidia` runtime
  is registered. Anything specifying `--runtime nvidia` fails here. — [D] found,
  [C] confirmed working.

### 2.2 Build pipeline — [C]

| Claim | Measurement |
|---|---|
| NGC base pulls anonymously | No 401; resolved by digest `sha256:417cbf33…` |
| ROS 2 Jazzy apt layer resolves | Codename derived → `noble`; `ros-jazzy-rmw-zenoh-cpp` `0.2.10-1noble.20260902.013751` arm64 |
| BuildKit heredoc works on 29.2.1 | Layer completed in 0.4 s |
| pip never displaces NVIDIA's torch | `Requirement already satisfied: torch … 2.10.0a0`; no `Uninstalling torch` in any build log |
| Zenoh override composed correctly | `transport/shared_memory/enabled=true;connect/endpoints=["tcp/127.0.0.1:7447"]` |

The build took **four attempts**; attempts 1–3 each shipped or hit a defect that
the preceding checks had reported healthy (§5).

### 2.3 Image path — [C]

The real `ros_node.py:225-226` path, exercised with data rather than imported:

```
CompressedImage -> np.frombuffer -> cv2.imdecode   (180, 320, 3) uint8
cv2.resize                                          (224, 224, 3)
numpy -> torch .cuda()                              cuda:0, NVIDIA GB10
```

### 2.4 Dimensions: proprio 23 / action 21 — [M]

User-approved; supersedes the 20-D arms-only space.

```
proprio 23 = [L arm 10 | R arm 10 | head_joint1, head_joint2, lift]   OBSERVED
action  21 = [L arm 10 | R arm 10 | lift]                             COMMANDED
```

The asymmetry is deliberate and worth recording, because it looks arbitrary later:

- **head** carries `cam_head` and sits on a separate branch from the arms, so it
  does not affect end-effector geometry. Observed but never commanded — the policy
  needs to know its own gaze to interpret the image, not to change it.
- **lift** sits *inside* the arm chain, so with poses in `base_link` a different
  torso height means a different z for the same arm pose. Observed **and**
  commanded: if a demonstrator raised the torso to reach a high shelf, that is part
  of the skill. Referencing from `arm_base_link` would have removed the lift from
  the maths and made the behaviour unreproducible.

Verified end to end on real ECU data: dataset len 1757, proprio `(32,16,23)`,
action `(32,16,22)` [21+eos]; model `proprio_dim 23`, `action_dim 21`, `num_arms 2`;
forward `action_loss` 0.3665; backward 162 grad tensors, mean |grad| 3.087e-04;
**inference action width `(21,)`** — the path that silently truncated before.

### 2.5 Forward kinematics vs Pinocchio — [M], NOT reproducible here — [C]

400 random configurations, both arms, full lift range, EEF in `base_link`:

```
position max  6.661e-16 m
rotation max  7.772e-16
```

Pinocchio is the library `cyclo_control`'s `KinematicsSolver` is built on, so this
is a **different class of evidence** from the Jacobian check: that one proved
self-consistency and would have differentiated a frame-convention error just as
faithfully as a correct one. This is the strongest single piece of evidence in the
project.

**Now executes in the container — SKIPPED → PASSED** — [C], verified after rebuild:

```
./container.sh parity     ->  1 passed, 32 deselected in 0.20s
```

`pin` could **not** simply be added to `docker/requirements.txt`. It installs
cleanly, satisfies the constraint file, and then fails to import — `eigenpy`,
pinocchio's binding layer, is compiled against NumPy 1.x while the base ships
2.1.0 (§5.11). The apt route (`ros-jazzy-pinocchio`) fails identically. A bare
requirements line would have been **a skip dressed as a fix**. — [D]

What works is an isolated sidecar, since the parity check is pure kinematics and
needs no torch:

```
/opt/pinvenv   numpy 1.26.4 | scipy 1.17.1 | pin 2.7.0 | eigenpy 3.5.1
built behind ARG INSTALL_PARITY=1 (default ON); cost +0.7 GB (31.5 -> 32.2 GB)
the only place in either Dockerfile where PIP_CONSTRAINT is deliberately bypassed
```

**`pin==2.7.0` is a ceiling, not a preference.** Every 3.x and 4.x release is
incompatible with `numpy<2`, so the version is forced. Verified independently — [C]:

```
pin==4.1.0   ResolutionImpossible
pin==3.3.1   ResolutionImpossible
pin==2.7.0   resolves
```

(An earlier report of 4.1.0 in the sidecar came from installing into the *image*
interpreter, where numpy is 2.1.0 and 4.1.0 resolves happily — and then fails at
import, which is the reason the sidecar exists at all. Now pinned explicitly in
both Dockerfiles with this table in the comment, so nobody "upgrades" it and meets
a bare `ResolutionImpossible`.)

**The version split makes the evidence stronger, not weaker.** The container
cross-checks our FK against **pinocchio 2.x** while the native measurement used
**4.x**. The naive reading — "the container only tests an old version" — gets it
backwards: two independent pinocchio majors, built years apart, both agree with our
hand-written FK to ~1e-16. A frame-convention error would have had to be replicated
identically in both to survive that.

**One caveat stands.** The skip message in the main interpreter is now stale:
running the main suite still prints *"Add `pin` to the image so it executes in
CI"* — advice that is complete, and which does not mention `./container.sh
parity`. A reader of `pytest -rs` is told to perform a finished action and is not
pointed at the command that runs the check. Right when written, wrong when read;
see §5.10.

### 2.6 Kinematics and dual-arm — [M]

| Claim | Measurement |
|---|---|
| Analytic Jacobian correct | max diff vs numerical **3.83e-07** |
| Warm-started IK tracking | 0/500 non-converged; mean **44.9 µm**, max 98.9 µm; ~1700 Hz/arm against a 15 Hz budget |
| Cold random seeds | 2/300 miss, both near-singular; converged cases mean **0.05 mm** |
| Velocity limits respected | max single-step jump 0.031 rad = **0.47 rad/s** at 15 Hz (limit 4.8) |
| IK cannot cheat the lift | +10 cm EEF height requested → lift held at −0.250, arm moved 0.203 rad |
| Dual-arm delta round trip | **1.11e-15**, no cross-arm bleed, single-arm path unchanged |

**`base_link` sits exactly at floor level** — verified from the URDF: wheel axle at
+0.0865 = exactly the wheel radius, so floor is z = 0.00000. EEF z is therefore
height above ground in metres, which is why the ECU episode reads 0.58 → 1.30 →
0.61 m as reach / lift / place.

**`cyclo_control`'s model is already reduced**: `nq = nv = 15` — lift + 14 arm
joints. No wheels, no head, no grippers, no sensors. It is an arms+lift solver, not
a whole-body one, so there is no risk of the QP driving the base. `dof_ = model_.nq`
is safe here only because no continuous joints survive the reduction. (This
supersedes an earlier, incorrect reading of that model.)

### 2.7 Conversion, model, training, evaluation — [M]

Conversion on **real** ECU data: 39 episodes, 13,357 frames, 30→15 fps, 280–412
frames/episode, 7.2 GB, ~4 min. LeRobot **v3.0** video alignment verified
frame-exactly against an independent full decode of the concatenated mp4.

Model: dataset len 12,565; observation `(32,3,3,224,224)`; 92.6 M trainable /
178.4 M total.

Training — **the two runs must be read together**:

```
seq_length 32    392 it, 0.19 s/it        -> loss 0.0000 on ~90% of iterations
seq_length 512    47 it, 2.32 s/it, 9.8 GB -> loss 0.3550 -> 0.2399 over one epoch
```

The `seq_length 32` run **completes, reports success, and learns nothing.** A zero
loss on 90% of iterations is the failure, not the achievement. Only the
`seq_length 512` number is a real training result. This is the most misreadable
number in the project.

Evaluation (1-epoch checkpoint, prompt ep0 / eval ep5, 60 steps):

```
left    pos_mae 0.126 m   rmse 0.165   max 0.305   gripper_agreement 1.00
right   pos_mae 0.017 m   rmse 0.027   max 0.144   gripper_agreement 1.00
```

**Plumbing results, not performance results.** They show the evaluation path runs
end to end on a real checkpoint. They are poor, as expected from one epoch, and
must not be quoted as model quality.

### 2.8 Multi-task data — audited, not converted — [M]

Four datasets: **120 episodes, 118,563 frames, 3 distinct tasks**, all v2.1 /
30 fps / 22-D / same three cameras / identical joint names.

- Joint names identical (22) across all four.
- `ffw_sg2_rev1_classical` is a **naming variant, not a different robot** — only one
  SG2 URDF exists and the joints match.
- Episode lengths at 15 fps: 259–422 (pick_box), 860–1070 (yellow). Drives
  `seq_length 1024` / `maximum_length 1200`.
- Head and lift are **constant within** every dataset (max within-episode motion
  0.0031 rad = servo jitter); lift varies only 1.4 mm *across* datasets.
- Uncommanded-joint audit on the **raw 30 fps source** (26,694 frames, not the
  subsampled copy): head commanded to a single constant in every frame of every
  episode; lift constant −0.0005; base stationary; **0/39 episodes moved >2°**.

`aiworker_icrt/audit_dataset.py` produces this and recommends a configuration. It
is intended to be run on any new data **before** converting it.

### 2.9 Test suite — [C]

All measured in the container after the parity-sidecar rebuild:

```
main suite, ICRT_DATASET_DIR=/data/icrt_ecu   32 passed,  1 skipped in 6.75s
main suite, no ICRT_DATASET_DIR               27 passed,  6 skipped in 5.94s
./container.sh parity (sidecar)                1 passed, 32 deselected in 0.20s
```

33 tests collected. With the dataset set, everything passes: 32 in the main
interpreter plus the parity check in the sidecar. The single skip in the first row
**is** the parity check, which is correct — it is deselected from the main
interpreter by design and runs in the sidecar instead. The five extra skips in the
second row are the dataset tier.

Read these together or not at all: "27 passed, 6 skipped" and "32 passed, 1
skipped" are the same suite with and without `ICRT_DATASET_DIR`.

**Superseded, and recorded because the mechanism matters (§5.10):** an earlier
revision reported the suite *red* — `test_dataset_roundtrip.py:79` asserting
`proprio == 20` against 23-D data. Real, and now fixed; both the standalone check
and the pytest wrapper derive dimensions from `aiworker_icrt.constants`
(`C.PROPRIO_DIM`, `C.ACTION_DIM + 1`) instead of hardcoding them.

Data directories are **not interchangeable**:

| Path | Size | What it is |
|---|---|---|
| `data/ecu_raw` | 459 MB | Raw ROBOTIS ECU recordings |
| `data/icrt_ecu` | 7.2 GB | Converted real data — **39 episodes of ONE task** |
| `data/icrt_synth_v30` | 45 MB | **Synthetic**, 24 generated episodes, 3 tasks |
| `data/raw_…pick_box_1` | 679 MB | Raw multi-task, unconverted |
| `data/raw_…bin_yellow_last` | 22 MB | Raw multi-task, unconverted |
| `data/raw_…bin_yellow_last_2` | 1.2 GB | Raw multi-task, unconverted |

Any number measured against `icrt_synth_v30` proves **plumbing**, not that the
pipeline handles real recordings.

---

## 3. What is NOT proven

Stated as plainly as §2 and deliberately not softened.

1. **The §1 question is unanswered and is the project's central risk.** No
   experiment distinguishes "learned in-context" from "recognised the scene".
2. **No training at 23/21 dimensions.** Shape and gradient flow only.
3. **No multi-task training of any kind.** The 120 new episodes are audited, not
   converted and not trained on.
4. **No evaluation beyond one plumbing run on a 1-epoch checkpoint.**
5. **23/21 usefulness is untestable on current data.** Head and lift are constant
   within every dataset, so the extra channels carry no within-episode signal.
   Training on this data teaches the model those channels never change.
   **"23/21 works" and "23/21 is useful" are different claims; only the first is
   evidenced.**
6. **The parity sidecar's skip message is stale** (§2.5) — it still advises adding
   a wheel that has been proven unable to work, and does not name
   `./container.sh parity`.
7. **`--data` / `INSTALL_DATA_DEPS=1` has never been built.** Every container build
   used the plain profile; this is the most likely thing to stress the pip
   constraint set.
8. **The x86 Dockerfile has never been built** — lint-clean only.
9. **No sustained or thermal run.** Longest observed is one epoch.
10. **Neither peer session can see the UI**; headless Firefox exits 0 without
    writing a file here, so no UI behaviour is verified.
11. **cv2 is pinned to 4.x** on the assumption existing code targets 4.x; no test
    exercises an API that differs between 4 and 5.

---

## 4. Out of scope as of revision 2

There is **no physical robot**; deployment is out of scope. The following are
*removed from scope*, **not proven** — they were never validated, and if simulation
is chosen (Phase E) they return immediately. They are kept visible so no reader
assumes they were checked.

| Item | Why out of scope |
|---|---|
| `ros_node.py` never executed | No robot |
| Nothing run on physical hardware | No robot |
| No Zenoh router ever run | No robot network |
| `cyclo_control` IK not integrated | Offline evaluation needs **no IK** |

The last needs a sentence, because it otherwise reads as a gap: evaluation compares
predicted against recorded actions entirely in the 21-D Cartesian space, so
**inverse** kinematics was only ever needed to drive hardware. **Forward**
kinematics still matters and is verified (§2.5, §2.6).

---

## 5. Green signals that hid real breakage

The throughline of this project. **A green signal only covers the path it
exercises.** Eleven instances, each of which reported healthy while something
was broken.

### 5.1 `import cv_bridge` succeeded while its conversion segfaulted — [C]

`import cv2` passed. `import cv_bridge` passed. `status` printed `cv2 5.0.0`. The
conversion path **segfaulted the process** on first real use. Two independent
causes, which is why pinning could not fix it:

```
opencv 5.0.0 : CV_8UC3 = 64  but cv_bridge.so getCvType(bgr8) = 16  -> KeyError: 16
               (OpenCV 5 changed the CV_MAKETYPE channel shift; C1 types still
                agree, so mono8/mono16 work and every colour encoding breaks)

opencv 4.14.0: CV_8UC3 = 16  type codes now AGREE, and it still dies:
               cv_bridge_boost.so is built against NumPy 1.x, base ships NumPy 2.1.0
               -> Segmentation fault (core dumped) in cvtColor2
```

Pinning OpenCV 4.x fixed the first cause and converted a catchable `KeyError` into
a **hard process kill**. NumPy 2 cannot be given up — NGC's torch is ABI-coupled to
it — so rebuilding `cv-bridge` against a matching OpenCV is **not sufficient**; it
would also need rebuilding against NumPy 2. Resolution: `cv-bridge`,
`image-transport` and `compressed-image-transport` removed entirely; nothing
imported them.

Narrower true statement worth preserving: `imgmsg_to_cv2(msg, "passthrough")` is
lossless even in the broken state, because it never enters the C++ path. The
*conversion* path was unusable, not the package entire.

### 5.2 `status` printed identically for healthy and broken — [C]

`container.sh exec`/`status` used `bash -lc`. `/root/.bashrc` carries Ubuntu's stock
guard on line 6 — `[ -z "$PS1" ] && return` — and the `source …/setup.bash` line was
appended *below* it, so it never ran non-interactively. Under `exec`: no `ros2`, no
`rclpy`, empty `AMENT_PREFIX_PATH`.

`status` then printed `no nodes / no router` — **the same string as the legitimate
no-router state**, which everyone had been told to expect. The check converted an
absence of information into confidence.

### 5.3 A CUDA-built torch that cannot drive the GPU — [D], mechanism [C]

`robotis/cyclo-intelligence:1.4.0`, measured **with `--gpus all`**:

```
version.cuda 12.8 | backends.is_built True | _is_compiled True
_cuda_getArchFlags 'sm_87'   <- Jetson Orin only
is_available False
```

Not a "CPU build", and not "zero compiled architectures" — both earlier diagnoses
were wrong. An **L4T/Orin-targeted torch on a Blackwell box**. `get_arch_list()`
returning `[]` was the *symptom*.

### 5.4 The guard that would have failed every build — [C]

```
NGC 2.10.0a0,      no GPU  ->  []          (gated on is_available)
PyPI 2.14.0+cu130, no GPU  ->  full list   (gated on _is_compiled)
```

Build layers have no GPU, so a `get_arch_list()` guard returns `[]` on a good image
and fails **100% of builds** — while looking exactly like §5.3. The gating is a
property of the individual torch *build*, not the environment, so it can start
returning `[]` on a base-image bump with no warning. Use
`torch._C._cuda_getArchFlags()`.

### 5.5 A complete training run while the right arm was discarded — [M]

28 passing tests, a clean forward/backward, and a **full training run** were all
green while the right arm was silently discarded at inference — because training
never calls `forward_inference`. On a robot the right arm would never have moved.

### 5.6 IK solved in the wrong frame, converging beautifully — [M]

Found because the user asked "if the lift moves during inference, is that
dangerous?" It was. `policy.py` read the **measured** lift for forward kinematics
but a **stale latched** value for inverse kinematics, so the two disagreed about
where the shoulders were:

```
torso moves  +5 cm  -> hand lands  50.0 mm off
torso moves +15 cm  -> hand lands 150.0 mm off
```

Error is 1:1 with torso movement. **Every check reported success** — IK converged
to sub-100 µm on the pose it was asked for, in the wrong frame. Convergence
measures agreement with the question asked, never whether it was the right
question. Fixed: IK now uses the live measurement (150 mm → 0.0 mm), with drift
reporting past 1 cm because it remains off-distribution.

### 5.7 `audit_dataset.py` reported "head is constant throughout" — [M]

True per dataset, and it misses the point entirely. The tool only examines one root
at a time; the 18° between-dataset spread that constitutes §1 was found by
comparing outputs **by hand**. A green signal from a scope that excluded the
failing case. Reported by its author as their own bug.

### 5.8 A video file that every proxy played and no consumer could — [M]

`visualize.py` wrote mpeg4/mp4v via `cv2.VideoWriter`; no browser plays it, and
every H.264 fourcc fails to open because `opencv-python-headless` ships FFmpeg
without libx264. **`ffprobe` and VLC both played the file happily** — they were
proxies for the consumer, not the consumer. Fixed by piping to system ffmpeg;
verified h264 / avc1 / yuv420p, moov at byte 0x24 (faststart), 5.3 MB → 1.57 MB.

### 5.9 A skipping check is a green signal covering nothing — [C]

`tests/check_pinocchio_parity.py` was not collected by pytest at all — `pytest.ini`
sets `python_files = test_*.py` and the file was named `check_*`. The suite was
green; the strongest evidence in the project ran nowhere.

**Resolved, and the resolution is not the obvious one.** It is not "the wheel
landed" — the wheel *cannot* land (§5.11). It is: the wheel cannot land, here is
why, and here is what replaced it — an isolated `numpy<2` sidecar at
`/opt/pinvenv`, run via `./container.sh parity`. Verified SKIPPED → PASSED in the
container (§2.5).

The general lesson stands and is now cheap to apply: a skip is not a pass, and a
suite reporting only "N passed" makes the two indistinguishable. **Always run
`pytest -rs`.**

### 5.10 A number that was true when measured and false when reported — [M], caught by [C]

The suite was reported as "32 passed, backward compatibility holds". It was
`1 failed, 31 passed` when checked. The measurement had been taken **before**
`data/icrt_ecu` was reconverted to 23/21, and never re-run — true when measured,
false when reported.

It survived because **two files carried the same hardcoded dimensions**: the
standalone `check_dataset_roundtrip.py` was fixed to derive them from constants,
while `test_dataset_roundtrip.py` — the pytest wrapper — kept its own hardcoded
20/21. The file its author was looking at went green while the suite stayed red.

This is the only entry in §5 whose failure mode is *staleness* rather than *scope*,
and it is the one most likely to recur, because nothing about a stale number looks
wrong. It was caught only because the number was re-measured instead of
transcribed. **Re-run, don't relay.**

### 5.11 The NumPy 2 ABI wall — three packages, one failure, always a clean install

Three separate packages hit the identical wall tonight, and **every one of them
installed successfully first**:

| Package | Presented as | Actually |
|---|---|---|
| `cv_bridge` (apt) | imports fine | segfaults in `cvtColor2` on first conversion |
| `opencv-python-headless` 4.14 | installs, imports | segfaults *via* cv_bridge's extension |
| `eigenpy` / `pin` (pip **and** apt) | installs, constraint satisfied, dry-run clean | `ImportError: compiled using NumPy 1.x` |

The rule: **anything in this image that links a NumPy C extension built before
NumPy 2.0 will fail, and it will always present as a working install.** The NGC
base ships NumPy 2.1.0 and torch is ABI-coupled to it, so NumPy 2 cannot be given
up to satisfy such a package. `pip install` succeeding proves nothing; only an
import — and for compiled extensions, an actual *call* — does.

**A constraint file cannot catch this class, and not because it is misconfigured.**
`eigenpy` genuinely satisfies its declared metadata — it declares no NumPy upper
bound because its authors could not know about a future ABI break. A constraint
file pins *what you already have*; it cannot pin *what a dependency was compiled
against*, and no package metadata expresses that. The only detection is running the
import. This is why installing `pin` into a live container produced the answer
while a clean `--dry-run` did not.

The workable responses are: drop the package (cv_bridge), pin below the break
(opencv 4.x, where the break was OpenCV-major rather than NumPy), or isolate it in
its own interpreter (the parity sidecar). Only the third preserves the capability.

### The rule

When adding a check, ask: **what would a broken version of this print?** If the
answer is "the same thing", the check is worse than nothing. Prefer exercising a
path over importing it; round-trip real data rather than asserting a module loads;
and confirm that the consumer accepts the output, not merely that a tool you
happened to have can read it.

---

## 6. How to re-run

Docker access: `dexstro` is in the `docker` group per `/etc/group`, but a login
session predating the membership has stale credentials. `container.sh` re-execs
under `sg docker` automatically; from a fresh login it is unnecessary.

```bash
cd /home/dexstro/development/aiworker_iclr

# build (plain profile; --data adds the LeRobot stack and is UNEXERCISED)
./container.sh build
ICRT_DATA_DIR=$PWD/data ./container.sh start
./container.sh status

# §2.1 environment snapshot
./container.sh exec "python -c 'import torch;print(torch.__version__, torch.version.cuda, torch._C._cuda_getArchFlags(), torch.cuda.is_available(), torch.cuda.get_device_capability())'"

# §5.1 cv_bridge must be cleanly ABSENT, not present-and-broken
./container.sh exec "python -c 'import cv_bridge'"        # expect ModuleNotFoundError

# §2.9 suite — ALWAYS pass -rs; a skip is not a pass (§5.9)
./container.sh exec "ICRT_DATASET_DIR=/data/icrt_ecu       python -m pytest /workspace/tests/ -q -rs"
./container.sh exec "ICRT_DATASET_DIR=/data/icrt_synth_v30 python -m pytest /workspace/tests/ -q -rs"

# §2.8 audit any new data BEFORE converting it
./container.sh exec "python -m aiworker_icrt.audit_dataset <dataset-root>"
```

Native runs (§2.5–§2.7) belong to the model session's environment, not the
container.

**Before trusting any green result from the above, re-read §5.**
