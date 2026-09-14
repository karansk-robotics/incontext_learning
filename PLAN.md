# ICRT × AI Worker — Plan

**Revision 2, 2026-09-14.** Rewritten after two facts landed that change the shape
of the project: the multi-task data arrived, and **there is no physical robot.**

Revision 1 is archived at `docs_PLAN_v1_archive.md` — its survey of the ROBOTIS
`cyclo` stack is still the best reference we have, even though the integration it
recommends is now out of scope.

Companions: `EXECUTION.md` (what was built), `changes.md` (what changed),
`validations.md` (what is proven), `eval.md` (how to evaluate),
`architecture.html` (the model, drawn).

---

## 0. What changed, revision 1 → 2

**No robot.** Deployment is out of scope. Everything whose only purpose was to
drive hardware is now dead weight:

| dropped | why |
|---|---|
| `ros_node.py` | nothing to command |
| Zenoh router / `config/zenoh` | no robot network |
| Preflight head check | no robot to preflight |
| **`cyclo_control` IK integration** | **offline evaluation needs no IK at all** |

That last one matters most. "Predicted action vs recorded action" is compared
entirely in the 21-D Cartesian space. Inverse kinematics existed solely to turn
those into joint commands. *Forward* kinematics still matters — it is what makes
the training data Cartesian — and it is done and verified to 6.661e-16.

**Multi-task data arrived.** Four datasets, three tasks, 120 episodes.

**And a new risk appeared that outranks everything else.** See §4.

---

## 1. The goal, restated for an offline project

Show that ICRT, retargeted onto the AI Worker's bimanual Cartesian space, **infers
a task from demonstrations given in context** rather than from the observation
alone.

Offline evaluation is now the *only* evidence. There is no hardware rollout to
fall back on if the result is ambiguous. That raises the bar on experimental
design, which is where §4 bites.

---

## 2. The data

| dataset | eps | frames | frames/ep @30fps | task |
|---|---|---|---|---|
| `pick_the_ecu_part_final` | 39 | 26,694 | 684 (23 s) | pick ECU part → box |
| `ffw_sg2_rev1_pick_box_1` | 50 | 32,587 | 652 (22 s) | pick blue-handkerchief box |
| `..._bin_yellow_last` | 1 | 1,011 | 1011 | pick yellow part from bin |
| `..._bin_yellow_last_2` | 30 | 58,271 | **1942 (65 s)** | pick yellow part from bin |

**120 episodes · 118,563 frames · 3 distinct tasks.** All v2.1, 30 fps, 22-D state
and action, same three cameras, identical joint names. `ffw_sg2_rev1_classical` is
a naming variant — there is only one SG2 URDF and the joints match.

Configuration consequences:
- At 15 fps the longest episodes reach **1070 frames** → `seq_length 1024`
  (~96% boundary coverage) and `maximum_length ~1200`.
- Lift varies 1.4 mm across datasets. Negligible.
- The **head does not**. See §4.

---

## 3. Design decisions (settled)

| | |
|---|---|
| proprio | **23-D** = `[L arm 10 \| R arm 10 \| head×2, lift]` — observed |
| action | **21-D** = `[L arm 10 \| R arm 10 \| lift]` — commanded |
| per-arm block | `[xyz(3), rot_6d(6), gripper(1)]` |
| reference frame | `base_link`, which sits exactly at floor level |
| gripper sense | **0 = open, 1 = closed** (RH-P12-RN) |
| backbone | LLaMA *architecture* from scratch — 768-dim, 12 layers, 92.6M trainable |
| vision | `vit_base_patch16_224.mae`, frozen |
| rate | 30 → 15 fps at conversion |

---

## 4. THE CENTRAL RISK — read before anything else

**The task is identifiable from a single observation, so the prompt may be
redundant.**

Three tasks were recorded in three different physical setups *and at three
different camera angles*:

| dataset | head_joint1 | head_joint2 |
|---|---|---|
| ECU | +0.6427 | +0.3496 |
| pick_box_1 | +0.4663 | +0.3482 |
| yellow | +0.7839 | **−0.1994** |

Up to **18°** and **31°** apart. Head angle alone predicts the task perfectly; so
does the scene.

ICRT's claim is *"infer the task from the demonstration."* If a model can tell
which task it is doing from one frame, it never needs to read the prompt — and it
will still score well on a naive same-task-vs-different-task comparison.

**The original Phase-6 gate can therefore pass for the wrong reason.**

### Testing it honestly — three conditions, not one

1. **Same-task prompt** — baseline.
2. **Different-task prompt** — control.
3. **Adversarial: task A's prompt, task B's observations** — the real test.
   Does the policy follow the *prompt* or the *scene*? Following the scene means
   recognition, not in-context learning.

Condition 3 costs nothing and uses data already on disk. **It is the core
experiment**, not a robustness check.

If the model follows the scene, the honest options are: report it as a negative
result about this data; or record **same-scene, different-task** episodes (same
bin, same camera: "pick the yellow part" vs "pick the blue part"), the only setup
where the observation is genuinely ambiguous and the prompt is the sole
disambiguator. That recording would be the highest-value addition to the project.

---

## 5. Phases

### Phase A — Convert (ready now)
Convert all four, then merge into one multi-task set. ICRT's `dataset_path` and
`hdf5_keys` are lists, so the merge is config plumbing plus a combined
`verb_to_episode.json`.

```bash
python -m aiworker_icrt.convert_lerobot_icrt \
    --lerobot-root data/<raw> --out-dir data/icrt_<name> --target-fps 15
```
**Gate:** every episode carries a task string; three distinct tasks in the merged
`verb_to_episode.json`; `check_dataset_roundtrip.py` passes at `seq_length 1024`.
~25 min, ~35 GB.

### Phase B — Train
```
--shared-cfg.num-arms 2 --shared-cfg.num-cameras 3
--shared-cfg.proprio-extra-dim 3 --shared-cfg.action-extra-dim 1
--shared-cfg.seq-length 1024
--dataset-cfg.maximum-length 1200
--model-cfg.policy-cfg.no-prompt-loss
```
**Gate:** loss decreasing and **non-zero**. A flat `0.0000` means the
prompt-masking window is wrong (EXECUTION.md §4.3). A job recording no loss
samples is a failure, not a success.

### Phase C — Evaluate (the paper)
The three conditions in §4. Report all three.
**Gate:** condition 1 beats condition 2, *and* condition 3 shows prompt-following
rather than scene-following.

### Phase D — Write up
`architecture.html` already carries the model figures.

### Phase E — Simulation (OPTIONAL)
`ai_worker` ships `ffw_sg2_follower_ai_gazebo.launch.py`. If a sim rollout would
strengthen the paper, `cyclo_control` IK and `ros_node.py` come back into scope.
**Decide before spending anything here.** Out of scope as of revision 2.

---

## 6. Status

**Verified by execution:** FK vs Pinocchio **6.661e-16** · IK warm-started 0/500,
<100 µm · dual-arm delta round trip 1.11e-15 · conversion on real data, v2.1 and
v3.0, video alignment frame-exact · 23/21 end to end (forward, backward, inference
width 21) · training 0.3550 → 0.2399 at seq_length 512 · container GPU True, 96%
under load · 32 tests passing.

**Not done:** no training at 23/21 · no multi-task training · no evaluation beyond
a plumbing run · **§4 unanswered**.

**Dead per §0:** ros_node, Zenoh, preflight, cyclo IK.

---

## 7. Ownership

| | |
|---|---|
| `-01` | `aiworker_icrt/`, `third_party/icrt/`, `assets/`, `tests/`, PLAN, EXECUTION |
| `-61` | `docker/`, compose, `container.sh`, `launch/`, `config/`, `aiworker_icrt/ui/`, changes.md, eval.md |
| `-91` | build execution, verification reporting, validations.md |

Strict, because two sessions overwrote each other's files irrecoverably before it
was assigned. **The project root is still not a git repo** — which is why those
files were unrecoverable.
