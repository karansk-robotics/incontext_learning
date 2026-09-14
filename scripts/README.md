# Analysis scripts

One-question scripts written while debugging, recovered from `zeux` where they
had been living outside version control. Each answers one specific question and
says so in its docstring. They are kept because the *questions* recur, and
because several of them are the evidence behind claims in the documents.

Run from the repo root with `PYTHONPATH=third_party/icrt:.` — most read a
dataset from `/dev/shm` and a run directory from `runs/`, so paths inside them
may need updating for a new dataset.

## Does the data contain what we think it does

| script | question |
|---|---|
| `arms.py` | Do any episodes actually use BOTH arms, or one at a time? |
| `gap.py` | Is there a systematic gap between observed state and commanded action? |
| `chan.py` | Per-channel statistics of the action tensor the loss actually sees |
| `scale.py` | Per-dimension magnitude of the delta action the model regresses |
| `dist.py` | The real spread of two channels, not just their std |
| `howstd.py` | Mean and std computed by hand on 10 real values, to check the pipeline |

**`gap.py` is worth reading before trusting any servo-lag number.** It found the
gap correctly but aggregated **by arm**, which is how the result entered
`documentation.md` as "mean 0.0207 m left, 0.0012 m right". Since `ylw2` is the
only task using the right arm, "the right arm has less lag" and "that recording
session had less lag" are indistinguishable in that grouping — and it is the
second one that is true. See `action_target_fix.md`. **Group by task, not by
arm.**

## Is the policy any good

| script | question |
|---|---|
| `base_acc.py` | What does a trivial predictor score? Without this an MAE means nothing |
| `baseline.py` | Control for the safety gate: push the RECORDED actions through it |
| `dump_traj.py` | Dump predicted vs ground-truth trajectories to JSON for plotting |

## Why does inference break

| script | question |
|---|---|
| `diag.py` | Why does IK fail on the policy's poses but not on the demonstrations? |
| `perturb.py` | Is a 3.6 mm error enough to break our DLS IK on a ground-truth pose? |
| `pv.py` | Do the joint-velocity spikes come from the policy, or from IK? |

## Superseded, kept for provenance

| script | superseded by |
|---|---|
| `cond3_run.py`, `cond3_ctrl.py` | `aiworker_icrt/condition3.py` |
| `heldout_cmp.py`, `sweep_heldout2.py` | `sweep_heldout.py`, `validate_achieved.py` |

These produced the numbers in `condition3_result.md` and `warm_start_result.md`.
The packaged versions fix two flaws they share: writing every epoch's
trajectories into one directory so they overwrite, and picking the
alphabetically-first training episode as the prompt.
