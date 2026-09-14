# Verification scripts

Exploratory checks written while building `aiworker_icrt`, kept because the
numbers in them are the evidence behind the kinematics and the ICRT patches.
They print PASS/FAIL rather than using pytest, and they were run with only
numpy+scipy (no torch), against a stub of ICRT's two rotation helpers.

`aiworker-iclr-61` owns turning these into proper regression tests.

## Measured on 2026-09-14 (numpy 2.5.3 / scipy 1.18.1)

`check_kinematics.py`
- analytic vs numerical Jacobian: max abs diff **3.83e-07**
- zero-pose EEF mirrored in y: L `[0, 0.2275, -0.8565]`, R `[0, -0.2275, -0.8565]`
- batch FK: 500 steps x 2 arms in ~2.3 ms (~217k steps/s)
- IK from a 0.25 rad perturbed seed: **119/120** converged, mean err 0.03 mm

`check_tracking.py` -- the realistic regime, warm-started from the previous solution
- **0/500 non-converged** per arm, mean **1.2-1.5 iterations**
- position error mean 40-51 um, max **<100 um**
- ~0.6 ms/step/arm (~1700 Hz), against a 15 Hz control budget
- max single-step joint jump 0.031 rad = 0.47 rad/s at 15 Hz (limit 4.8)

`check_delta_action.py` -- the dual-arm `convert_delta_action` patch
- round trip action -> delta -> abs: max err **1.11e-15**, with and without eos
- single-arm path unchanged (same 1.11e-15)
- no cross-arm bleed: perturbing arm 1's proprio leaves arm 0's delta bit-identical

## Known, accepted limits

`check_kinematics.py` reports 2 FAILs. Both are the script's thresholds being
stricter than physics allows, not defects:

- 1/120 IK seeds does not converge within 100 iterations, and worst-case residual
  is 0.12 mm. Every such case is a *deep* singularity, and the separation is
  stark: the non-converged case sits at sigma_min ~4e-04 against a sample median
  of 8.2e-02 -- roughly 200x below median, and below even the 1st percentile
  (2.0e-03). Damped least squares trades exactness for stability at a singularity
  by design, and 0.12 mm is far below the robot's repeatability.

  (An earlier draft of this file quoted the range "sigma_min 0.0009-0.09", which
  made the failures look like they blended into the normal distribution. That
  range was measured *before* the joint-limit deadlock fix and lumped in
  well-conditioned deadlock cases at sigma_min 0.05-0.09 that the fix eliminated.
  Post-fix, a repeat of the 800-sample sweep gives **0/800** non-converged.
  Corrected after aiworker-iclr-61 reproduced the numbers independently.)

When these become regression tests, assert the *warm-started* numbers from
`check_tracking.py` (which are clean) and allow a singular-configuration
tolerance in the random-seed test rather than demanding 120/120.

## Running

    python -m venv .venv && .venv/bin/pip install numpy scipy
    PYTHONPATH=<stub>:. .venv/bin/python tests/check_kinematics.py

where `<stub>` provides a torch-free `icrt.data.utils` exposing
`rot_mat_to_rot_6d` / `rot_6d_to_rot_mat` (copy them verbatim from
`third_party/icrt/icrt/data/utils.py` lines 1-32 and 88-108, minus the torch
import). Inside the container, torch is present and no stub is needed.

---

# Pytest regression suite (aiworker-iclr-61)

The scripts above stay runnable standalone. These files wrap the same
properties as assertions:

    tests/conftest.py          sys.path + conditional torch stub + shared fixtures
    tests/test_kinematics.py   11 tests
    tests/test_delta_action.py  4 tests
    tests/pytest.ini

Run:

    python -m venv .venv && .venv/bin/pip install numpy scipy pytest
    .venv/bin/python -m pytest tests/

No stub setup needed — `conftest.py` installs the torch stub **only if torch is
absent**, so the same command works inside the container against real torch.
The header line reports which (`torch: stub` / `torch: real`).

## Independent reproduction, 2026-09-14 (numpy 2.5.3 / scipy 1.18.1)

Re-measured from scratch, matching the values above:

| quantity | README above | re-measured |
|---|---|---|
| random-seed IK converged | 119/120 | **119/120** |
| worst residual | 0.12 mm | **0.1285 mm** |
| warm-started non-converged | 0/500 | **0/500** both arms |
| warm-started mean iterations | 1.2–1.5 | **1.00 / 1.34** |
| warm-started max position error | <100 µm | **97.4 / 99.8 µm** |

15 passed in 1.6 s.

## How the accepted failures are handled

The 2 known FAILs were NOT made green by loosening the solver. Instead:

- `test_ik_stress_accuracy` allows ≤2% of seeds to miss the iteration budget
  and caps the worst residual at 0.5 mm.
- `test_ik_nonconvergence_is_singular_only` then requires every miss to sit at
  or below the median `sigma_min` of the sample. A miss at a well-conditioned
  pose fails the suite even while the count is still inside the 2% allowance.
  This is what prevents someone widening the tolerance to hide a real solver
  regression.

  Measured separation is much cleaner than the range quoted above: the single
  non-converged case has `sigma_min` **4.16e-04** against a sample median of
  **8.54e-02** — roughly 200x below median, not merely inside a band.

The warm-started bounds carry 2x headroom over measurement (`IK_WARM_POS_MAX =
2e-4` vs 97–100 µm measured) and are the hard ones; they are the regime that
actually runs at inference.

Verified the suite can fail: tightening `IK_WARM_POS_MAX` to 1e-12 produces
2 failures, and reverting restores 15 passed.

## Scope

`SequenceDataset` is deliberately untested. As of 2026-09-14 loading a
converted dataset through it gives `len(dataset) == 0`, unresolved as to
whether that is a small-data windowing artifact or a defect. Do not add
assertions on that layer until aiworker-iclr-01 settles it.

## Container-only tier

`tests/test_dataset_roundtrip.py` wraps `check_dataset_roundtrip.py` (which
stays runnable standalone). It needs **real torch + torchvision** — the conftest
stub is not enough, since `SequenceDataset` does real tensor work — and a
converted dataset on disk. Both are absent outside the container, so it skips
rather than fails:

    ICRT_DATASET_DIR=/data/icrt/ffw_sg2_mt pytest tests/

Asserts `len(dataset) > 0` (with the `seq_length <= task_length` explanation on
failure, not an IndexError), observation `(B,3,3,224,224)`, proprio `(B,16,20)`,
action `(B,16,21)`, no NaNs, per-arm grippers in `[0,1]`, and binary masks —
matching the 24-episode run verified by `-01` on 2026-09-14 (len 1247).

Outside the container the full suite is **15 passed, 5 skipped**.
