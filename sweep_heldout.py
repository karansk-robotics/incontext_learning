"""Held-out sweep: warm vs scratch, matched checkpoints.

Answers two questions in one pass:
  1. Does the warm start's training-loss advantage survive on data the model has
     never seen, or was it memorising 151 episodes faster?
  2. Where does each run turn over into overfitting? mt21_gpu bottomed at epoch
     15 and then degraded for 20 more epochs while its train loss kept falling,
     so the LAST checkpoint is not automatically the best one.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "third_party/icrt"); sys.path.insert(0, ".")
from aiworker_icrt.evaluate import evaluate

D = Path("/dev/shm/icrt_multitask_split")
EPOCHS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [4, 8, 12, 16, 20]
WORK = {"ylw2": "right", "box": "left", "ecu": "left"}

v = json.load(open(D / "verb_to_episode.json"))
tr = set(json.load(open("runs/mt21_warm/train_split.json")))
val = sorted(json.load(open("runs/mt21_warm/val_split.json")))
short = lambda t: ("ylw2" if "yellow" in t.lower() else "ecu" if "ecu" in t.lower() else "box")
ep2 = {e: short(t) for t, eps in v.items() for e in eps}
# same-task prompt, always from TRAIN, identical for both runs
prompts = {short(t): sorted(x for x in eps if x in tr)[0] for t, eps in v.items()}

print(f"{len(val)} held-out episodes, epochs {EPOCHS}")
print("prompts:", json.dumps(prompts, indent=1), "\n")

out = {}
for run in ("mt21_warm", "mt21_scratch_split"):
    for E in EPOCHS:
        ck = Path(f"runs/{run}/checkpoint-{E}.pth")
        if not ck.exists():
            print(f"  {run} ep{E}: MISSING"); continue
        t0, maes = time.time(), []
        for e in val:
            t = ep2[e]
            r = evaluate(D, ck, Path(f"runs/{run}/run.yaml"), prompts[t], e,
                         Path(f"runs/{run}/sweep"), max_steps=40, prompt_max_steps=300)
            maes.append(r[WORK[t]]["pos_mae_m"])
        out[(run, E)] = sum(maes) / len(maes)
        print(f"  {run:<20} ep{E:<3} mean MAE {out[(run,E)]:.4f}  ({time.time()-t0:.0f}s)")

print("\n============ HELD-OUT WORKING-ARM MAE ============")
print("epoch    warm       scratch    warm better by")
for E in EPOCHS:
    w, s = out.get(("mt21_warm", E)), out.get(("mt21_scratch_split", E))
    if w is None or s is None: continue
    print(f"{E:5d}    {w:.4f}     {s:.4f}     {100*(s-w)/s:+.1f}%")
print("\nbaseline to beat: repeat-previous-action 0.0065 m")
json.dump({f"{k[0]}:{k[1]}": val_ for k, val_ in out.items()},
          open("runs/heldout_sweep.json", "w"), indent=2)
