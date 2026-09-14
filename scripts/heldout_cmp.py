"""Is the warm run learning, or memorising faster?

Training loss cannot answer it: on this project a falling train loss coexisted
with 12cm of hand error for 46 epochs. Held-out error can.

Matched epoch, identical val split, identical prompts. CPU so the two live
training runs are not disturbed.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "third_party/icrt"); sys.path.insert(0, ".")
from aiworker_icrt.evaluate import evaluate

D = Path("/dev/shm/icrt_multitask_split")
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 14
WORK = {"ylw2": "right", "box": "left", "ecu": "left"}

v = json.load(open(D / "verb_to_episode.json"))
tr = set(json.load(open("runs/mt21_warm/train_split.json")))
val = sorted(json.load(open("runs/mt21_warm/val_split.json")))
short = lambda t: ("ylw2" if "yellow" in t.lower() else "ecu" if "ecu" in t.lower() else "box")
ep2 = {e: short(t) for t, eps in v.items() for e in eps}

# one held-out episode per task, and a same-task prompt from train
evals, seen = [], set()
for e in val:
    if ep2[e] not in seen:
        seen.add(ep2[e]); evals.append(e)
prompts = {}
for t, eps in v.items():
    cand = sorted(x for x in eps if x in tr)
    prompts[short(t)] = cand[0]

print(f"epoch {EP}, {len(evals)} held-out episodes\n")
res = {}
for run in ("mt21_warm", "mt21_scratch_split"):
    ck = Path(f"runs/{run}/checkpoint-{EP}.pth")
    if not ck.exists():
        print(f"{run}: no checkpoint-{EP}"); continue
    maes = []
    for e in evals:
        t = ep2[e]
        t0 = time.time()
        r = evaluate(D, ck, Path(f"runs/{run}/run.yaml"), prompts[t], e,
                     Path(f"runs/{run}/heldout_ep{EP}"),
                     max_steps=40, prompt_max_steps=300, device="cpu")
        m = r[WORK[t]]["pos_mae_m"]
        maes.append(m)
        print(f"  {run:<20} {e:<28} {t:<5} mae={m:.4f}  ({time.time()-t0:.0f}s)")
    res[run] = sum(maes) / len(maes)

print()
for k, v_ in res.items():
    print(f"{k:<22} mean working-arm MAE {v_:.4f}")
if len(res) == 2:
    w, s = res["mt21_warm"], res["mt21_scratch_split"]
    print(f"\nwarm vs scratch: {100*(w-s)/s:+.1f}%   (negative = warm better)")
print("baseline to beat: repeat-previous-action 0.0065 m")
