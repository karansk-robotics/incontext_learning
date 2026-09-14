"""Control: does the prompt reach the model AT ALL?

Condition 3 showed same-task and cross-task MAE identical to 4 decimals on some
episodes. Before concluding "the policy ignores the prompt", rule out "the prompt
never arrives". Prompting with the EXACT episode about to be replayed is the
strongest possible prompt -- the model has literally seen the answer. If that is
also indistinguishable, the prompt pathway is broken, not the learning.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "third_party/icrt"); sys.path.insert(0, ".")
from aiworker_icrt.evaluate import evaluate

D = Path("/dev/shm/icrt_multitask"); OUT = Path("runs/mt21_gpu/cond3_ctrl")
CKPT = Path("runs/mt21_gpu/checkpoint-15.pth"); YAML = Path("runs/mt21_gpu/run.yaml")
v = json.load(open(D / "verb_to_episode.json"))
val = sorted(json.load(open("runs/mt21_gpu/val_split.json")))
short = {t: ("ylw2" if "yellow" in t else "ecu" if "ecu" in t else "box") for t in v}
ep2 = {x: short[t] for t, eps in v.items() for x in eps}
WORK = {"ylw2": "right", "box": "left", "ecu": "left"}
prev = json.load(open("runs/mt21_gpu/cond3/cond3_results.json"))

print("%-22s %-6s %-10s %-10s %-10s" % ("eval", "task", "SELF", "same-task", "cross-task"))
for ev in val:
    arm = WORK[ep2[ev]]
    r = evaluate(D, CKPT, YAML, ev, ev, OUT, max_steps=40, prompt_max_steps=300)
    self_mae = r[arm]["pos_mae_m"]
    rs = [x for x in prev if x["eval"] == ev]
    s = sum(x["mae"] for x in rs if x["same"]) / max(1, len([x for x in rs if x["same"]]))
    c = sum(x["mae"] for x in rs if not x["same"]) / max(1, len([x for x in rs if not x["same"]]))
    print("%-22s %-6s %-10.4f %-10.4f %-10.4f" % (ev, ep2[ev], self_mae, s, c))
