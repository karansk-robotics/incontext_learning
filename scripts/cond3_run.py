"""Condition 3: does prompting with the SAME task beat prompting with a DIFFERENT one?

Prompt length is held constant at 300 frames (--prompt-max-steps) so the
comparison varies task, not length. Prompts come only from the train split.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "third_party/icrt"); sys.path.insert(0, ".")
from aiworker_icrt.evaluate import evaluate

D = Path("/dev/shm/icrt_multitask")
CKPT = Path("runs/mt21_gpu/checkpoint-15.pth")
YAML = Path("runs/mt21_gpu/run.yaml")
OUT = Path("runs/mt21_gpu/cond3")
PROMPT_LEN, STEPS = 300, 40

v = json.load(open(D / "verb_to_episode.json"))
tr = set(json.load(open("runs/mt21_gpu/train_split.json")))
val = json.load(open("runs/mt21_gpu/val_split.json"))
short = {t: ("ylw2" if "yellow" in t else "ecu" if "ecu" in t else "box") for t in v}
ep2task = {x: short[t] for t, eps in v.items() for x in eps}
WORKING = {"ylw2": "right", "box": "left", "ecu": "left"}

prompts = {}
for t, eps in v.items():
    cand = sorted(x for x in eps if x in tr and x.startswith(short[t]))
    prompts[short[t]] = cand[:2]
print("prompt episodes:", json.dumps(prompts, indent=1))

rows = []
for ev in sorted(val):
    et = ep2task[ev]
    for pt, peps in sorted(prompts.items()):
        for pe in peps:
            r = evaluate(D, CKPT, YAML, pe, ev, OUT,
                         max_steps=STEPS, prompt_max_steps=PROMPT_LEN)
            arm = WORKING[et]
            rows.append({"eval": ev, "eval_task": et, "prompt": pe, "prompt_task": pt,
                         "same": pt == et, "arm": arm,
                         "mae": r[arm]["pos_mae_m"],
                         "grip_agree": r[arm]["gripper_agreement"]})
            print("  %-22s eval=%-5s prompt=%-5s %-6s mae=%.4f" %
                  (ev, et, pt, "SAME" if pt == et else "cross", rows[-1]["mae"]))

json.dump(rows, open(OUT / "cond3_results.json", "w"), indent=2)

print("\n================ CONDITION 3 ================")
print("%-24s %-6s  %-12s %-12s %s" % ("eval episode", "task", "SAME-task", "CROSS-task", "delta"))
import statistics as st
alls, allc = [], []
for ev in sorted(val):
    rs = [r for r in rows if r["eval"] == ev]
    s = [r["mae"] for r in rs if r["same"]]
    c = [r["mae"] for r in rs if not r["same"]]
    alls += s; allc += c
    print("%-24s %-6s  %-12.4f %-12.4f %+.4f" %
          (ev, ep2task[ev], st.mean(s), st.mean(c), st.mean(s) - st.mean(c)))
print("-" * 66)
print("%-24s %-6s  %-12.4f %-12.4f %+.4f" %
      ("MEAN", "", st.mean(alls), st.mean(allc), st.mean(alls) - st.mean(allc)))
print("\nsame-task better on %d of %d episodes" %
      (sum(1 for ev in sorted(val)
           if st.mean([r["mae"] for r in rows if r["eval"] == ev and r["same"]])
            < st.mean([r["mae"] for r in rows if r["eval"] == ev and not r["same"]])), len(val)))
print("baseline to beat: repeat-previous-action 0.0065 m")
