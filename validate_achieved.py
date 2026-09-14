"""Validate the corrected-target run, and recompute the baselines it must beat.

The baselines MUST be recomputed. On the old leader target, "command the pose you
already observe" scored 0.4mm on ylw2 -- not because it was a good policy, but
because action and observation were nearly identical there. That yardstick was
broken in the same way the training target was.

On the achieved target, action[t] = obs[t+1], so both trivial baselines collapse
to the same thing: predict no motion, and be wrong by exactly one step of motion.
That is an honest bar.
"""
import json, sys, time
from pathlib import Path
import numpy as np, h5py
sys.path.insert(0, "third_party/icrt"); sys.path.insert(0, ".")
from aiworker_icrt.evaluate import evaluate

D = Path("/dev/shm/icrt_mt_achieved")
RUN = "mt21_warm_ach"
EPOCHS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [4, 8, 12, 16, 20]
STEPS, PLEN = 40, 300
WORK = {"ylw2": "right", "box": "left", "ecu": "left"}
SL = {"left": slice(0, 3), "right": slice(10, 13)}

v = json.load(open(D / "verb_to_episode.json"))
tr = set(json.load(open(f"runs/{RUN}/train_split.json")))
val = sorted(json.load(open(f"runs/{RUN}/val_split.json")))
short = lambda t: ("ylw2" if "yellow" in t.lower() else "ecu" if "ecu" in t.lower() else "box")
ep2 = {e: short(t) for t, eps in v.items() for e in eps}
prompts = {short(t): sorted(x for x in eps if x in tr)[0] for t, eps in v.items()}

# ---- baselines on the CORRECTED targets ----------------------------------
print("=" * 70)
print("BASELINES, recomputed on the achieved target (%d held-out episodes)" % len(val))
zero, prev = [], []
with h5py.File(D / "ffw_sg2.hdf5", "r") as h5:
    for e in val:
        s = SL[WORK[ep2[e]]]
        a = h5[e]["action/cartesian_position"][:STEPS][:, s]
        o = h5[e]["observation/cartesian_position"][:STEPS][:, s]
        zero.append(np.linalg.norm(o - a, axis=1).mean())          # predict no motion
        pa = np.concatenate([a[:1], a[:-1]])                        # repeat previous action
        prev.append(np.linalg.norm(pa - a, axis=1).mean())
print("  predict no motion (= observed pose)   %7.4f m" % np.mean(zero))
print("  repeat the previous action            %7.4f m" % np.mean(prev))
BAR = min(np.mean(zero), np.mean(prev))
print("  ---> bar to beat: %.4f m" % BAR)

# ---- the model ------------------------------------------------------------
print("=" * 70)
print("HELD-OUT WORKING-ARM MAE, %s" % RUN)
res = {}
for E in EPOCHS:
    ck = Path(f"runs/{RUN}/checkpoint-{E}.pth")
    if not ck.exists():
        print("  ep%-3d MISSING" % E); continue
    t0, maes, ratios = time.time(), [], []
    for e in val:
        t = ep2[e]
        r = evaluate(D, ck, Path(f"runs/{RUN}/run.yaml"), prompts[t], e,
                     Path(f"runs/{RUN}/val/ep{E}"), max_steps=STEPS, prompt_max_steps=PLEN)
        maes.append(r[WORK[t]]["pos_mae_m"])
        z = np.load(Path(f"runs/{RUN}/val/ep{E}") / f"{e}__prompt_{prompts[t]}.npz")
        s = SL[WORK[t]]
        with h5py.File(D / "ffw_sg2.hdf5", "r") as h5:
            o = h5[e]["observation/cartesian_position"][:len(z["pred"])][:, s]
        dp = np.linalg.norm(z["pred"][:, s] - o, axis=1).mean()
        dc = np.linalg.norm(z["gt"][:, s] - o, axis=1).mean()
        ratios.append(dp / max(dc, 1e-9))
    res[E] = (np.mean(maes), np.mean(ratios))
    print("  ep%-3d MAE %.4f m   delta-magnitude ratio %.2fx   (%.0fs)"
          % (E, res[E][0], res[E][1], time.time() - t0))

print("=" * 70)
print("epoch    MAE        vs bar     delta ratio")
for E, (m, r) in res.items():
    print("%5d    %.4f     %+5.1f%%     %.2fx" % (E, m, 100*(m-BAR)/BAR, r))
print("\ndelta ratio near 1.0 = the model's motion magnitude now TRACKS the truth.")
print("On the old targets it was 0.6-2.0x on box/ecu and 54-65x on ylw2 -- a")
print("constant ~22mm emitted regardless of context.")
if res:
    best = min(res, key=lambda k: res[k][0])
    print("\nbest: epoch %d at %.4f m  (bar %.4f m)  -> %s"
          % (best, res[best][0], BAR, "BEATS the bar" if res[best][0] < BAR else "still above the bar"))
json.dump({str(k): v for k, v in res.items()} | {"bar": float(BAR)},
          open(f"runs/{RUN}/validation_summary.json", "w"), indent=2)
