import sys, types, numpy as np
# utils.py imports torch only for type hints / unrelated helpers; stub it.
t = types.ModuleType("torch"); t.Tensor = object
t.from_numpy = np.asarray; t.cat = None
sys.modules["torch"] = t
sys.path.insert(0, "/home/dexstro/development/aiworker_iclr/third_party/icrt")
from icrt.data.utils import convert_delta_action, convert_abs_action, rot_mat_to_rot_6d
from scipy.spatial.transform import Rotation

np.random.seed(0)
S, T, NA = 4, 16, 2
def rand_block(n):
    R = Rotation.random(n).as_matrix()
    return np.concatenate([np.random.randn(n,3), rot_mat_to_rot_6d(R), np.random.rand(n,1)], -1)

proprio = np.stack([rand_block(S) for _ in range(NA)], 1).reshape(S,1,NA*10)
proprio = np.repeat(proprio, T, axis=1)
action  = np.stack([rand_block(S*T).reshape(S,T,10) for _ in range(NA)], 2).reshape(S,T,NA*10)

for with_eos in (False, True):
    a = np.concatenate([action, np.zeros((S,T,1))], -1) if with_eos else action
    d = convert_delta_action(a, proprio)
    r = convert_abs_action(d, proprio)
    err = np.abs(r - a).max()
    print(f"eos={with_eos!s:5} num_arms=2  action{a.shape} -> delta{d.shape} -> abs  "
          f"max round-trip err = {err:.2e}  {'PASS' if err < 1e-9 else 'FAIL'}")

# single-arm must still behave exactly as upstream
p1 = proprio[..., :10]; a1 = np.concatenate([action[..., :10], np.zeros((S,T,1))], -1)
err = np.abs(convert_abs_action(convert_delta_action(a1, p1), p1) - a1).max()
print(f"num_arms=1 (upstream path)  max round-trip err = {err:.2e}  {'PASS' if err<1e-9 else 'FAIL'}")

# arms must not bleed into each other: perturb arm 1, arm 0's delta must not move
d0 = convert_delta_action(action, proprio)
pert = proprio.copy(); pert[..., 10:] += 0.5
d1 = convert_delta_action(action, pert)
print(f"arm-0 delta unchanged when arm-1 proprio moves: "
      f"{'PASS' if np.abs(d0[...,:10]-d1[...,:10]).max()<1e-12 else 'FAIL'}  "
      f"(arm-1 delta did change: {np.abs(d0[...,10:]-d1[...,10:]).max():.3f})")
