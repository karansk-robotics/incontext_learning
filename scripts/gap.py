"""Is there a systematic gap between OBSERVED state and COMMANDED action?

The policy predicts a delta and adds it to observed proprio. Accuracy is scored
against the commanded action. If commanded systematically leads observed (servo
lag), that gap is baked into the metric as error the policy cannot remove.
"""
import json, numpy as np, h5py
from aiworker_icrt import constants as C

EPS = json.load(open('runs/mt21_gpu/val_split.json'))
print(f"{'episode':24} {'arm':6} {'|act-obs| mean':>15} {'max':>9} {'per-step motion':>16}")
allg = {}
with h5py.File('/dev/shm/icrt_multitask/ffw_sg2.hdf5','r') as h5:
    for ep in EPS:
        g = h5[ep]
        obs = g['observation/cartesian_position'][:40].astype(np.float64)
        act = g['action/cartesian_position'][:40].astype(np.float64)
        for side, sl in (('left', C.SLICE_CART_L), ('right', C.SLICE_CART_R)):
            po, pa = obs[:, sl][:, C.SLICE_BLOCK_POS], act[:, sl][:, C.SLICE_BLOCK_POS]
            d = np.linalg.norm(pa - po, axis=-1)
            step = np.linalg.norm(np.diff(pa, axis=0), axis=-1).mean()
            print(f"{ep:24} {side:6} {d.mean():15.4f} {d.max():9.4f} {step:16.4f}")
            allg.setdefault(side, []).append(d.mean())
print()
for s, v in allg.items():
    print(f"  mean |action - observation|, {s:6}: {np.mean(v):.4f} m")
