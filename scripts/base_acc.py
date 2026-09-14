"""What does a trivial predictor score? Without this, an MAE is just a number.

Three references on the SAME 40 steps of the SAME held-out episodes:
  STATIC   : command the pose you started at, forever.
  PREV     : command the previous frame's pose (a one-step lag).
  MOTION   : how far the arm actually travels -- the scale an error lives on.
"""
import json, numpy as np, h5py
from aiworker_icrt import constants as C

EPS = json.load(open('runs/mt21_gpu/val_split.json'))
N = 40
print(f"{'episode':24} {'arm':6} {'STATIC':>9} {'PREV':>9} {'travel':>9} {'range':>9}")
agg = {}
with h5py.File('/dev/shm/icrt_multitask/ffw_sg2.hdf5','r') as h5:
    for ep in EPS:
        a = h5[ep]['action/cartesian_position'][:N].astype(np.float64)
        for side, sl in (('left', C.SLICE_CART_L), ('right', C.SLICE_CART_R)):
            p = a[:, sl][:, C.SLICE_BLOCK_POS]
            static = np.linalg.norm(p - p[0], axis=-1).mean()
            prev   = np.linalg.norm(p[1:] - p[:-1], axis=-1).mean()
            travel = np.linalg.norm(np.diff(p, axis=0), axis=-1).sum()
            rng    = np.linalg.norm(p.max(0) - p.min(0))
            print(f"{ep:24} {side:6} {static:9.4f} {prev:9.4f} {travel:9.4f} {rng:9.4f}")
            agg.setdefault(side, []).append((static, prev))
print()
for side, v in agg.items():
    v = np.array(v)
    print(f"  mean {side:6} STATIC {v[:,0].mean():.4f} m   PREV {v[:,1].mean():.4f} m")
