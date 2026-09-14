"""Per-dimension magnitude of the DELTA action the model regresses on.

MSE weights every channel equally. If translation deltas are an order of
magnitude smaller than rotation channels, translation receives a proportionally
tiny share of the gradient -- and translation is what position error measures.
"""
import json, numpy as np, h5py
from aiworker_icrt import constants as C
from icrt.data.utils import convert_delta_action

eps = json.load(open('runs/mt21_gpu/train_split.json'))[:40]
D = []
with h5py.File('/dev/shm/icrt_multitask/ffw_sg2.hdf5','r') as h5:
    for ep in eps:
        g = h5[ep]
        a = g['action/cartesian_position'][:].astype(np.float64)
        p = g['observation/cartesian_position'][:].astype(np.float64)
        d = convert_delta_action(a[None, :, :C.CARTESIAN_DIM], p[None, :, :C.CARTESIAN_DIM], num_arms=2)
        D.append(np.asarray(d).reshape(-1, C.CARTESIAN_DIM))
D = np.concatenate(D)
print(f"{len(D)} frames from {len(eps)} episodes\n")
names = (['L dx','L dy','L dz'] + [f'L r{i}' for i in range(6)] + ['L grip']
       + ['R dx','R dy','R dz'] + [f'R r{i}' for i in range(6)] + ['R grip'])
print(f"{'channel':8} {'std':>10} {'mean|.|':>10} {'share of MSE':>13}")
var = D.var(0); tot = var.sum()
for i, n in enumerate(names):
    print(f"{n:8} {D[:,i].std():10.5f} {np.abs(D[:,i]).mean():10.5f} {100*var[i]/tot:12.2f}%")
tr = [0,1,2,10,11,12]; ro = [3,4,5,6,7,8,13,14,15,16,17,18]; gr=[9,19]
print(f"\n  translation (6 ch): {100*var[tr].sum()/tot:5.2f}% of total variance")
print(f"  rotation    (12 ch): {100*var[ro].sum()/tot:5.2f}%")
print(f"  gripper     (2 ch): {100*var[gr].sum()/tot:5.2f}%")
