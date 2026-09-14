"""Do any episodes actually use BOTH arms, or one at a time?"""
import numpy as np, h5py
from aiworker_icrt import constants as C

both = one = none = 0
rows = []
with h5py.File('/dev/shm/icrt_multitask/ffw_sg2.hdf5','r') as h5:
    for ep in sorted(h5.keys()):
        a = h5[ep]['action/cartesian_position'][:].astype(np.float64)
        tr = {}
        for side, sl in (('L', C.SLICE_CART_L), ('R', C.SLICE_CART_R)):
            p = a[:, sl][:, C.SLICE_BLOCK_POS]
            tr[side] = float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum())
        act = [s for s in ('L','R') if tr[s] > 0.05]     # >5 cm of travel = used
        rows.append((ep, tr['L'], tr['R'], ''.join(act) or '-'))
        if len(act)==2: both+=1
        elif len(act)==1: one+=1
        else: none+=1

task = {}
for ep,l,r,a in rows:
    t = ep.rsplit('_episode_',1)[0]
    task.setdefault(t, {'n':0,'L':0,'R':0,'both':0})
    task[t]['n']+=1
    if 'L' in a: task[t]['L']+=1
    if 'R' in a: task[t]['R']+=1
    if a=='LR': task[t]['both']+=1

print(f"{'task':8} {'eps':>5} {'uses L':>8} {'uses R':>8} {'uses BOTH':>10}")
for t,v in task.items():
    print(f"{t:8} {v['n']:5d} {v['L']:8d} {v['R']:8d} {v['both']:10d}")
print(f"\n  episodes using both arms : {both}/{len(rows)}")
print(f"  episodes using one arm   : {one}/{len(rows)}")
print(f"  episodes using neither   : {none}/{len(rows)}")
print("\n  median travel per episode (m):")
for t,v in task.items():
    L=[l for ep,l,r,a in rows if ep.startswith(t)]; R=[r for ep,l,r,a in rows if ep.startswith(t)]
    print(f"    {t:8} L {np.median(L):6.3f}   R {np.median(R):6.3f}")
