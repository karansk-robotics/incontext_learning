"""Show the real spread of two channels, not just their std."""
import numpy as np, h5py, json, torch
from icrt.data.utils import convert_multi_step, convert_delta_action
from aiworker_icrt import constants as C

chunks=[]
split=json.load(open('runs/mt21_gpu/train_split.json'))
with h5py.File('/dev/shm/icrt_multitask/ffw_sg2.hdf5','r') as h5:
    for ep in [e for e in split if e in h5]:
        g=h5[ep]
        a=torch.tensor(g['action/cartesian_position'][:],dtype=torch.float64)
        p=torch.tensor(g['observation/cartesian_position'][:],dtype=torch.float64)
        d=convert_delta_action(convert_multi_step(a,16).numpy(),
                               convert_multi_step(p,16).numpy(),num_arms=2)
        chunks.append(np.asarray(d).reshape(-1,a.shape[1]))
A=np.concatenate(chunks)
print(f"{A.shape[0]:,} samples\n")
for name,i,unit,scale in [("R dz  (right hand height)",12,"cm",100),
                          ("L grip (left gripper)",9,"",1)]:
    v=A[:,i]*scale
    q=np.percentile(np.abs(v),[50,68,90,99])
    print(f"{name}")
    print(f"   std                    {v.std():8.3f} {unit}")
    print(f"   median |value|         {q[0]:8.3f} {unit}")
    print(f"   68% of values within   {q[1]:8.3f} {unit}   <- this is what 'std' captures")
    print(f"   90% within             {q[2]:8.3f} {unit}")
    print(f"   99% within             {q[3]:8.3f} {unit}")
    print(f"   full range             {v.min():8.3f} .. {v.max():.3f} {unit}\n")
