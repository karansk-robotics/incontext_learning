"""Dump predicted vs ground-truth trajectories to JSON for plotting.

Also dumps proprio (the observed state), because the zero-delta baseline IS
proprio -- seeing all three on one axis shows at a glance whether the policy
improves on doing nothing.
"""
import json, sys, numpy as np, h5py
from pathlib import Path
from aiworker_icrt import constants as C
from aiworker_icrt.policy import DualArmICRT

CKPT, EP, PROMPT, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
pol = DualArmICRT(train_yaml_path=Path('runs/mt21_gpu/run.yaml'),
                  checkpoint_path=Path(CKPT), device='cuda')
with h5py.File('/dev/shm/icrt_multitask/ffw_sg2.hdf5','r') as h5:
    P, E = h5[PROMPT], h5[EP]
    T_p = P['observation/joint_position'].shape[0]
    pol.prompt(images=[{k: P[f'observation/image_{k}'][t] for k in C.CAMERA_KEYS}
                       for t in range(T_p)],
               joint_states=P['observation/joint_position'][:],
               actions_cartesian=P['action/cartesian_position'][:])
    js  = E['observation/joint_position'][:N].astype(np.float64)
    gt  = E['action/cartesian_position'][:N].astype(np.float64)
    pro = E['observation/cartesian_position'][:N].astype(np.float64)
    pred = []
    for t in range(N):
        obs = {k: E[f'observation/image_{k}'][t] for k in C.CAMERA_KEYS}
        _jt, cart, _info = pol.step(obs, js[t])
        pred.append(cart)
pred = np.asarray(pred)

def blocks(a):
    out = {}
    for side, sl in (('left', C.SLICE_CART_L), ('right', C.SLICE_CART_R)):
        p = a[:, sl][:, C.SLICE_BLOCK_POS]
        out[side] = {'x': p[:,0].round(5).tolist(), 'y': p[:,1].round(5).tolist(),
                     'z': p[:,2].round(5).tolist(),
                     'grip': a[:, sl][:, C.BLOCK_GRIPPER].round(4).tolist()}
    return out

doc = {'checkpoint': Path(CKPT).name, 'episode': EP, 'prompt': PROMPT,
       'steps': N, 'fps': 15,
       'pred': blocks(pred), 'gt': blocks(gt), 'proprio': blocks(pro)}
name = f"traj_{Path(CKPT).stem}_{EP}.json"
Path(name).write_text(json.dumps(doc))
print("wrote", name)
for side, sl in (('left', C.SLICE_CART_L), ('right', C.SLICE_CART_R)):
    e = np.linalg.norm(pred[:,sl][:,C.SLICE_BLOCK_POS]-gt[:,sl][:,C.SLICE_BLOCK_POS],axis=-1)
    z = np.linalg.norm(pro[:,sl][:,C.SLICE_BLOCK_POS]-gt[:,sl][:,C.SLICE_BLOCK_POS],axis=-1)
    print(f"  {side:6} policy MAE {e.mean():.4f}   zero-delta MAE {z.mean():.4f}")
