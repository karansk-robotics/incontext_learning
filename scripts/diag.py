"""Why does IK fail on the policy's poses but not on the demonstrator's?

Discriminates: position vs orientation, which arm, and whether the requested
pose is simply far from the demonstration (accuracy) or geometrically awkward.
"""
import sys, numpy as np, h5py, torch
from pathlib import Path
from aiworker_icrt import constants as C
from aiworker_icrt.policy import DualArmICRT

ep, prompt_ep = sys.argv[1], sys.argv[2]
N = 40
pol = DualArmICRT(train_yaml_path=Path('runs/mt21_gpu/run.yaml'),
                  checkpoint_path=Path('runs/mt21_gpu/checkpoint-15.pth'), device='cuda')
with h5py.File('/dev/shm/icrt_multitask/ffw_sg2.hdf5','r') as h5:
    P, E = h5[prompt_ep], h5[ep]
    pol.prompt(images=[{k: P[f'observation/image_{k}'][t] for k in C.CAMERA_KEYS}
                       for t in range(P['observation/joint_position'].shape[0])],
               joint_states=P['observation/joint_position'][:],
               actions_cartesian=P['action/cartesian_position'][:])
    js = E['observation/joint_position'][:N]
    gt = E['action/cartesian_position'][:N]
    rows=[]
    for t in range(N):
        obs={k: E[f'observation/image_{k}'][t] for k in C.CAMERA_KEYS}
        jt, cart, info = pol.step(obs, js[t])
        rows.append((cart, info))

print(f"{ep}  prompted with {prompt_ep}")
print(f"{'arm':6} {'ik_conv':8} {'pos_err':>9} {'rot_err':>9} {'|pred-gt|pos':>13} {'|pred-gt|rot6d':>15}")
for side, sl in (('left', C.SLICE_CART_L), ('right', C.SLICE_CART_R)):
    conv = sum(1 for _,i in rows if i[side]['converged'])
    pe  = np.array([i[side]['pos_err'] for _,i in rows])
    re_ = np.array([i[side]['rot_err'] for _,i in rows])
    pred = np.array([c for c,_ in rows])
    dpos = np.linalg.norm(pred[:,sl][:,C.SLICE_BLOCK_POS]-gt[:,sl][:,C.SLICE_BLOCK_POS],axis=-1)
    drot = np.linalg.norm(pred[:,sl][:,C.SLICE_BLOCK_ROT6D]-gt[:,sl][:,C.SLICE_BLOCK_ROT6D],axis=-1)
    print(f"{side:6} {conv:3d}/{N:<4} {pe.max():9.5f} {re_.max():9.5f} {dpos.mean():13.4f} {drot.mean():15.4f}")

# Is the requested motion smooth, or does it jump?
pred = np.array([c for c,_ in rows])
for side, sl in (('left', C.SLICE_CART_L), ('right', C.SLICE_CART_R)):
    dp_pred = np.linalg.norm(np.diff(pred[:,sl][:,C.SLICE_BLOCK_POS],axis=0),axis=-1)
    dp_gt   = np.linalg.norm(np.diff(gt[:,sl][:,C.SLICE_BLOCK_POS],axis=0),axis=-1)
    dr_pred = np.linalg.norm(np.diff(pred[:,sl][:,C.SLICE_BLOCK_ROT6D],axis=0),axis=-1)
    dr_gt   = np.linalg.norm(np.diff(gt[:,sl][:,C.SLICE_BLOCK_ROT6D],axis=0),axis=-1)
    print(f"  {side:6} per-step  pos pred {dp_pred.mean()*1000:6.2f}mm (max {dp_pred.max()*1000:6.2f})  "
          f"gt {dp_gt.mean()*1000:6.2f}mm (max {dp_gt.max()*1000:6.2f})")
    print(f"  {side:6} per-step  rot pred {dr_pred.mean():6.4f}    (max {dr_pred.max():6.4f})     "
          f"gt {dr_gt.mean():6.4f}    (max {dr_gt.max():6.4f})")
