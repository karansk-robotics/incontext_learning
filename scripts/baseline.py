"""Control for the safety gate: push the RECORDED actions through the same IK.

If the teleoperator's own trajectories trip a check, that check is measuring the
robot or the solver, not the checkpoint.
"""
import json, sys
import h5py, numpy as np
from aiworker_icrt import constants as C
from aiworker_icrt.kinematics import FFWKinematics

kin = FFWKinematics('assets/ffw_sg2/ffw_sg2_follower.urdf')
eps = json.load(open('runs/mt21_gpu/runs_val.json')) if False else sys.argv[1:]
fps, dt = 15.0, 1/15.0

with h5py.File('/dev/shm/icrt_multitask/ffw_sg2.hdf5','r') as h5:
    for ep in eps:
        g = h5[ep]
        cart = g['action/cartesian_position'][:40]
        jstate = g['observation/joint_position'][:40]
        seed = jstate[0].astype(np.float64).copy()
        joints, nonconv, worst_err = [], 0, 0.0
        for t in range(len(cart)):
            seed[C.SLICE_LIFT] = jstate[t][C.SLICE_LIFT]
            jt, info = kin.action_to_joints(cart[t], seed)
            for s in ('left','right'):
                if not info[s].get('converged', False): nonconv += 1
                worst_err = max(worst_err, info[s]['pos_err'])
            joints.append(jt); seed = jt.copy()
        joints = np.asarray(joints)

        # joint limit headroom
        worst_head, n_lim = np.inf, 0
        for side, sl in (('left', C.SLICE_ARM_L), ('right', C.SLICE_ARM_R)):
            lim = kin.limits[side]; q = joints[:, sl]
            head = np.minimum(q - lim[:,0], lim[:,1] - q)
            n_lim += int((head < 0.02).sum()); worst_head = min(worst_head, float(head.min()))
        # joint velocity vs 4.8*0.8
        worst_vel = 0.0; n_vel = 0
        for sl in (C.SLICE_ARM_L, C.SLICE_ARM_R):
            dq = np.abs(np.diff(joints[:, sl], axis=0))/dt
            cap = 4.8*0.8
            n_vel += int((dq > cap).sum()); worst_vel = max(worst_vel, float(dq.max()/cap))
        print(f"{ep:26} GT-baseline: ik_nonconv={nonconv:3d}/80 worst_ikerr={worst_err:.4f}m  "
              f"limit_viol={n_lim:3d} worst_head={worst_head:+.4f}  vel={worst_vel:.3f}x viol={n_vel}")

        # which joint is pinned?
        for side, sl, names in (('left', C.SLICE_ARM_L, C.ARM_L_JOINTS),
                                ('right', C.SLICE_ARM_R, C.ARM_R_JOINTS)):
            lim = kin.limits[side]; q = joints[:, sl]
            head = np.minimum(q - lim[:,0], lim[:,1] - q)
            for j in range(7):
                if head[:,j].min() < 0.02:
                    print(f"      pinned: {names[j]} head_min={head[:,j].min():+.4f} "
                          f"range=[{lim[j,0]:+.3f},{lim[j,1]:+.3f}] q_range=[{q[:,j].min():+.3f},{q[:,j].max():+.3f}]")
