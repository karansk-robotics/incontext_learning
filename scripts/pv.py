"""Do the joint-velocity spikes come from the policy, or from IK restarts?

Perturb GROUND TRUTH poses by the same magnitude the policy is off by. If the
joint velocities spike too, the spikes are a solver artifact (elbow flips on
restart), not a property of the checkpoint.
"""
import numpy as np, h5py
from aiworker_icrt import constants as C
from aiworker_icrt.kinematics import FFWKinematics

kin = FFWKinematics('assets/ffw_sg2/ffw_sg2_follower.urdf')
rng = np.random.default_rng(0)
dt, CAP = 1/15.0, 4.8*0.8

with h5py.File('/dev/shm/icrt_multitask/ffw_sg2.hdf5','r') as h5:
    g = h5['ecu_episode_000017']
    cart = g['action/cartesian_position'][:40].astype(np.float64)
    js   = g['observation/joint_position'][:40].astype(np.float64)

for mag in (0.0, 0.0036, 0.045):
    seed = js[0].copy(); J = []
    for t in range(40):
        c = cart[t].copy()
        if mag > 0:
            for sl in (C.SLICE_CART_L, C.SLICE_CART_R):
                d = rng.normal(size=3); d /= np.linalg.norm(d)
                c[sl][C.SLICE_BLOCK_POS] = c[sl][C.SLICE_BLOCK_POS] + d*mag
        seed[C.SLICE_LIFT] = js[t][C.SLICE_LIFT]
        jt, _ = kin.action_to_joints(c, seed)
        J.append(jt); seed = jt.copy()
    J = np.asarray(J)
    worst = 0.0; n = 0
    for sl in (C.SLICE_ARM_L, C.SLICE_ARM_R):
        dq = np.abs(np.diff(J[:, sl], axis=0))/dt
        worst = max(worst, float(dq.max()/CAP)); n += int((dq > CAP).sum())
    print(f"  GT perturbed by {mag*1000:6.2f} mm -> joint vel worst {worst:5.2f}x cap, {n} violations")

