"""Is a 3.6 mm error enough to break our DLS IK on a GT pose?

The right arm's predicted poses sit 3.6 mm from ground truth, move only
0.95 mm/step -- and still fail IK 35/40 times, while the GT poses converge
0/80. Either a few mm of perturbation genuinely breaks this solver, or the
predicted poses are unreachable for some other reason. This distinguishes them.
"""
import numpy as np, h5py
from aiworker_icrt import constants as C
from aiworker_icrt.kinematics import FFWKinematics

kin = FFWKinematics('assets/ffw_sg2/ffw_sg2_follower.urdf')
rng = np.random.default_rng(0)
EP = 'ecu_episode_000017'

with h5py.File('/dev/shm/icrt_multitask/ffw_sg2.hdf5','r') as h5:
    g = h5[EP]
    cart = g['action/cartesian_position'][:40].astype(np.float64)
    js   = g['observation/joint_position'][:40].astype(np.float64)

for mag in (0.0, 0.001, 0.0036, 0.010, 0.045):
    seed = js[0].copy(); conv = 0; errs = []
    for t in range(40):
        c = cart[t].copy()
        if mag > 0:
            for sl in (C.SLICE_CART_L, C.SLICE_CART_R):
                d = rng.normal(size=3); d /= np.linalg.norm(d)
                c[sl][C.SLICE_BLOCK_POS] = c[sl][C.SLICE_BLOCK_POS] + d*mag
        seed[C.SLICE_LIFT] = js[t][C.SLICE_LIFT]
        jt, info = kin.action_to_joints(c, seed)
        for s in ('left','right'):
            conv += int(info[s]['converged']); errs.append(info[s]['pos_err'])
        seed = jt.copy()
    print(f"  perturbation {mag*1000:6.2f} mm -> converged {conv:3d}/80   worst_resid {max(errs)*1000:7.3f} mm")

# Control: are the GT poses themselves exactly on the reachable manifold?
print("\n  FK round-trip of the recorded joints (should be ~0):")
seed = js[0].copy(); e = []
for t in range(40):
    c_fk = kin.joints_to_action(js[t])
    seed[C.SLICE_LIFT] = js[t][C.SLICE_LIFT]
    jt, info = kin.action_to_joints(c_fk, seed)
    e.append(max(info['left']['pos_err'], info['right']['pos_err']))
    seed = jt.copy()
print(f"    worst {max(e)*1000:.4f} mm")
