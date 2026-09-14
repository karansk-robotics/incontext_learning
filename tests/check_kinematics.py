import numpy as np
from aiworker_icrt.kinematics import FFWKinematics, SerialChain, _pose_error
from aiworker_icrt import constants as C

np.random.seed(0)
K = FFWKinematics()
fails = []

def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok: fails.append(name)

# 1. chain structure
check("left chain joints", K.left.joint_names == C.ARM_L_JOINTS, f"n={K.left.n}")
check("right chain joints", K.right.joint_names == C.ARM_R_JOINTS, f"n={K.right.n}")

# 2. zero-pose sanity: arms should be mirrored about y
Tl = K.left.fk(np.zeros(7)); Tr = K.right.fk(np.zeros(7))
print(f"      zero-pose L eef xyz = {np.round(Tl[:3,3],4)}")
print(f"      zero-pose R eef xyz = {np.round(Tr[:3,3],4)}")
check("zero pose mirrored in y",
      np.allclose(Tl[:3,3]*[1,-1,1], Tr[:3,3], atol=1e-9),
      f"dy={Tl[1,3]:.4f} vs {Tr[1,3]:.4f}")

# 3. analytic vs numerical Jacobian
def num_jac(chain, q, eps=1e-6):
    J = np.zeros((6, chain.n))
    T0 = chain.fk(q)
    for i in range(chain.n):
        qp = q.copy(); qp[i] += eps
        J[:, i] = _pose_error(T0, chain.fk(qp)) / eps
    return J

worst = 0.0
for which, chain in (("left", K.left), ("right", K.right)):
    lim = K.limits[which]
    for _ in range(20):
        q = np.random.uniform(lim[:,0], lim[:,1])
        d = np.abs(chain.jacobian(q) - num_jac(chain, q)).max()
        worst = max(worst, d)
check("analytic Jacobian == numerical", worst < 1e-5, f"max abs diff = {worst:.2e}")

# 4. FK -> IK round trip from a perturbed seed
pos_errs, rot_errs, iters, conv = [], [], [], 0
N = 60
for which, arm_sl in (("left", C.SLICE_ARM_L), ("right", C.SLICE_ARM_R)):
    chain = K.left if which == "left" else K.right
    lim = K.limits[which]
    for _ in range(N):
        q_true = np.random.uniform(lim[:,0]*0.8, lim[:,1]*0.8)
        T = chain.fk(q_true)
        seed = np.clip(q_true + np.random.normal(0, 0.25, 7), lim[:,0], lim[:,1])
        q, info = K._ik_arm(which, T, seed)
        e = _pose_error(chain.fk(q), T)
        pos_errs.append(np.linalg.norm(e[:3])); rot_errs.append(np.linalg.norm(e[3:]))
        iters.append(info["iters"]); conv += info["converged"]
pos_errs, rot_errs = np.array(pos_errs), np.array(rot_errs)
check("IK round trip converges", conv == 2*N, f"{conv}/{2*N} converged")
check("IK position error < 0.1mm", pos_errs.max() < 1e-4,
      f"max={pos_errs.max()*1000:.4f}mm mean={pos_errs.mean()*1000:.4f}mm")
check("IK rotation error < 1mrad", rot_errs.max() < 1e-3,
      f"max={rot_errs.max()*1000:.3f}mrad")
print(f"      IK iterations: mean={np.mean(iters):.1f} max={max(iters)}")

# 5. full 22 -> 20 -> 22 round trip
q22 = np.zeros(C.JOINT_DIM)
q22[C.SLICE_ARM_L] = np.random.uniform(K.limits['left'][:,0]*0.7, K.limits['left'][:,1]*0.7)
q22[C.SLICE_ARM_R] = np.random.uniform(K.limits['right'][:,0]*0.7, K.limits['right'][:,1]*0.7)
q22[C.SLICE_GRIP_L] = 0.55; q22[C.SLICE_GRIP_R] = 0.0
q22[C.SLICE_HEAD] = [0.1, -0.2]; q22[C.SLICE_LIFT] = -0.3; q22[C.SLICE_BASE] = [0.1,0.0,0.2]

cart = K.joints_to_cartesian(q22)
check("cartesian dim", cart.shape == (C.CARTESIAN_DIM,), f"shape={cart.shape}")
seed = q22.copy()
seed[C.SLICE_ARM_L] += np.random.normal(0,0.15,7)
seed[C.SLICE_ARM_R] += np.random.normal(0,0.15,7)
q_rt, info = K.cartesian_to_joints(cart, seed)
cart_rt = K.joints_to_cartesian(q_rt)
check("22->20->22 pose preserved", np.abs(cart_rt - cart).max() < 1e-3,
      f"max cart diff={np.abs(cart_rt-cart).max():.2e}")
check("gripper round trip", abs(q_rt[C.SLICE_GRIP_L]-0.55) < 1e-6,
      f"{q_rt[C.SLICE_GRIP_L]:.6f}")
check("uncontrolled dims passthrough",
      np.allclose(q_rt[C.SLICE_HEAD], seed[C.SLICE_HEAD]) and
      q_rt[C.SLICE_LIFT] == seed[C.SLICE_LIFT] and
      np.allclose(q_rt[C.SLICE_BASE], seed[C.SLICE_BASE]))

# 6. batch conversion over an "episode"
ep = np.tile(q22, (500, 1)) + np.random.normal(0, 0.01, (500, C.JOINT_DIM))
import time; t0=time.time(); cb = K.joints_to_cartesian(ep); dt=time.time()-t0
check("batch shape", cb.shape == (500, C.CARTESIAN_DIM), f"shape={cb.shape}")
print(f"      batch FK: 500 steps x 2 arms in {dt*1000:.1f} ms  "
      f"(~{500/dt/1000:.0f}k steps/s)")

# 7. joint limits respected
check("IK respects joint limits",
      np.all(q_rt[C.SLICE_ARM_L] >= K.limits['left'][:,0]-1e-9) and
      np.all(q_rt[C.SLICE_ARM_L] <= K.limits['left'][:,1]+1e-9))

print()
print("FAILURES:", fails if fails else "none")
