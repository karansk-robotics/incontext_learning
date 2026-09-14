"""Realistic test: track a smooth trajectory warm-started from the previous
solution -- which is how IK is actually used at inference."""
import numpy as np, time
from aiworker_icrt.kinematics import FFWKinematics, _pose_error
from aiworker_icrt import constants as C
np.random.seed(1)
K = FFWKinematics()

print("--- random-seed IK (stress case, 0.25 rad seed error) ---")
nc=0; errs=[]; rest=0
for which in ("left","right"):
    chain=K.left if which=="left" else K.right; lim=K.limits[which]
    for _ in range(400):
        q_true=np.random.uniform(lim[:,0]*0.8, lim[:,1]*0.8)
        T=chain.fk(q_true)
        seed=np.clip(q_true+np.random.normal(0,0.25,7), lim[:,0], lim[:,1])
        q,info=K._ik_arm(which,T,seed)
        nc += not info["converged"]; rest += info.get("restarts",0)
        errs.append(np.linalg.norm(_pose_error(chain.fk(q),T)[:3]))
errs=np.array(errs)
print(f"non-converged {nc}/800   restarts used {rest}")
print(f"pos err: mean={errs.mean()*1000:.4f}mm  p99={np.percentile(errs,99)*1000:.4f}mm  max={errs.max()*1000:.3f}mm")

print("\n--- trajectory tracking, warm-started (the real use case) ---")
T_steps=500
for which in ("left","right"):
    chain=K.left if which=="left" else K.right; lim=K.limits[which]
    q0=np.clip(K.rest[which]+np.random.normal(0,0.2,7), lim[:,0], lim[:,1])
    # smooth sinusoidal joint-space trajectory -> Cartesian targets
    t=np.linspace(0,4*np.pi,T_steps)
    traj=q0[None,:]+0.3*np.sin(t[:,None]+np.linspace(0,2,7)[None,:])
    traj=np.clip(traj,lim[:,0],lim[:,1])
    targets=chain.fk_batch(traj)
    seed=traj[0].copy(); pe=[]; its=[]; nc=0
    t0=time.time()
    for k in range(T_steps):
        q,info=K._ik_arm(which,targets[k],seed)
        seed=q; nc += not info["converged"]; its.append(info["iters"])
        pe.append(np.linalg.norm(_pose_error(chain.fk(q),targets[k])[:3]))
    dt=time.time()-t0
    pe=np.array(pe)
    print(f"{which:6} non-conv={nc}/{T_steps}  mean_iters={np.mean(its):.1f}  "
          f"pos_err mean={pe.mean()*1e6:.2f}um max={pe.max()*1e6:.2f}um  "
          f"{dt/T_steps*1000:.2f} ms/step ({1/(dt/T_steps):.0f} Hz/arm)")

print("\n--- continuity: no elbow flips between adjacent solutions ---")
chain=K.left; lim=K.limits['left']
q0=np.clip(K.rest['left'],lim[:,0],lim[:,1])
t=np.linspace(0,2*np.pi,200)
traj=np.clip(q0[None,:]+0.3*np.sin(t[:,None]+np.linspace(0,2,7)[None,:]),lim[:,0],lim[:,1])
targets=chain.fk_batch(traj); seed=traj[0].copy(); jumps=[]
for k in range(200):
    q,_=K._ik_arm('left',targets[k],seed); jumps.append(np.abs(q-seed).max()); seed=q
print(f"max single-step joint jump = {max(jumps):.4f} rad "
      f"({max(jumps)/ (1/15) :.2f} rad/s at 15Hz, limit {C.JOINT_VELOCITY_LIMIT})")
