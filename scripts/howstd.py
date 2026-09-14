"""Show mean and std being computed, by hand, on 10 real values."""
import numpy as np, h5py, json, torch
from icrt.data.utils import convert_multi_step, convert_delta_action

split = json.load(open('runs/mt21_gpu/train_split.json'))
with h5py.File('/dev/shm/icrt_multitask/ffw_sg2.hdf5','r') as h5:
    ep = [e for e in split if e in h5][0]
    g = h5[ep]
    a = torch.tensor(g['action/cartesian_position'][:], dtype=torch.float64)
    p = torch.tensor(g['observation/cartesian_position'][:], dtype=torch.float64)
    d = np.asarray(convert_delta_action(convert_multi_step(a,16).numpy(),
                                        convert_multi_step(p,16).numpy(), num_arms=2))
D = d.reshape(-1, a.shape[1])
v = D[200:210, 12]            # channel 12 = R dz, ten consecutive samples

print("Ten real values of channel 12 (R dz — right hand up/down), in metres:\n")
for i, x in enumerate(v): print(f"   sample {i+1:2d}   {x:+.6f}")

n = len(v)
mean = sum(v)/n
print(f"\nSTEP 1 — mean = add them up, divide by how many")
print(f"   sum = {sum(v):+.6f}   n = {n}")
print(f"   mean = {sum(v):+.6f} / {n} = {mean:+.6f}")

dev = v - mean
sq  = dev**2
print(f"\nSTEP 2 — how far is each value from the mean, squared")
for i in range(4):
    print(f"   ({v[i]:+.6f} - {mean:+.6f}) = {dev[i]:+.6f}   squared = {sq[i]:.9f}")
print(f"   ... ({n-4} more)")

var = sum(sq)/n
print(f"\nSTEP 3 — average those squares  (this is the VARIANCE)")
print(f"   variance = {sum(sq):.9f} / {n} = {var:.9f}")

print(f"\nSTEP 4 — take the square root  (this is the STD)")
print(f"   std = sqrt({var:.9f}) = {np.sqrt(var):.6f} m  =  {np.sqrt(var)*100:.3f} cm")

print(f"\n   numpy agrees:  mean {v.mean():+.6f}   std {v.std():.6f}")
print(f"\nOver ALL 898,192 samples (not just these 10) the same computation gives:")
print(f"   mean = +0.00256   std = 0.01211 m = 1.21 cm   <- the number in action_stats.json")
