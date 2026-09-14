"""Per-channel statistics of the ACTION TENSOR THE LOSS ACTUALLY SEES.

MSE weights every channel equally, so a channel's share of total variance is
its share of the gradient. If translation is a small slice, position accuracy
gets trained proportionally little -- and position error is what we measure.
"""
import numpy as np, torch
from icrt.data.dataset import SequenceDataset
from icrt.util.args import DatasetConfig, SharedConfig

ds = DatasetConfig(dataset_json="/dev/shm/icrt_multitask/dataset_config.json", vision_aug=False,
                   proprio_noise=0.0, action_noise=0.0, rebalance_tasks=False, num_repeat_traj=1,
                   maximum_length=1120, non_overlapping=128)
sh = SharedConfig(seq_length=1024, num_pred_steps=16, rot_6d=True, num_cameras=3, num_arms=2,
                  use_delta_action=True, proprio_extra_dim=1, action_extra_dim=1,
                  gpu_vision_transform=True)
d = SequenceDataset(dataset_config=ds, shared_config=sh, vision_transform=None, split="train")
A = torch.cat([d[i]["action"][:, 0, :21].reshape(-1, 21) for i in range(0, 300, 23)], 0).numpy()

names = (["L dx","L dy","L dz"] + ["L r%d" % i for i in range(6)] + ["L grip"]
       + ["R dx","R dy","R dz"] + ["R r%d" % i for i in range(6)] + ["R grip"] + ["lift"])
v = A.var(0); tot = v.sum()
print("%d timesteps, exactly what the loss sees\n" % A.shape[0])
print("%-8s %9s %9s %13s" % ("channel", "std", "mean", "share of MSE"))
for i, n in enumerate(names):
    print("%-8s %9.5f %9.5f %12.2f%%" % (n, A[:, i].std(), A[:, i].mean(), 100*v[i]/tot))
tr=[0,1,2,10,11,12]; ro=[3,4,5,6,7,8,13,14,15,16,17,18]; gr=[9,19]
print("\n  translation %5.2f%%   rotation %5.2f%%   gripper %5.2f%%   lift %5.2f%%"
      % (100*v[tr].sum()/tot, 100*v[ro].sum()/tot, 100*v[gr].sum()/tot, 100*v[20]/tot))
