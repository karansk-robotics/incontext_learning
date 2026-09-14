# Using the authors' ICRT weights on the AI Worker

A guide to warm-starting our FFW-SG2 training from the released ICRT checkpoint
instead of random initialisation. This is the difference between their setup and
ours that the evidence actually supports (`training_tips.md` §K).

Everything below was verified against the real files, not inferred. Nothing here
has been *run as a training job* yet — the checkpoint surgery and the encoder
load are measured; the result of training from it is not.

---

## 0. Why this and not something else

Their released `run.yaml` shows `decoder_pred_head: mlp`, no action
normalisation, and a gripper carrying 97% of the action variance — all the
things we suspected of breaking our runs, present in their working model. What
they have and we do not is a **warm start**:

```
                    theirs                          ours
pretrained_path     .../pretrain/checkpoint-3.pth   null
vision_encoder      cross-mae-rtx.pth               vit_base_patch16_224.mae
```

**95.7% of our parameters transfer** — 125 tensors, 88.6 M of 92.6 M.

---

## 1. The files

Already downloaded, on `zeux`:

```
~/aiworker_iclr/checkpoints_icrt/
├── icrt_vitb_droid_pretrained/icrt_vitb_droid_pretrained.pth   366 MB  the policy trunk
└── crossmae_rtx/cross-mae-rtx-vitb.pth                         509 MB  the vision encoder
```

The policy `.pth` contains **only** `model` — no `args`, no `epoch`, no
optimizer state. So it is a warm start, not a resume: `--resume` will not work,
`--model-cfg.policy-cfg.pretrained-path` will.

---

## 2. Filter the checkpoint

`misc.load_model` calls `load_state_dict(..., strict=False)`. That tolerates
missing and unexpected keys but **not shape mismatches** — those raise a
`RuntimeError` regardless of `strict`. 24 tensors mismatch, all of them at the
boundary of our retarget:

```
icrt_proprio_encoder.fc1/fc2   (10,10)/(768,10)  ->  (21,21)/(768,21)   21-D proprio
icrt_action_encoder.fc1/fc2    (10,10)/(768,10)  ->  (21,21)/(768,21)   21-D action
icrt_action_decoder.mlp.fc2    (160,128)         ->  (336,128)          16x10 -> 16x21
icrt_attn_pooling.{0,1}.*      width 384         ->  256                2 cameras -> 3
icrt_attn_pooling.2.*          absent            ->  present            our 3rd camera
```

Write `aiworker_icrt/filter_pretrained.py`:

```python
"""Strip the tensors that cannot fit our retargeted model, so the 88.6M that can
will load. See using_icrt_pretrained.md."""
import argparse, torch
from pathlib import Path

# Everything that changed shape when we went 10-D single-arm -> 21-D bimanual,
# or 2 cameras -> 3. Keep icrt_attn_pooling if you train with num_cameras 2.
DROP = ("icrt_proprio_encoder.", "icrt_action_encoder.",
        "icrt_action_decoder.", "icrt_attn_pooling.")

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--keep-pooling", action="store_true",
                help="train with num_cameras 2, so the pooling blocks fit")
a = ap.parse_args()

drop = DROP[:-1] if a.keep_pooling else DROP
sd = torch.load(a.src, map_location="cpu", weights_only=False)["model"]
kept = {k: v for k, v in sd.items() if not k.startswith(drop)}
print("kept %d of %d tensors, %.1fM params"
      % (len(kept), len(sd), sum(v.numel() for v in kept.values()) / 1e6))
for k in sd:
    if k not in kept:
        print("  dropped", k)
torch.save({"model": kept}, a.out)
print("wrote", a.out)
```

Run it:

```bash
cd ~/aiworker_iclr
./.venv/bin/python -m aiworker_icrt.filter_pretrained \
  --src checkpoints_icrt/icrt_vitb_droid_pretrained/icrt_vitb_droid_pretrained.pth \
  --out checkpoints_icrt/icrt_trunk_21d.pth
```

Expect `kept 125 of 149 tensors, 88.6M params`. **If it says anything else, stop
and find out why before training on it.**

---

## 3. Point at the CrossMAE encoder

```
--model-cfg.vision-encoder-cfg.vision-encoder \
    checkpoints_icrt/crossmae_rtx/cross-mae-rtx-vitb.pth
```

### The trap in this one line

`encoders.py:126` branches on the **substring `"cross-mae-rtx"` in the path**:

```python
if "cross-mae-rtx" in name:
    self.model = timm.create_model("vit_base_patch16_224.mae", ...)
    timm.models.load_checkpoint(self.model, name, strict=False)
else:
    self.model = timm.create_model(name, ...)     # treats it as a timm model NAME
```

**Rename the file and it silently stops being a checkpoint.** It falls to the
`else`, timm gets a filesystem path as a model name, and you get either a crash
or — worse — a differently-named file that happens to resolve. Keep
`cross-mae-rtx` in the filename.

The load itself is `strict=False`, which would happily match zero keys and say
nothing. Verified against the real file:

```
tensors in file            347   (includes the discarded MAE decoder)
matching name and shape    148 / 150 timm tensors
params loaded              85.8M / 85.8M
not in file                norm.weight, norm.bias   <- stay at LayerNorm identity init, benign
```

So it does land. Keep `vision_unfreeze_all: false` — theirs was frozen too, and
their README warns unfreezing costs > 2 days on 8×A100.

---

## 4. Optional — drop to 2 cameras and transfer the pooling too

The pooling width is `latent_dim // num_cameras` (`icrt.py:155`), so 768/2 = 384
for their 2 cameras and 768/3 = 256 for our 3. Training with **2 cameras makes
those blocks fit**, transferring ~3 M more parameters — the vision→token adapter,
which is the part that has actually seen robot images.

**No re-conversion needed.** Edit `image_keys` in
`/dev/shm/icrt_multitask/dataset_config.json`:

```json
"image_keys": [
  "observation/image_cam_head",
  "observation/image_cam_wrist_left"
]
```

and pass `--shared-cfg.num-cameras 2`, plus `--keep-pooling` to the filter
script. This also matches their layout — one exterior, one wrist — and drops the
right wrist camera, which is the one you already said to leave out.

Cost: the right wrist view is gone. On `ylw2` the right arm is the working arm,
so consider `cam_wrist_right` instead of `cam_wrist_left` — or accept 3 cameras
and lose the pooling transfer. Either is defensible; decide before the run, not
after.

---

## 5. The training command

Start from `train_v2.sh`, keeping the `task_grouping` fix already in place:

```bash
#!/usr/bin/env bash
set -euo pipefail
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONPATH=third_party/icrt:${PYTHONPATH:-}
exec ./.venv/bin/python third_party/icrt/scripts/train.py \
  --dataset-cfg.dataset-json /dev/shm/icrt_multitask/dataset_config.json \
  --dataset-cfg.maximum-length 1120 --dataset-cfg.non-overlapping 128 \
  --dataset-cfg.num-repeat-traj 2 --dataset-cfg.shuffle-repeat-traj \
  --shared-cfg.scale-action /dev/shm/icrt_multitask/action_stats_v2.json \
  --shared-cfg.gpu-vision-transform --shared-cfg.num-arms 2 --shared-cfg.num-cameras 3 \
  --shared-cfg.proprio-extra-dim 1 --shared-cfg.action-extra-dim 1 \
  --shared-cfg.seq-length 1024 --shared-cfg.batch-size 2 \
  --shared-cfg.num-pred-steps 16 --shared-cfg.save-every 2 \
  --model-cfg.policy-cfg.scratch-llama-config third_party/icrt/config/model_config/custom_transformer.json \
  --model-cfg.policy-cfg.phase pretrain --model-cfg.policy-cfg.no-prompt-loss \
  --model-cfg.policy-cfg.pretrained-path checkpoints_icrt/icrt_trunk_21d.pth \
  --model-cfg.vision-encoder-cfg.vision-encoder checkpoints_icrt/crossmae_rtx/cross-mae-rtx-vitb.pth \
  --trainer-cfg.epochs 25 --trainer-cfg.accum-iter 4 --trainer-cfg.num-workers 8 \
  --optimizer-cfg.warmup-epochs 1.0 --optimizer-cfg.lr 5e-4 \
  --logging-cfg.output-dir runs --logging-cfg.log-name mt21_warmstart
```

Three things changed from `train_v2.sh`: `pretrained-path`, `vision-encoder`,
and `num-repeat-traj 2`. That is more than one variable
(`training_tips.md` §D3) — but the trunk and the encoder are one intervention
(they were pretrained together), and `num_repeat_traj` matches their recipe. If
you want it strictly attributable, drop `num-repeat-traj 2` from this run and add
it to the next.

**Lower the learning rate if it diverges early.** 5e-4 was their *finetune* rate
from a checkpoint, so it should be in range — but a warm start that immediately
climbs past its initial loss means the trunk is being destroyed, not adapted.

---

## 6. Verify it actually loaded — do not skip this

This project's recurring failure is a green run that never exercised the path
(`training_tips.md` §J). A warm start that silently loaded nothing looks exactly
like a normal run.

`train.py:67` prints on load:

```
Finetuning from checkpoints_icrt/icrt_trunk_21d.pth
Load checkpoint checkpoints_icrt/icrt_trunk_21d.pth
Missing keys:  ['icrt_proprio_encoder...', 'icrt_action_encoder...', ...]
Unexpected keys:  []
```

- **`Missing keys` must list only the 24 you dropped.** Anything else — a
  `llama.*` key in particular — means the trunk did not transfer.
- **`Unexpected keys` must be empty.** Non-empty means your filter left
  something our model has no slot for.
- Note `load_model` hides `vision_encoder` and `llama` from the *reported*
  missing list, so a silently-empty trunk load will **not** show up there. Check
  a weight directly if in doubt:

```bash
./.venv/bin/python -c "
import torch
a=torch.load('checkpoints_icrt/icrt_trunk_21d.pth',map_location='cpu',weights_only=False)['model']
b=torch.load('runs/mt21_warmstart/checkpoint-0.pth',map_location='cpu',weights_only=False)['model']
k='llama.layers.0.attention.wq.weight'
print('cosine to source:', torch.nn.functional.cosine_similarity(
      a[k].flatten().float(), b[k].flatten().float(), dim=0).item())"
```

After one epoch this should still be close to 1.0. Near 0 means it trained from
random weights and the load did nothing.

**Expect the first loss to start LOWER than a from-scratch run.** If epoch 0
opens at the same place `mt21_v2` did, the warm start did not take.

---

## 7. Then judge it the same way as everything else

Nothing about a warm start changes how it gets scored:

```bash
PYTHONPATH=third_party/icrt:. ./.venv/bin/python -m aiworker_icrt.validate_checkpoint \
  --dataset /dev/shm/icrt_multitask \
  --checkpoint runs/mt21_warmstart/checkpoint-N.pth \
  --train-yaml runs/mt21_warmstart/run.yaml \
  --out-dir runs/mt21_warmstart/validation --max-steps 40
```

The bar is the **0.0065 m "repeat the previous action" baseline**, not zero
error, and not the training loss — which is not comparable across runs once
`scale_action` is on (`training_tips.md` §B6).

---

## 8. Honest expectations

Their trunk was pretrained on **DROID: a Franka, one arm, 10-D actions, 2
cameras, a different workspace.** We are loading it into a bimanual FFW-SG2 with
a torso lift, and the layers that read proprio and write actions are precisely
the ones that could not come along. What transfers is the transformer's model of
*sequence* — how an action follows from a history of observations — plus a vision
encoder that has seen robot scenes rather than ImageNet.

That is a real prior and a reasonable bet. It is not a guarantee, and a
cross-embodiment warm start can also do nothing. The measurement decides.

What it will **not** fix: there is still no genuinely bimanual data, `ylw2` still
has the left arm frozen in every frame, and condition 3 — the actual in-context
claim — still has never been run.
