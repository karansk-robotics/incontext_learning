#!/usr/bin/env bash
# Warm start: DROID-pretrained ICRT trunk (88.6M of 92.6M params) + their
# CrossMAE encoder, on the grasp-split dataset.
#
# NOTE ON ATTRIBUTION: this changes three things at once vs mt21_gpu -- the warm
# start, the episode split, and task_grouping. The control that isolates the warm
# start is the identical command with --pretrained-path and --vision-encoder
# removed, on the same split dataset.
set -euo pipefail
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONPATH=third_party/icrt:${PYTHONPATH:-}
exec ./.venv/bin/python third_party/icrt/scripts/train.py \
  --dataset-cfg.dataset-json /dev/shm/icrt_multitask_split/dataset_config.json \
  --dataset-cfg.maximum-length 1120 --dataset-cfg.non-overlapping 128 \
  --shared-cfg.scale-action /dev/shm/icrt_multitask_split/action_stats_split_v1.json \
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
  --logging-cfg.output-dir runs --logging-cfg.log-name mt21_warm
