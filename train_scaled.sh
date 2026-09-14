#!/usr/bin/env bash
# Corrected run: per-channel action normalisation enabled.
# Without it the gripper channel carried 61.8% of the MSE gradient and
# translation 5.2%; standardised, every live channel carries 4.76%.
set -euo pipefail
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONPATH=third_party/icrt:${PYTHONPATH:-}
exec ./.venv/bin/python third_party/icrt/scripts/train.py \
  --dataset-cfg.dataset-json /dev/shm/icrt_multitask/dataset_config.json \
  --dataset-cfg.maximum-length 1120 --dataset-cfg.non-overlapping 128 \
  --shared-cfg.scale-action /dev/shm/icrt_multitask/action_stats.json \
  --shared-cfg.gpu-vision-transform --shared-cfg.num-arms 2 --shared-cfg.num-cameras 3 \
  --shared-cfg.proprio-extra-dim 1 --shared-cfg.action-extra-dim 1 \
  --shared-cfg.seq-length 1024 --shared-cfg.batch-size 2 \
  --shared-cfg.num-pred-steps 16 --shared-cfg.save-every 2 \
  --model-cfg.policy-cfg.scratch-llama-config third_party/icrt/config/model_config/custom_transformer.json \
  --model-cfg.policy-cfg.phase pretrain --model-cfg.policy-cfg.no-prompt-loss \
  --trainer-cfg.epochs 25 --trainer-cfg.accum-iter 4 --trainer-cfg.num-workers 8 \
  --optimizer-cfg.warmup-epochs 1.0 --optimizer-cfg.lr 5e-4 \
  --logging-cfg.output-dir runs --logging-cfg.log-name mt21_scaled
