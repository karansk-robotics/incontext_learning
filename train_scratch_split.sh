#!/usr/bin/env bash
# CONTROL for train_warm.sh: identical in every respect except initialisation.
# Random transformer init + the stock timm ImageNet MAE encoder, on the same
# grasp-split dataset, same action statistics, same task_grouping, same
# hyperparameters, same 21 epochs.
#
# Its only job is to make train_warm.sh attributable. Warm changed three things
# at once against mt21_gpu (warm start, episode split, task_grouping); this
# holds the last two fixed so the difference is the warm start alone.
#
# Result (held-out working-arm MAE, 8 episodes -- see warm_start_result.md):
#   epoch      warm     this
#       4    0.0513   0.0535
#      12    0.0308   0.0618
#      20    0.0308   0.0752    <- warm 59% better; this one OVERFITS from ep4
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
  --trainer-cfg.epochs 25 --trainer-cfg.accum-iter 4 --trainer-cfg.num-workers 8 \
  --optimizer-cfg.warmup-epochs 1.0 --optimizer-cfg.lr 5e-4 \
  --logging-cfg.output-dir runs --logging-cfg.log-name mt21_scratch_split
