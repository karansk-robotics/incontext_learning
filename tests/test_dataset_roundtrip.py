"""Container-only tier: load a converted dataset through ICRT's real SequenceDataset.

This wraps tests/check_dataset_roundtrip.py, which stays runnable standalone.

Unlike the rest of the suite it needs REAL torch + torchvision (the conftest
stub is not enough — SequenceDataset does actual tensor work) and a converted
dataset on disk. Both are absent outside the container, so these skip by default
rather than failing.

    # inside the container, after Phase 2:
    ICRT_DATASET_DIR=/data/icrt/ffw_sg2_mt pytest tests/

Shapes asserted here match what aiworker-iclr-01 verified on 2026-09-14 at
24 episodes. Widths come from aiworker_icrt.constants rather than being
written out here -- they were hardcoded once and went stale when the action
space moved from 20-D to proprio 23 / action 21.
"""
import os
from pathlib import Path

import pytest

from conftest import TORCH_KIND

from aiworker_icrt import constants as C

DATASET_DIR = os.environ.get("ICRT_DATASET_DIR", "")
SEQ_LENGTH = int(os.environ.get("ICRT_SEQ_LENGTH", "32"))

pytestmark = [
    pytest.mark.skipif(TORCH_KIND != "real",
                       reason="needs real torch (stub is not enough for SequenceDataset)"),
    pytest.mark.skipif(not DATASET_DIR,
                       reason="set ICRT_DATASET_DIR=/data/icrt/<name> to run"),
]


@pytest.fixture(scope="module")
def dataset():
    pytest.importorskip("torchvision")
    from torchvision import transforms
    from icrt.data.dataset import SequenceDataset
    from icrt.util.args import DatasetConfig, SharedConfig

    out_dir = Path(DATASET_DIR)
    cfg_json = out_dir / "dataset_config.json"
    if not cfg_json.is_file():
        pytest.skip(f"no dataset_config.json under {out_dir}")

    ds_cfg = DatasetConfig(dataset_json=str(cfg_json), vision_aug=True,
                           proprio_noise=0.0, action_noise=0.0,
                           rebalance_tasks=False, num_repeat_traj=1)
    sh_cfg = SharedConfig(seq_length=SEQ_LENGTH, num_pred_steps=16, rot_6d=True,
                          num_cameras=3, num_arms=2, use_delta_action=True)
    vt = transforms.Compose([
        transforms.Resize(248, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return SequenceDataset(dataset_config=ds_cfg, shared_config=sh_cfg,
                           vision_transform=vt, split="train")


def test_dataset_is_not_empty(dataset):
    """An empty dataset is the documented small-data trap, not an IndexError.

    usable_indices only admits an episode start when seq_length <= task_length,
    where task_length is the SUMMED frames of that task's episodes in the train
    split. Too few episodes per task and no window fits.
    """
    assert len(dataset) > 0, (
        f"dataset empty: seq_length={SEQ_LENGTH} does not fit inside any task's "
        "summed train-split frames. Lower seq_length, add episodes per task, or "
        "check that verb_to_episode.json actually groups them.")


def test_batch_shapes(dataset):
    b = dataset[0]
    assert b["observation"].shape[-3:] == (3, 224, 224)
    assert b["observation"].shape[-4] == 3, "expected 3 cameras"
    # Derived from the canonical layout, never hardcoded: this assertion went
    # stale when the action space moved 20-D -> proprio 23 / action 21, and a
    # correctly converted dataset failed a green-looking suite.
    assert b["proprio"].shape[-1] == C.PROPRIO_DIM, (
        f"proprio is {C.PROPRIO_DIM}-D: arms {C.CARTESIAN_DIM} + "
        f"{'+'.join(C.PROPRIO_EXTRA)}")
    assert b["action"].shape[-1] == C.ACTION_DIM + 1, (
        f"action is {C.ACTION_DIM}-D (arms {C.CARTESIAN_DIM} + "
        f"{'+'.join(C.ACTION_EXTRA)}) plus ICRT's eos channel")
    assert b["proprio"].shape[-2] == b["action"].shape[-2]


def test_no_nans(dataset):
    import torch
    b = dataset[0]
    for k in ("observation", "proprio", "action"):
        assert not torch.isnan(b[k]).any(), f"NaNs in {k}"


def test_grippers_in_range(dataset):
    """Per-arm gripper is the last channel of each 10-D block."""
    b = dataset[0]
    prop = b["proprio"]
    for arm in range(2):
        g = prop[..., arm * 10 + 9]
        assert float(g.min()) >= -1e-6 and float(g.max()) <= 1.0 + 1e-6, (
            f"arm {arm} gripper outside [0,1]: [{float(g.min())}, {float(g.max())}]")


def test_masks_are_binary(dataset):
    b = dataset[0]
    for k in ("prompt_mask", "weight_mask"):
        if k not in b:
            pytest.skip(f"{k} not present in this batch")
        vals = set(b[k].flatten().tolist())
        assert vals <= {0.0, 1.0}, f"{k} not binary: {sorted(vals)[:5]}"
