"""Container/GPU tier: construct the bimanual ICRT and run one forward pass.

This is an INDEPENDENT check of aiworker-iclr-01's dimension plumbing in
third_party/icrt/icrt/util/model_constructor.py — nothing had ever actually
built the num_arms=2 model before this file.

Needs real torch on a GPU: icrt/models/policy/icrt.py:187 does
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
at construction time, so there is no CPU path. Skips otherwise.

Also needs timm and diffusers:
    pip install timm==1.0.11 diffusers==0.31.0

Vision weights are NOT downloaded — vision_nonpretrained=True. We are testing
dimension plumbing, not representation quality.
"""
import pytest

from conftest import TORCH_KIND

pytestmark = pytest.mark.skipif(
    TORCH_KIND != "real", reason="needs real torch (model construction is GPU-only)")


@pytest.fixture(scope="module")
def torch_gpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("diffusers")
    if not torch.cuda.is_available():
        pytest.skip("icrt.py:187 sets torch.cuda.HalfTensor — needs a visible GPU")
    return torch


def _build(torch_gpu, num_arms, num_cameras=3, rot_6d=True, llama="vittiny"):
    """Construct an ICRT policy. Returns (model, proprio_dim, action_dim)."""
    from pathlib import Path
    from icrt.util.args import (SharedConfig, PolicyConfig, VisionEncoderConfig)
    from icrt.util.model_constructor import (vision_encoder_constructor,
                                             policy_constructor)
    from conftest import ICRT as ICRT_ROOT

    stem = "custom_transformer" if not llama else f"custom_transformer_{llama}"
    cfg_path = ICRT_ROOT / "config" / "model_config" / f"{stem}.json"
    assert cfg_path.is_file(), cfg_path

    shared = SharedConfig(seq_length=4, num_pred_steps=16, rot_6d=rot_6d,
                          num_cameras=num_cameras, num_arms=num_arms,
                          use_delta_action=True, batch_size=1)
    vis_cfg = VisionEncoderConfig(vision_encoder="vit_tiny_patch16_224",
                                  vision_nonpretrained=True)
    pol_cfg = PolicyConfig(load_llama=False, scratch_llama_config=str(cfg_path),
                           separate_camera_adapter=True, pred_action_only=True)

    vision = vision_encoder_constructor(vis_cfg)
    model = policy_constructor(pol_cfg, shared, vision, train=True)

    per_arm = 10 if rot_6d else 8
    return model, num_arms * per_arm, num_arms * per_arm + (1 if rot_6d else 0)


# --------------------------------------------------------------- dimensions --
def test_dual_arm_dims(torch_gpu):
    """num_arms=2, rot_6d -> proprio 20, action 21 (20 + eos)."""
    model, proprio_dim, action_dim = _build(torch_gpu, num_arms=2)
    assert proprio_dim == 20
    assert action_dim == 21
    assert model.proprio_dim == 20
    # pred_action_only strips the eos channel from the decoder side
    assert model.action_dim == 20, "pred_action_only should drop eos: 21 -> 20"
    assert model.icrt_action_encoder.fc1.in_features == 20


def test_single_arm_reproduces_upstream(torch_gpu):
    """num_arms=1 must be bit-identical to upstream ICRT's 10/11."""
    model, proprio_dim, action_dim = _build(torch_gpu, num_arms=1)
    assert (proprio_dim, action_dim) == (10, 11), "upstream single-arm contract broke"
    assert model.proprio_dim == 10
    assert model.action_dim == 10


def test_no_rot6d_path_unchanged(torch_gpu):
    """Without rot_6d there is no eos channel upstream; keep it that way."""
    _, proprio_dim, action_dim = _build(torch_gpu, num_arms=1, rot_6d=False)
    assert (proprio_dim, action_dim) == (8, 8)


# ---------------------------------------------------------- camera adapters --
def test_separate_camera_adapter_divides_exactly(torch_gpu):
    """3 cameras into a 192-dim latent: 192/3 = 64, so the padding branch at
    icrt.py:142 must be a no-op. A non-zero padding would mean the adapters do
    not tile the latent exactly and some channels carry no camera signal."""
    model, _, _ = _build(torch_gpu, num_arms=2, num_cameras=3)
    assert len(model.icrt_attn_pooling) == 3, "expected one adapter per camera"
    assert model.padding == 0, f"latent does not divide by 3: padding={model.padding}"
    assert model.latent_dim % 3 == 0


def test_padding_zero_at_production_width(torch_gpu):
    """The real config is dim=768 (custom_transformer.json): 768/3 = 256 exact."""
    model, _, _ = _build(torch_gpu, num_arms=2, num_cameras=3, llama="")
    assert model.latent_dim == 768
    assert model.padding == 0, f"768/3 should be exact, got padding={model.padding}"


# ---------------------------------------------------------------- forward ----
@pytest.mark.parametrize("llama,width", [("vittiny", 192), ("", 768)])
def test_forward_pass_loss_is_finite(torch_gpu, llama, width):
    """One forward pass on a batch shaped like the real dataloader output.

    Run at both the fast width and the production width (custom_transformer.json,
    dim=768) — the dims are checked at 768 elsewhere, but only an actual forward
    proves the 20/21 plumbing survives the transformer and the decoder heads.
    """
    torch = torch_gpu
    model, _, action_dim = _build(torch_gpu, num_arms=2, num_cameras=3, llama=llama)
    assert model.latent_dim == width
    model = model.cuda()

    B, T, N, S = 1, 4, 3, 16
    batch = {
        'observation': torch.randn(B, T, N, 3, 224, 224, device='cuda'),
        'proprio': torch.randn(B, T, S, 20, device='cuda'),
        'action': torch.randn(B, T, S, action_dim, device='cuda'),
        'prompt_mask': torch.zeros(B, T, dtype=torch.bool, device='cuda'),
        'weight_mask': torch.ones(B, 1, device='cuda'),
    }
    out = model(batch)
    loss = out[0] if isinstance(out, (tuple, list)) else out
    if isinstance(loss, dict):
        loss = loss.get('loss', next(iter(loss.values())))

    assert torch.isfinite(loss).all(), f"non-finite loss: {loss}"
    assert loss.ndim == 0 or loss.numel() == 1, f"expected scalar loss, got {loss.shape}"
