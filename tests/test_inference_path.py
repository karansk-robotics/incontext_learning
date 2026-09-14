"""Regression tests for the INFERENCE path, which nothing else covers.

This file exists because of a specific near-miss. ICRT's rot_6d
re-orthogonalisation rebuilt the predicted action as

    cat([xyz, b1, b2, gripper])        # a single 10-D arm

which, given our 20-D bimanual action, SILENTLY TRUNCATED it to 10 and
discarded the right arm. No exception at the point of failure; it surfaced two
functions away as a width mismatch in convert_abs_action.

The part worth remembering: that code path is inference-only. `forward()` never
touches it, so a full training run, the forward/backward checks and all 23
other tests in this suite were green while inference was structurally broken.
On the robot the right arm would simply never have moved.

These run on an UNTRAINED model — no checkpoint required. forward_inference is
happy with random weights, and every property asserted here is structural.
"""
import pytest

from conftest import TORCH_KIND

pytestmark = pytest.mark.skipif(
    TORCH_KIND != "real", reason="needs real torch (model construction is GPU-only)")

PER_ARM = 10  # [xyz(3), rot_6d(6), gripper(1)]


@pytest.fixture(scope="module")
def torch_gpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("diffusers")
    if not torch.cuda.is_available():
        pytest.skip("icrt builds the transformer on cuda at construction time")
    return torch


def _infer(torch, num_arms, num_cameras=3, steps=2):
    """Construct a model and run one forward_inference. Returns the action."""
    from test_model_construction import _build
    model, _, _ = _build(torch, num_arms=num_arms, num_cameras=num_cameras)
    model = model.cuda().eval()

    B, T, N = 1, steps, num_cameras
    seq = {
        # float32 in: the adapters are fp32 and cast into the fp16 transformer
        "observation": torch.randn(B, T, N, 3, 224, 224, device="cuda"),
        "proprio": torch.randn(B, T, num_arms * PER_ARM, device="cuda"),
        # action is T-1: there is no action for the last observation
        "action": torch.randn(B, T - 1, num_arms * PER_ARM + 1, device="cuda"),
    }
    with torch.no_grad():
        out = model.forward_inference(seq, start_pos=0)
    return out[0] if isinstance(out, (tuple, list)) else out


def test_inference_action_is_full_width(torch_gpu):
    """THE regression test. Before the per-arm fix this returned width 10 and
    the right arm was gone."""
    a = _infer(torch_gpu, num_arms=2)
    assert a.shape[-1] == 2 * PER_ARM, (
        f"inference returned width {a.shape[-1]}, expected {2 * PER_ARM}. "
        "A width of 10 means the rot_6d re-orthogonalisation truncated to one "
        "arm and the right arm has been silently discarded.")


def test_every_arm_is_reorthonormalised(torch_gpu):
    """The regression head emits 6 unconstrained numbers per rotation, so each
    arm needs its own Gram-Schmidt. Checking only arm 0 would have passed
    against the truncating version."""
    a = _infer(torch_gpu, num_arms=2).float()
    for arm in range(2):
        b1 = a[..., arm * PER_ARM + 3: arm * PER_ARM + 6]
        b2 = a[..., arm * PER_ARM + 6: arm * PER_ARM + 9]
        assert abs(float(b1.norm(dim=-1).mean()) - 1.0) < 1e-3, f"arm {arm} b1 not unit"
        assert abs(float(b2.norm(dim=-1).mean()) - 1.0) < 1e-3, f"arm {arm} b2 not unit"
        assert float((b1 * b2).sum(-1).abs().max()) < 1e-3, f"arm {arm} b1/b2 not orthogonal"


def test_single_arm_inference_unchanged(torch_gpu):
    """num_arms=1 must still produce the upstream 10-wide action."""
    a = _infer(torch_gpu, num_arms=1)
    assert a.shape[-1] == PER_ARM


def test_prediction_horizon_is_preserved(torch_gpu):
    """num_pred_steps actions come back, not one."""
    a = _infer(torch_gpu, num_arms=2)
    assert a.shape[1] == 16, f"expected 16 predicted steps, got {a.shape[1]}"
