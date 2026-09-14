"""Shared setup for the regression suite.

Puts the repo root and third_party/icrt on sys.path, and — when torch is
absent — installs the minimal stub that lets ICRT's rotation helpers import.
`icrt.data.utils` pulls torch in at module scope but the functions under test
(convert_delta_action / convert_abs_action / rot_mat_to_rot_6d) are pure numpy.

Inside the container torch is real and the stub is never installed. Do NOT
stub unconditionally: it would shadow the real torch for anything else in the
same pytest process.
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ICRT = ROOT / "third_party" / "icrt"

for _p in (ROOT, ICRT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _ensure_torch() -> str:
    try:
        import torch  # noqa: F401
        return "real"
    except ImportError:
        import numpy as np
        stub = types.ModuleType("torch")
        stub.Tensor = object
        stub.from_numpy = np.asarray
        stub.cat = None
        sys.modules["torch"] = stub
        return "stub"


TORCH_KIND = _ensure_torch()


def pytest_report_header(config):
    return f"torch: {TORCH_KIND}   icrt: {ICRT}"


@pytest.fixture(scope="session")
def K():
    """One FFWKinematics for the whole session — it parses the URDF on init."""
    from aiworker_icrt.kinematics import FFWKinematics
    return FFWKinematics()


@pytest.fixture(scope="session")
def C():
    from aiworker_icrt import constants
    return constants
