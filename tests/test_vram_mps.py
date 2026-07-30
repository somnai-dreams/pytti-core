"""
MPS backend for the vram_usage_mode profiling buckets (Slice 3).

Skipped entirely off-Mac; on MPS these run in the default suite (no
download/gpu/extras markers needed).
"""

from collections import defaultdict

import pytest
import torch

import pytti.device
from pytti import vram_tools

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)


@pytest.fixture
def mps_profiling(monkeypatch):
    """Fresh profiling state on an MPS default device; restored on teardown."""
    monkeypatch.setattr(pytti.device, "_default", torch.device("mps"))
    monkeypatch.setattr(vram_tools, "track_vram", False)
    monkeypatch.setattr(vram_tools, "usage_mode", "Unknown")
    monkeypatch.setattr(vram_tools, "prev_usage", 0)
    monkeypatch.setattr(vram_tools, "usage_dict", defaultdict(lambda: 0))
    monkeypatch.setattr(vram_tools, "usage_frozen", defaultdict(lambda: False))


def test_profiling_gate_enables_on_mps(mps_profiling):
    vram_tools.vram_profiling(True)
    assert vram_tools.track_vram is True


def test_allocated_reports_mps_memory(mps_profiling):
    baseline = vram_tools._allocated()
    keep = torch.ones(16, 1024, 1024, device="mps")  # 64 MB fp32
    assert vram_tools._allocated() - baseline >= keep.numel() * 4
    del keep


def test_usage_mode_records_nonzero_mps_delta(mps_profiling):
    vram_tools.vram_profiling(True)
    vram_tools.reset_vram_usage()

    keep = None
    with vram_tools.vram_usage_mode("Test Allocation"):
        keep = torch.ones(16, 1024, 1024, device="mps")  # 64 MB fp32
        keep.mul_(2)  # touch it so the allocation is definitely live

    # the delta is attributed to the bucket when the mode is exited
    delta = vram_tools.usage_dict["Test Allocation"]
    assert delta >= keep.numel() * 4, f"expected >=64MB recorded, got {delta}"
    del keep
