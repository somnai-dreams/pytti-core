import pytest
import torch

from pytti.device import best_available_device, resolve_device


def test_cpu_always_resolves():
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_specs_resolve_to_best_available():
    best = best_available_device()
    assert resolve_device(None) == best
    assert resolve_device("") == best
    assert resolve_device("auto") == best


def test_passthrough_of_device_object():
    dev = torch.device("cpu")
    assert resolve_device(dev) == dev


def test_unavailable_backend_fails_loudly():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda:1")
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device(0)  # legacy integer device spec
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="MPS is not available"):
            resolve_device("mps")
