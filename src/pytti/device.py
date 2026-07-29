"""
Single source of truth for torch device selection.

The process-wide default is resolved once (by the CLI entry point, from config)
via `set_default_device`; everything else calls `default_device()` instead of
re-deriving cuda-vs-cpu locally.
"""

from __future__ import annotations

import torch

_default: torch.device | None = None


def best_available_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(spec: str | int | torch.device | None) -> torch.device:
    """
    Resolve a config-supplied device spec.

    None / "" / "auto" -> best available (cuda > mps > cpu).
    int N              -> cuda:N (legacy config format).
    str                -> torch.device(str), e.g. "cuda:1", "mps", "cpu".
    """
    if spec is None or spec == "" or spec == "auto":
        return best_available_device()
    if isinstance(spec, int):
        device = torch.device("cuda", spec)
    else:
        device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Config requests device {device}, but CUDA is not available on this "
            "machine. Set device to null (auto), 'mps', or 'cpu'."
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "Config requests device 'mps', but MPS is not available on this "
            "machine. Set device to null (auto), 'cuda', or 'cpu'."
        )
    return device


def set_default_device(spec: str | int | torch.device | None) -> torch.device:
    global _default
    _default = resolve_device(spec)
    if _default.type == "cuda":
        torch.cuda.set_device(_default)
    return _default


def default_device() -> torch.device:
    global _default
    if _default is None:
        _default = best_available_device()
    return _default


def memory_format_for(device) -> torch.memory_format:
    """
    channels_last speeds up CUDA tensor cores, but MPS autograd breaks on
    channels_last tensors in the ResNet-CLIP backward (view-vs-stride
    RuntimeError), so use it on CUDA only.
    """
    if torch.device(device).type == "cuda":
        return torch.channels_last
    return torch.contiguous_format


def empty_cache() -> None:
    """Release cached allocator memory on whatever backend is active."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
