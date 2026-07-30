"""
Optional VRAM usage profiling on CUDA or MPS; every entry point no-ops when
profiling is disabled or no accelerator is available.
"""

import functools
import gc
from collections import defaultdict

import torch
from loguru import logger

from pytti.device import default_device, empty_cache

track_vram = False
usage_mode = "Unknown"
prev_usage = 0
usage_dict = defaultdict(lambda: 0)
usage_frozen = defaultdict(lambda: False)


def _allocated() -> int:
    device_type = default_device().type
    if device_type == "cuda":
        return torch.cuda.memory_allocated()
    if device_type == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def _peak_allocated() -> int:
    device_type = default_device().type
    if device_type == "cuda":
        return torch.cuda.max_memory_allocated()
    if device_type == "mps":
        # MPS has no max_memory_allocated; driver-level total (includes the
        # allocator's cached blocks) is the closest available peak proxy.
        return torch.mps.driver_allocated_memory()
    return 0


def vram_profiling(enabled):
    global track_vram
    if enabled and not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        logger.warning("VRAM profiling requires CUDA or MPS; disabling.")
        enabled = False
    track_vram = enabled


def reset_vram_usage():
    global prev_usage, usage_dict, usage_mode, usage_frozen
    if not track_vram:
        return
    if usage_dict:
        logger.warning(
            "VRAM tracking does not work more than once per process; "
            "restart for accurate usage numbers."
        )
    usage_mode = "Unknown"
    prev_usage = _allocated()
    usage_dict = defaultdict(lambda: 0)
    usage_frozen = defaultdict(lambda: False)


def set_usage_mode(new_mode, force_update=False):
    global usage_mode, prev_usage, usage_dict
    if not track_vram:
        return
    if usage_mode != new_mode or force_update:
        if not usage_frozen[usage_mode]:
            gc.collect()
            empty_cache()
            current_usage = _allocated()
            delta = current_usage - prev_usage
            if delta < 0 and usage_mode != "Unknown":
                logger.warning(f"{usage_mode} has negative delta of {delta}")

            usage_dict[usage_mode] += delta
            prev_usage = current_usage
        usage_mode = new_mode


def freeze_vram_usage(mode=None):
    if not track_vram:
        return
    global usage_mode, usage_frozen
    mode = usage_mode if mode is None else mode
    if not usage_frozen[mode]:
        set_usage_mode(usage_mode, force_update=True)
        usage_frozen[mode] = True


class vram_usage_mode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cached_mode = usage_mode
            set_usage_mode(self.mode)
            try:
                return func(*args, **kwargs)
            finally:
                set_usage_mode(cached_mode)

        return wrapper

    def __enter__(self):
        self.cached_mode = usage_mode
        set_usage_mode(self.mode)

    def __exit__(self, type, value, traceback):
        set_usage_mode(self.cached_mode)


def _fmt_bytes(v: float) -> str:
    if v < 1e3:
        return f"{v}B"
    if v < 1e6:
        return f"{v / 1e3:.2f}kB"
    if v < 1e9:
        return f"{v / 1e6:.2f}MB"
    return f"{v / 1e9:.2f}GB"


def print_vram_usage():
    if not track_vram:
        return
    set_usage_mode(usage_mode, force_update=True)
    total = sum(usage_dict.values()) - usage_dict["Unknown"]
    usage_dict["Unknown"] = _allocated() - total
    for k, v in usage_dict.items():
        logger.info(f"{k}: {_fmt_bytes(v)}")
    logger.info(f"Total: {_fmt_bytes(total)}")
    if total != 0:
        overhead = (_peak_allocated() - total) / total
        logger.info(f"Overhead: {overhead * 100:.2f}%")
