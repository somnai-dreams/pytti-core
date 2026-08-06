"""
MLX still-image engine — phase M2 of docs/mlx-port-plan.md.

Everything between image decode and the post-Adam clamps runs inside one
``mx.compile``'d step function (docs/mlx-m2-seam-map.md). M1's towers and
conversion cache (``pytti.Perceptor.mlx_backend``) are reused via import,
never modified.

``mlx`` is a darwin-only dependency. Submodules import mlx at module level;
importing *this* package stays mlx-free — names are re-exported lazily via
``__getattr__`` (the ``mlx_backend`` pattern) so linux CI never touches mlx.

Image state is a flat params tree: a dict of fp32 ``mx.array`` (shapes
documented in ``image_models``). That mlx-free vocabulary lives here: the
tuples below name which tree entries the optimizer updates — everything
else is a constant buffer that only rides along for save/restore.
"""

# PixelImage tree: Adam-updated params, clamped by pixel_update after each step
PIXEL_TRAINABLE_KEYS = ("value", "tensor", "palette")
# PixelImage tree: constant buffers. "hdr_comp"/"hdr_weight" are present iff
# the torch module was built with hdr_weight != 0 (both or neither).
PIXEL_CONSTANT_KEYS = ("palette_target", "hdr_comp", "hdr_weight", "norm_weight")
# RGBImage tree
RGB_TRAINABLE_KEYS = ("tensor",)
# FourierImage tree: the trainable rfft2 spectrum plus derived constants
# (the 1/f^decay scale grid and lucid's color matrix — non-persistent
# buffers torch-side, recomputed at tree construction).
FOURIER_TRAINABLE_KEYS = ("spectrum_real", "spectrum_imag")
FOURIER_CONSTANT_KEYS = ("spectrum_scale", "color_matrix")

# PixelImage.palette_inertia (image_models/pixel.py:190) — a hardcoded
# class constant, never serialized.
PALETTE_INERTIA = 2.0

# lazy re-exports: name -> submodule (submodules import mlx at module level)
_LAZY_EXPORTS = {
    "AugConfig": "augs",
    "AugParams": "augs",
    "DirectLossPlan": "step",
    "MLXStillEngine": "engine",
    "PromptPlan": "step",
    "StepConfig": "step",
    "apply_augs": "augs",
    "build_step": "step",
    "clamp_with_grad": "image_models",
    "clip_inclusive": "image_models",
    "color_matrices": "augs",
    "draw_aug_params": "augs",
    "geometric_mask_stops": "step",
    "edge_loss": "losses",
    "edges": "losses",
    "erase_keep_mask": "augs",
    "fourier_decode": "image_models",
    "fourier_params_from_state_dict": "image_models",
    "fourier_state_dict_from_params": "image_models",
    "fourier_update": "image_models",
    # augs.grid_sample_border is a documented local copy of the same
    # gather-bilinear recipe (sampler.py was in flight when S4 was built);
    # deliberately not re-exported here — fold onto sampler's at integration
    "grid_sample_border": "sampler",
    "hdr_loss": "image_models",
    "hsv_loss": "losses",
    "loss_forward": "losses",
    "make_adam": "step",
    "make_cutter": "step",
    "mse_loss": "losses",
    "nearest_upsample": "image_models",
    "pad_image": "sampler",
    "palette_loss": "image_models",
    "palette_sort_indices": "image_models",
    "pixel_decode": "image_models",
    "pixel_params_from_state_dict": "image_models",
    "pixel_state_dict_from_params": "image_models",
    "pixel_update": "image_models",
    "pytti_batched": "sampler",
    "pytti_smart": "sampler",
    "reset_adam_state": "step",
    "rgb_decode": "image_models",
    "rgb_params_from_state_dict": "image_models",
    "rgb_state_dict_from_params": "image_models",
    "rgb_to_rgbsv": "losses",
    "rgb_update": "image_models",
    "sort_palette": "image_models",
    "straight_through": "losses",
    "trainable_keys_for": "step",
    "tv_loss": "losses",
    "warp_matrices": "augs",
    "zero_loss": "losses",
}

__all__ = [
    "FOURIER_CONSTANT_KEYS",
    "FOURIER_TRAINABLE_KEYS",
    "PALETTE_INERTIA",
    "PIXEL_CONSTANT_KEYS",
    "PIXEL_TRAINABLE_KEYS",
    "RGB_TRAINABLE_KEYS",
    *sorted(_LAZY_EXPORTS),
]


def __getattr__(name: str):
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
