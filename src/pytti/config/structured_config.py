"""
The canonical pytti config schema — the single source of every setting and
its default. assets/default.yaml only selects this schema and the demo
values; presets override on top. Composed into every run via Hydra's
ConfigStore, so unknown keys and invalid values fail at startup with a
message instead of surfacing as AttributeErrors mid-render.
"""

from attrs import define, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from pytti.config.model_names import (
    LLAMAGEN_MODEL_NAMES,
    VQGAN_MODEL_ALIASES,
    VQGAN_MODEL_NAMES,
)

# bump when the set of config keys changes; drives ./config migration
CONFIG_VERSION = 2


def _choice(valid_values):
    def validator(self, attribute, value):
        if value not in valid_values:
            raise ValueError(
                f"{value!r} is not a valid value for {attribute.name}. "
                f"Valid values: {list(valid_values)}"
            )

    return validator


def _init_spectrum_falloff_validator(self, attribute, value):
    # deferred import: keeps this schema module import-light; fires at
    # attrs instantiation (OmegaConf.to_object), long after import time
    from pytti.image_models.init_noise import validate_spectrum_falloff

    validate_spectrum_falloff(value)


def _init_spectrum_chroma_validator(self, attribute, value):
    # deferred import: keeps this schema module import-light; fires at
    # attrs instantiation (OmegaConf.to_object), long after import time
    from pytti.image_models.init_noise import validate_spectrum_chroma

    validate_spectrum_chroma(value)


def _fourier_decay_validator(self, attribute, value):
    # deferred import: keeps this schema module import-light; fires at
    # attrs instantiation (OmegaConf.to_object), long after import time.
    # fourier_parameterization is defined before this field, so it is set
    # when the inert-knob check runs.
    from pytti.image_models.fourier import validate_fourier_decay

    validate_fourier_decay(
        value, fourier_parameterization=self.fourier_parameterization
    )


def _coarse_stages_validator(self, attribute, value):
    # deferred import: keeps this schema module import-light; fires at
    # attrs instantiation (OmegaConf.to_object), long after import time
    from pytti.coarse_to_fine import validate_coarse_stages

    validate_coarse_stages(coarse_to_fine=self.coarse_to_fine, coarse_stages=value)


def _anneal_validator(check_name):
    # deferred import, same pattern as above. Each anneal_* field checks
    # its own bounds/choices AND the inert-knob rule (a non-default value
    # alongside structure_annealing: false is a config lie — the
    # coarse_stages rule); structure_annealing is defined before these
    # fields, so self.structure_annealing is set when they validate.
    def validator(self, attribute, value):
        import pytti.structure_annealing as sa

        getattr(sa, check_name)(
            value, structure_annealing=self.structure_annealing
        )

    return validator


@define(auto_attribs=True)
class AudioFilterConfig:
    variable_name: str = ""
    # None = unset; validated loudly when audio is actually used
    f_center: float | None = None
    f_width: float | None = None
    order: int = 5


@define(auto_attribs=True)
class ConfigSchema:
    #############
    ## Prompts ##
    #############

    scenes: str = MISSING
    scene_prefix: str = ""
    scene_suffix: str = ""

    direct_image_prompts: str = ""
    init_image: str = ""
    direct_init_weight: str = ""
    semantic_init_weight: str = ""

    ###################
    ## Image & model ##
    ###################

    image_model: str = field(
        default="Unlimited Palette",
        validator=_choice(
            ["Unlimited Palette", "Limited Palette", "VQGAN", "LlamaGen"]
        ),
    )
    vqgan_model: str = field(
        default="sflickr",
        validator=_choice(VQGAN_MODEL_NAMES + list(VQGAN_MODEL_ALIASES)),
    )
    # LlamaGen VQ variant: ds16 = the default look (f=16, same latent-grid
    # math as taming f16); ds8 = f=8, 4x the tokens, finer texture
    llamagen_model: str = field(
        default="ds16",
        validator=_choice(LLAMAGEN_MODEL_NAMES),
    )
    animation_mode: str = field(
        default="off", validator=_choice(["off", "2D", "3D", "Video Source"])
    )

    # Init-noise spectrum for the no-init_image start (encode_random; with an
    # init_image this knob is never consulted). white = the historical iid
    # uniform noise, bit-for-bit. gray = mid-gray + one-bit symmetry-breaking
    # noise. pink = FFT-shaped 1/f^alpha-amplitude gaussian noise (natural
    # images live near alpha 1). fractal = multi-octave pyramid noise, a
    # second power-law family with no FFT. Shaped fields match the white
    # init's moments (mean 0.5, std 1/sqrt(12), clamped to [0,1]) on the
    # logical grid, per channel / palette plane. Limited Palette shapes both
    # the value plane and the palette-selection logits; Unlimited Palette
    # shapes RGB. VQGAN/LlamaGen random init is a categorical codebook draw —
    # no spectrum to shape — so any non-white value fails loudly there.
    init_spectrum: str = field(
        default="white",
        validator=_choice(["white", "gray", "pink", "fractal"]),
    )
    # amplitude decay power alpha for init_spectrum=pink: amplitude ~
    # 1/f^alpha, so alpha 1 => the natural ~1/f^2 POWER spectrum; higher =
    # smoother/cloudier, lower = closer to white. Validated to [0, 8]
    # (init_noise.MAX_SPECTRUM_FALLOFF) at compose time.
    init_spectrum_falloff: float = field(
        default=1.0, validator=_init_spectrum_falloff_validator
    )
    # Chroma structure of the shaped inits (pink/fractal). Independent
    # per-channel fields leave low-frequency COLOR blobs that CLIP never
    # cleans up and that steer the final palette. full = the original
    # independent per-channel fields, bit-for-bit (the default — existing
    # seeds stay reproducible). natural = fields drawn in a decorrelated
    # basis and mapped through lucid's ImageNet color matrix: mostly luma,
    # faint chroma. mono = one shaped luminance field broadcast to all
    # channels: full spatial prior, zero chroma. white and gray ignore it
    # (no low-frequency chroma to shape). Limited Palette has no RGB
    # channels at init, so there the knob governs the palette-selection
    # logit planes instead: mono = plain uniform logits (no pre-committed
    # palette regions), natural = shaped logits at reduced amplitude
    # (init_noise.NATURAL_TENSOR_AMPLITUDE), full = original full-strength
    # shaped logits.
    init_spectrum_chroma: str = field(
        default="full", validator=_init_spectrum_chroma_validator
    )

    # Fourier parameterization (Unlimited Palette + torch or mlx_full
    # backend, stills only — anything else fails loud at startup naming
    # the scope): optimize the image as a 1/f-scaled Fourier spectrum instead
    # of raw pixels (distill.pub 2018 / lucid fft_image; Aphantasia's port
    # proved it for CLIP guidance). Low frequencies (composition) move with
    # large image-space amplitude from step one and texture arrives later,
    # BY CONSTRUCTION — the structural counterpart to what coarse_to_fine
    # does with stages. Decode: spectrum x (1/f^fourier_decay) scale ->
    # irfft2 -> lucid's ImageNet color matrix -> sigmoid. init_spectrum
    # must stay white (the Fourier init IS 1/f-shaped already);
    # structure_annealing doesn't compose (pixel-domain planes). Full
    # semantics: src/pytti/image_models/fourier.py.
    fourier_parameterization: bool = False
    # amplitude decay power for the Fourier scale: 1.0 = lucid's
    # natural-image default; lower = closer to pixel behavior, higher =
    # softer, more composition-dominant (Aphantasia's "compositional
    # softness" knob). Shape only: the scale grid is energy-normalized to
    # the decay-1 level (fourier_scale), so any value in range starts
    # near-gray and optimizes at any resolution — higher decay reallocates
    # the optimizer's step budget toward big masses, it does not inflate
    # contrast. Only meaningful with fourier_parameterization: true (any
    # other value alongside false is rejected loudly). Validated to
    # [0.1, 4.0] at compose time.
    fourier_decay: float = field(default=1.0, validator=_fourier_decay_validator)

    width: int = 512
    height: int = 512

    steps_per_scene: int = 100
    steps_per_frame: int = 50
    interpolation_steps: int = 0

    # Convergence auto-stop: end a scene early once the TOTAL loss plateaus;
    # steps_per_scene stays the hard cap. The loss is sampled every 10 steps
    # (one host sync per interval); the scene stops when relative improvement
    # over the trailing auto_stop_window steps falls below
    # auto_stop_threshold. Never fires before one full window of samples past
    # the interpolation ramp; multi-scene runs judge each scene independently.
    auto_stop: bool = False
    # trailing window, in steps, over which loss improvement is judged
    auto_stop_window: int = 50
    # relative improvement over the window below which the scene is converged
    auto_stop_threshold: float = 0.002

    # Phase scheduling: quality-phase behavior over normalized scene time
    # t_hat = step/steps_per_scene (still mode only — fails loud under any
    # animation_mode). One switch, three fixed schedules (the table lives in
    # src/pytti/phase_scheduling.py): the smoothing/TV weight ramps
    # 2x -> 0.5x of its configured value across each scene (structure early,
    # detail late), Limited Palette locks its palette for the final third
    # (clean crystallization), and the direct init-hold weight decays
    # 1x -> 0.5x (anchored start, freer finish). Scales multiply the
    # evaluated weight, so parametric weight expressions keep working.
    phase_scheduling: bool = False

    # Coarse-to-fine still rendering (still mode, single scene — fails loud
    # otherwise): the render runs as a ladder of stages (coarse_stages,
    # below); each stage renders at reduced dims, then the result is
    # bicubic-upscaled and re-encoded into a fresh image rep at the next
    # stage's dims (Limited Palette carries its learned palette across),
    # and the next stage continues with the previous stage's image as a
    # weight-2 direct init hold plus the normal prompts. Frame numbering
    # and backups continue across every stage boundary (non-final-stage
    # frames save upscaled to the full canvas); auto_stop judges each stage
    # independently; phase_scheduling's t_hat spans each stage's own steps.
    coarse_to_fine: bool = False
    # Number of coarse-to-fine stages — the pyramid: compose as a small
    # thumbnail first, then repeatedly scale up and keep rendering. Only
    # meaningful with coarse_to_fine: true (any other value alongside
    # coarse_to_fine: false is rejected loudly). Geometric dims ladder
    # ending at the configured canvas, every non-final stage /8-rounded
    # with a 64px floor: stages=3 -> 1/4, 1/2, 1; stages=4 -> 1/8, 1/4,
    # 1/2, 1. A 512 canvas at stages=3 opens at 128px — SMALLER than the
    # 224px perceptor input, i.e. the full frame is observed at
    # better-than-native resolution while the composition forms; any stage
    # whose canvas short side is <= the largest perceptor input forces
    # cutout_sampler=full for that stage (random crops there would
    # upsample the whole frame anyway). Earlier stages get fewer steps:
    # stages=2 -> 40/60% of steps_per_scene, 3 -> 25/25/50,
    # 4 -> 15/20/25/40 (table in src/pytti/coarse_to_fine.py; needs
    # steps_per_scene >= 3 * coarse_stages).
    coarse_stages: int = field(default=2, validator=_coarse_stages_validator)

    # Structure annealing: periodically re-liquify ONLY the low-frequency
    # band of the image (FFT-masked blend toward shaped noise or toward the
    # image mean) so CLIP gets fresh votes on composition while accumulated
    # mid/high-frequency detail survives — a diffusion-style structure
    # schedule without a denoiser, generalizing coarse_to_fine's single
    # reset (full semantics + decisions: src/pytti/structure_annealing.py).
    # Cycles are evenly spaced through each scene's steps, the last ~35%
    # of steps stay anneal-free (protected tail), and cycle strength
    # decays geometrically from anneal_strength to 0.1x of it. Still mode
    # only; Limited Palette anneals the value plane only (palette +
    # selection logits untouched); Unlimited Palette anneals all channels.
    # Fails loud for VQGAN/LlamaGen (latent state), auto_stop (deliberate
    # loss resets break plateau semantics), optimizer=adamw_sf (a
    # host-side overwrite desynchronizes its Polyak average), and a
    # multi-scene crossfade overlapping the first cycle (the re-liquified
    # band would recompose toward the OUTGOING scene). With
    # coarse_to_fine, cycles run in the FINAL stage only, within that
    # stage's own step budget — and any direct init hold (including the
    # stage's weight-2 hold on the previous stage's image) is RELEASED at
    # the first cycle, since a full-band pull toward a pre-anneal image
    # would cancel the re-liquification.
    structure_annealing: bool = False
    # number of re-liquify events across the run (each needs >= 5 steps of
    # optimization before the next / the protected tail — too-small step
    # budgets fail loud)
    anneal_cycles: int = field(
        default=3, validator=_anneal_validator("validate_anneal_cycles")
    )
    # low-band blend factor at the FIRST cycle, in (0, 1]; later cycles
    # decay geometrically to 0.1x of it at the last. Zero is rejected — a
    # zero-strength cycle is annealing that does nothing (inert-knob rule)
    anneal_strength: float = field(
        default=0.5, validator=_anneal_validator("validate_anneal_strength")
    )
    # fraction of the Nyquist radius below which frequencies re-liquify
    # (raised-cosine edge — hard masks ring; DC/mean is always preserved)
    anneal_band: float = field(
        default=0.15, validator=_anneal_validator("validate_anneal_band")
    )
    # noise = replace the band with shaped pink noise (falloff from
    # init_spectrum_falloff; chroma honors an explicit init_spectrum_chroma
    # of mono/natural, while the back-compat default 'full' maps to mono —
    # repeated per-cycle injection of independent per-channel fields is the
    # measured chroma-leak pathology);
    # blur = decay the band toward the image mean (zero foreign content)
    anneal_source: str = field(
        default="noise", validator=_anneal_validator("validate_anneal_source")
    )

    learning_rate: float | None = None
    reset_lr_each_frame: bool = True
    seed: int | None = None  # None = a fresh random seed each run
    # 40: the modern default ensemble (FARE+SigLIP2) is judged-monotonic in
    # cut count (16: 0.284, 24: 0.291, 40: 0.314 held-out, 2026-08-04) and
    # 40 is the tested-and-loved config. The classic pair held quality at
    # smart@16 if speed matters more than composition.
    cutouts: int = 40
    cut_pow: float = 2
    cutout_border: float = 0.25
    border_mode: str = field(
        default="clamp",
        validator=_choice(["clamp", "mirror", "wrap", "black", "smear"]),
    )

    ##########
    # Camera #
    ##########

    field_of_view: int = 60
    near_plane: int = 1
    far_plane: int = 10000

    #######################
    ### Audioreactivity ###
    #######################

    input_audio: str = ""
    input_audio_offset: float = 0
    # None means no filters (omegaconf cannot ingest attrs list factories)
    input_audio_filters: list[AudioFilterConfig] | None = None

    ######################
    ### Induced Motion ###
    ######################

    #  _2d and _3d only apply to those animation modes
    translate_x: str = "0"
    translate_y: str = "0"
    translate_z_3d: str = "0"
    rotate_3d: str = "[1, 0, 0, 0]"
    rotate_2d: str = "0"
    zoom_x_2d: str = "0"
    zoom_y_2d: str = "0"

    sampling_mode: str = field(
        default="bicubic", validator=_choice(["nearest", "bilinear", "bicubic"])
    )
    infill_mode: str = field(
        default="wrap", validator=_choice(["mirror", "wrap", "black", "smear"])
    )

    pre_animation_steps: int = 50
    lock_camera: bool = True

    #######################
    ### Limited Palette ###
    #######################

    pixel_size: int = 1
    smoothing_weight: float = 0.02
    random_initial_palette: bool = False
    palette_size: int = 5
    palettes: int = 20
    gamma: float = 1
    hdr_weight: float = 0.01
    palette_normalization_weight: float = 0.2
    target_palette: str = ""
    lock_palette: bool = False

    #####################
    ### Stabilization ###
    #####################

    frames_per_second: int = 12

    direct_stabilization_weight: str = ""
    semantic_stabilization_weight: str = ""
    depth_stabilization_weight: str = ""
    edge_stabilization_weight: str = ""
    flow_stabilization_weight: str = ""

    #####################################
    ### animation_mode = Video Source ###
    #####################################

    video_path: str = ""
    frame_stride: int = 1
    reencode_each_frame: bool = True
    flow_long_term_samples: int = 1

    ############
    ### CLIP ###
    ############

    ViTB32: bool = True
    ViTB16: bool = False
    ViTL14: bool = False
    ViTL14_336px: bool = False
    # FARE robust CLIP: cleaner, perceptually-aligned gradients
    FARE4ViTB32: bool = False
    FARE2ViTL14: bool = False
    # SigLIP2: much stronger prompt semantics than the classic towers
    SigLIP2B16: bool = False
    SigLIP2SO400M: bool = False
    RN50: bool = False
    RN101: bool = False
    RN50x4: bool = False
    RN50x16: bool = False
    RN50x64: bool = False

    ###############
    ### Outputs ###
    ###############

    file_namespace: str = "default"
    allow_overwrite: bool = False
    display_every: int = 50
    # 0 = save one frame per animation frame (steps_per_frame). The 0-default
    # keeps saved frames locked to animation frames even when steps_per_frame
    # changes — a manual value is an explicit override.
    save_every: int = 0

    # crossfade saved frames from init_image to the optimized output over the
    # course of the render (requires init_image)
    breath_mode: bool = False

    backups: int = 3
    approximate_vram_usage: bool = False

    # resume from the latest backup in backup/<file_namespace>/
    restore: bool = False

    #################
    ### Model I/O ###
    #################

    # This is where pytti will expect to find model weights.
    # Each model will be assigned a separate subdirectory within this folder
    # If the expected model artifacts are not present, pytti will attempt to download them.
    models_parent_dir: str = "${user_cache:}"

    ##########################
    ### Performance tuning ###
    ##########################

    # smart (default) = designed two-population sampler: full-frame global
    # anchors + stratified detail cuts, sync-free. A/B verdict 2026-07-31:
    # at cutn 16 it BEAT batched@40 on held-out ViT-L/14 adherence
    # (0.375 vs 0.366) at ~2.3x less tower compute. batched = classic's
    # distribution in one grid_sample (wants cutn ~40); classic = the
    # original 2021 per-crop loop, kept as the legacy preset.
    # full = exclusively full vision: EVERY cutout is the full inscribed
    # square (smart's anchor population at n_global == cutn) — no cut
    # smaller than 100% resolution. On a square canvas every view is the
    # SAME crop augmented (augs + noise_fac are the diversity source), so
    # the designed pairing is LOW cutn: 8-16 is the sensible band. With
    # coherence_weighting an all-anchor batch renormalizes to exactly
    # uniform — a harmless no-op (test_coherence_weighting.py::
    # test_all_anchor_batch_is_exactly_uniform).
    cutout_sampler: str = field(
        default="smart", validator=_choice(["classic", "batched", "smart", "full"])
    )
    # Coherence weighting: redistribute per-cutout SEMANTIC-loss weight by
    # view size. With uniform weights ~75% of the semantic gradient pushes
    # each small detail crop toward the FULL prompt — per-patch prompt
    # stuffing, the classic pytti "tapestry" look. On, full-frame anchor
    # cuts (the smart sampler's designed population; any full-inscribed-
    # square draw from batched/classic counts too) weigh 3x and every cut
    # additionally scales by its size fraction, then the weights renormalize
    # to mean 1 AND are rescaled per prompt against its mask weights
    # (mean(|mask| * coh) == mean(|mask|)): gradient DISTRIBUTION shifts
    # toward global composition while every prompt keeps its configured
    # strength — including image-masked prompts. Geometric masks gate via
    # stops, so their interaction remains data-dependent.
    # Designed for cutout_sampler=smart — under batched/classic only the
    # size-scaling half reliably applies (their anchor draws are chance,
    # not designed). Applies on every backend (torch, mlx, mlx_full).
    # Judged A/B (2026-08-03, golden-lp-still @ 200 steps, modern defaults
    # FARE4ViTB32+SigLIP2B16/smart/cutn40/mlx_full, same ensemble both legs):
    # ON beat OFF on BOTH held-out judges — ViT-L/14 0.3064 vs 0.2959
    # (+0.0105, ~2.5x the ±0.004 judge-noise floor) and SigLIP2-SO400M
    # 0.2126 vs 0.1946 (+0.0180); final-frame LPIPS 0.575 (a genuinely
    # different image). Eyeball agrees: OFF is an even-density tapestry,
    # ON has fore/mid/background and a light source.
    coherence_weighting: bool = False
    # adamw_sf = schedule-free AdamW (Polyak-averaged eval iterate)
    optimizer: str = field(default="adam", validator=_choice(["adam", "adamw_sf"]))

    # mlx = run the CLIP towers + prompt-loss reduction on MLX (Apple Metal,
    # fp16; macOS only, classic ViT towers only — docs/mlx-port-plan.md M1).
    # mlx_full = run the ENTIRE still-image step on MLX (M2): still mode,
    # Limited/Unlimited Palette, batched|smart sampler, plain adam, no
    # semantic masks / semantic image prompts / depth — anything else fails
    # loudly at startup naming the backend that supports it.
    # torch = the reference engine, every tower, every platform.
    perceptor_backend: str = field(
        default="torch", validator=_choice(["torch", "mlx", "mlx_full"])
    )

    gradient_accumulation_steps: int = 1
    # None = auto (cuda > mps > cpu); or e.g. "cuda:1", "mps", "cpu"
    device: str | None = None

    # version of the local ./config format; used for migration
    config_version: int = CONFIG_VERSION


def register():
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=ConfigSchema)


register()
