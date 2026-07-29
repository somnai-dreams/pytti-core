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

from pytti.config.model_names import VQGAN_MODEL_ALIASES, VQGAN_MODEL_NAMES

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


@define(auto_attribs=True)
class AudioFilterConfig:
    variable_name: str = ""
    f_center: float = -1
    f_width: float = -1
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
        validator=_choice(["Unlimited Palette", "Limited Palette", "VQGAN"]),
    )
    vqgan_model: str = field(
        default="sflickr",
        validator=_choice(VQGAN_MODEL_NAMES + list(VQGAN_MODEL_ALIASES)),
    )
    animation_mode: str = field(
        default="off", validator=_choice(["off", "2D", "3D", "Video Source"])
    )

    width: int = 512
    height: int = 512

    steps_per_scene: int = 100
    steps_per_frame: int = 50
    interpolation_steps: int = 0

    learning_rate: float | None = None
    reset_lr_each_frame: bool = True
    seed: int | None = None  # None = a fresh random seed each run
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
    save_every: int = 50

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

    gradient_accumulation_steps: int = 1
    # None = auto (cuda > mps > cpu); or e.g. "cuda:1", "mps", "cpu"
    device: str | None = None

    # version of the local ./config format; used for migration
    config_version: int = CONFIG_VERSION


def register():
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=ConfigSchema)


register()
