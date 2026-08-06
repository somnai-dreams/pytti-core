import gc
import hashlib
import urllib.request
from pathlib import Path

import torch
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from tqdm import tqdm

from pytti import (
    clamp_with_grad,
    default_device,
    empty_cache,
    replace_grad,
    vram_usage_mode,
)
from pytti.config.model_names import VQGAN_MODEL_ALIASES, VQGAN_MODEL_NAMES
from pytti.image_models import EMAImage
from pytti.image_models.init_noise import (
    require_white_init,
    validate_spectrum_chroma,
)

VQGAN_MODEL = None
VQGAN_NAME = None
VQGAN_IS_GUMBEL = None

# Self-owned preservation mirror (2026-08-06 rescue: the original hosts were
# decaying — wikiart's eaidata.bmk.sh is dead; every file below was recovered,
# sha256-verified against its provenance, and re-hosted). Revision-pinned for
# immutability; provenance + hashes documented in the repo's manifest.json.
VQGAN_MIRROR_REPO = "Somnai/pytti-vqgan-checkpoints"
VQGAN_MIRROR_REVISION = "debc22c8f285c7734980c464ff992d335dde6e9a"

# model_name -> (ckpt sha256, yaml sha256), from the mirror's manifest.json.
VQGAN_SHA256 = {
    "imagenet": (
        "845a68805098cb666420d5db93df53f3a3b6dd443e6dd85c05759c5b998cd663",
        "00e2c6189926f1d89ecfef73e9598db77981c1982f0555fbade963ffd16143c7",
    ),
    "coco": (
        "106cc20fde571df14afc4349d62218d8213cf47177c682ecaa78a8c28b12f9de",
        "17d0c2d9fda59eccade2a6070d833804b759dc817da42c2268be8cf1eb047676",
    ),
    "wikiart": (
        "bd08bb46301f98be1712bb2be9f8868cea30b53137cd985d4d6e8da8b3e02c36",
        "6e78241d2828ff35b8381839ce249c24a6c468ed429760d33a25e98e7f24499a",
    ),
    "sflickr": (
        "8a8adea3da8dab412675772831370dd948e0aa97bb11a9488a2f328b58dbc929",
        "c27f012996f3f1f02580f063be47fd60bdd8528efb04d7318b2cbe8a86505b1b",
    ),
    "faceshq": (
        "3274bba51bd565c90990bbcf7bb5dad0b9b95a6a77011980afc4c3cfc4d1cad1",
        "726ca97fb266c3c744f26e5aba2e579a51e310eee711a15f22224031dfdd57e2",
    ),
    "openimages": (
        "5cd6c74810ab97e00e942c25403f73afc081e8b19987b31ec0d9ff5b68e7ab14",
        "91110bab325067b37f85d1e75895a837d59c292bae4f3097f80c518714d5caff",
    ),
}


def _mirror_url(model_name: str, ext: str) -> str:
    return (
        f"https://huggingface.co/{VQGAN_MIRROR_REPO}/resolve/"
        f"{VQGAN_MIRROR_REVISION}/{model_name}/{model_name}.{ext}"
    )


VQGAN_CONFIG_URLS = {name: [_mirror_url(name, "yaml")] for name in VQGAN_SHA256}
VQGAN_CHECKPOINT_URLS = {name: [_mirror_url(name, "ckpt")] for name in VQGAN_SHA256}


def _download(url, dest, timeout=60, expected_sha256=None):
    """
    Download url to dest atomically (via a .part temp file, renamed on
    success), so an interrupted download can never leave a truncated file
    that later passes the exists() cache check. With expected_sha256, the
    temp file is hash-verified before the rename — a corrupted or tampered
    download never lands under the cached name.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return True

    req = urllib.request.Request(url, headers={"User-Agent": "pytti-core"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=timeout) as source:
        length = source.info().get("Content-Length")
        total = int(length) if length is not None else None
        logger.info(f"Downloading {url} to {dest}")
        with open(tmp, "wb") as output, tqdm(total=total) as progress:
            while True:
                buffer = source.read(65536)
                if not buffer:
                    break
                output.write(buffer)
                digest.update(buffer)
                progress.update(len(buffer))
    if total is not None and tmp.stat().st_size != total:
        tmp.unlink()
        raise OSError(
            f"Download of {url} was truncated "
            f"({tmp.stat().st_size if tmp.exists() else 0}/{total} bytes)"
        )
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        actual = digest.hexdigest()
        tmp.unlink()
        raise OSError(
            f"Download of {url} failed sha256 verification "
            f"(expected {expected_sha256}, got {actual})"
        )
    tmp.rename(dest)
    return True


def load_vqgan_model(config_path, checkpoint_path):
    """
    Build the frozen VQGAN described by a published taming config + checkpoint
    pair, using the vendored inference-only models (no pytorch-lightning).
    Net2NetTransformer checkpoints load just their first-stage weights.
    """
    from pytti.vendor.taming.vqgan_models import build_vqgan, load_checkpoint_state

    config = OmegaConf.load(config_path)
    model, gumbel = build_vqgan(config)

    state = load_checkpoint_state(checkpoint_path)
    if config.model.target.endswith("Net2NetTransformer"):
        prefix = "first_stage_model."
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    # drop training-only weights (discriminator/LPIPS live under loss.*)
    state = {k: v for k, v in state.items() if not k.startswith("loss.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"VQGAN checkpoint {checkpoint_path} is missing weights: {missing[:5]}..."
        )
    if unexpected:
        logger.debug(f"Ignored {len(unexpected)} training-only checkpoint keys")
    model.eval().requires_grad_(False)
    return model, gumbel


def vector_quantize(x, codebook, fake_grad=True, l2_norm=False):
    """
    Snap each row of x to its nearest codebook row, with a straight-through
    gradient (replace_grad) back to x.

    l2_norm=True is the LlamaGen codebook convention: rows of x AND the
    codebook are L2-normalized before the distance matrix, quantized values
    are the normalized codebook rows, and the straight-through gradient flows
    through F.normalize into the raw latent (matching upstream
    VectorQuantizer.forward, vendor/llamagen/vq_model.py).
    """
    if l2_norm:
        x = F.normalize(x, dim=-1)
        codebook = F.normalize(codebook, dim=-1)
    d = (
        x.pow(2).sum(dim=-1, keepdim=True)
        + codebook.pow(2).sum(dim=1)
        - 2 * x @ codebook.T
    )
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class VQGANImage(EMAImage):
    """
    VQGAN latent image representation
    width:  (positive integer) approximate image width in pixels  (will be rounded down to nearest multiple of 16)
    height: (positive integer) approximate image height in pixels (will be rounded down to nearest multiple of 16)
    model:  (VQGAN) vqgan model
    """

    @vram_usage_mode("VQGAN Image")
    def __init__(
        self, width, height, scale=1, model=None, ema_val=0.99, device=None
    ):
        if device is None:
            device = default_device()
        self.device = device

        if model is None:
            model = VQGAN_MODEL
            if model is None:
                raise RuntimeError(
                    "ERROR: model is None and VQGAN is not initialized loaded"
                )

        is_gumbel = hasattr(model.quantize, "embed")
        if is_gumbel:
            e_dim = 256
            n_toks = model.quantize.n_embed
            vqgan_quantize_embedding = model.quantize.embed.weight
        else:
            e_dim = model.quantize.e_dim
            n_toks = model.quantize.n_e
            vqgan_quantize_embedding = model.quantize.embedding.weight

        f = 2 ** (model.decoder.num_resolutions - 1)
        self.e_dim = e_dim
        self.n_toks = n_toks

        width *= scale
        height *= scale
        # set up parameter dimensions
        toksX, toksY = width // f, height // f
        sideX, sideY = toksX * f, toksY * f
        self.toksX, self.toksY = toksX, toksY

        # we can't use our own vqgan_quantize_embedding yet because the buffer isn't
        # registered, and we can't register the buffer without the value of z

        z = self.rand_latent(vqgan_quantize_embedding=vqgan_quantize_embedding)
        super().__init__(sideX, sideY, z, ema_val)
        self.output_axes = ("n", "s", "y", "x")
        self.lr = 0.15 if is_gumbel else 0.1
        self.latent_strength = 1

        # extract the parts of VQGAN we need
        self.register_buffer(
            "vqgan_quantize_embedding", vqgan_quantize_embedding, persistent=False
        )
        # self.vqgan_quantize_embedding = torch.nn.Parameter(vqgan_quantize_embedding)
        self.vqgan_decode = model.decode
        self.vqgan_encode = model.encode

    def clone(self):
        dummy = VQGANImage(*self.image_shape)
        with torch.no_grad():
            dummy.tensor.set_(self.tensor.clone())
            dummy.accum.set_(self.accum.clone())
            dummy.biased.set_(self.biased.clone())
            dummy.average.set_(self.average.clone())
            dummy.decay = self.decay
        return dummy

    def get_latent_tensor(self, detach=False, device=None):
        if device is None:
            device = self.device
        z = self.tensor
        if detach:
            z = z.detach()
        z_q = vector_quantize(z, self.vqgan_quantize_embedding).movedim(3, 1).to(device)
        return z_q

    @classmethod
    def get_preferred_loss(cls):
        from pytti.LossAug.LatentLossClass import LatentLoss

        return LatentLoss

    def decode(self, z, device=None):
        if device is None:
            device = self.device
        z_q = vector_quantize(z, self.vqgan_quantize_embedding).movedim(3, 1).to(device)
        out = self.vqgan_decode(z_q).add(1).div(2)
        width, height = self.image_shape
        return clamp_with_grad(out, 0, 1)
        # return F.interpolate(clamp_with_grad(out, 0, 1).to(device, memory_format = torch.channels_last), (height, width), mode='nearest')

    @torch.no_grad()
    def encode_image(self, pil_image, device=None, **kwargs):
        if device is None:
            device = self.device
        pil_image = pil_image.resize(self.image_shape, Image.LANCZOS)
        pil_image = TF.to_tensor(pil_image)
        z, *_ = self.vqgan_encode(pil_image.unsqueeze(0).to(device) * 2 - 1)
        self.tensor.set_(z.movedim(1, 3))
        self.reset()

    @torch.no_grad()
    def make_latent(self, pil_image, device=None):
        if device is None:
            device = self.device
        pil_image = pil_image.resize(self.image_shape, Image.LANCZOS)
        pil_image = TF.to_tensor(pil_image)
        z, *_ = self.vqgan_encode(pil_image.unsqueeze(0).to(device) * 2 - 1)
        z_q = (
            vector_quantize(z.movedim(1, 3), self.vqgan_quantize_embedding)
            .movedim(3, 1)
            .to(device)
        )
        return z_q

    @torch.no_grad()
    def encode_random(
        self,
        init_spectrum="white",
        init_spectrum_falloff=1.0,
        init_spectrum_chroma="full",
    ):
        # init_spectrum_chroma only shapes shaped (non-white) inits, and
        # only 'white' passes the guard here — accepted (workhorse always
        # passes it), validated, never consulted
        validate_spectrum_chroma(init_spectrum_chroma)
        require_white_init("VQGANImage", init_spectrum)
        self.tensor.set_(self.rand_latent())
        self.reset()

    def rand_latent(self, device=None, vqgan_quantize_embedding=None):
        if device is None:
            device = self.device
        if vqgan_quantize_embedding is None:
            vqgan_quantize_embedding = self.vqgan_quantize_embedding
        n_toks = self.n_toks
        toksX, toksY = self.toksX, self.toksY
        one_hot = F.one_hot(
            torch.randint(n_toks, [toksY * toksX], device=device), n_toks
        ).float()
        z = one_hot @ vqgan_quantize_embedding
        z = z.view([-1, toksY, toksX, self.e_dim])
        return z

    # Why is this a static method? Make it a regular method and kill the globals.
    @staticmethod
    def init_vqgan(model_name, model_artifacts_path, device=None):
        if device is None:
            device = default_device()
        model_name = VQGAN_MODEL_ALIASES.get(model_name, model_name)
        global VQGAN_MODEL, VQGAN_NAME, VQGAN_IS_GUMBEL  # uh.... fix this nonsense.
        if VQGAN_NAME == model_name:
            return
        if model_name not in VQGAN_MODEL_NAMES:
            raise ValueError(
                f"VQGAN model {model_name} is not supported. Supported models are {VQGAN_MODEL_NAMES}"
            )
        model_artifacts_path = Path(model_artifacts_path)
        logger.info(model_artifacts_path)
        model_artifacts_path.mkdir(parents=True, exist_ok=True)
        vqgan_config = model_artifacts_path / f"{model_name}.yaml"
        vqgan_checkpoint = model_artifacts_path / f"{model_name}.ckpt"
        logger.debug(vqgan_config)
        logger.debug(vqgan_config.absolute())
        logger.debug(vqgan_checkpoint.absolute())
        logger.debug(vqgan_checkpoint)

        if not vqgan_config.exists():
            logger.warning(
                f"WARNING: VQGAN config file {vqgan_config} not found. Initializing download."
            )

            url = VQGAN_CONFIG_URLS[model_name][0]

            if not _download(
                url, vqgan_config, expected_sha256=VQGAN_SHA256[model_name][1]
            ):
                logger.critical(
                    f"ERROR: VQGAN model {model_name} config failed to download! Please contact model host or find a new one."
                )
                raise FileNotFoundError(f"VQGAN {model_name} config not found")
        # if not path_exists(vqgan_checkpoint):
        if not vqgan_checkpoint.exists():
            logger.warning(
                f"WARNING: VQGAN checkpoint file {vqgan_checkpoint} not found. Initializing download."
            )

            url = VQGAN_CHECKPOINT_URLS[model_name][0]

            if not _download(
                url, vqgan_checkpoint, expected_sha256=VQGAN_SHA256[model_name][0]
            ):
                logger.critical(
                    f"ERROR: VQGAN model {model_name} checkpoint failed to download! Please contact model host or find a new one."
                )
                raise FileNotFoundError(f"VQGAN {model_name} checkpoint not found")

        VQGAN_MODEL, VQGAN_IS_GUMBEL = load_vqgan_model(vqgan_config, vqgan_checkpoint)
        with vram_usage_mode("VQGAN"):
            VQGAN_MODEL = VQGAN_MODEL.to(device)
        VQGAN_NAME = model_name

    @staticmethod
    def free_vqgan():
        global VQGAN_MODEL, VQGAN_NAME, VQGAN_IS_GUMBEL
        VQGAN_MODEL = None
        VQGAN_NAME = None
        VQGAN_IS_GUMBEL = None
        gc.collect()
        empty_cache()
