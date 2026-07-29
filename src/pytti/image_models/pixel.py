

# from pytti.ImageGuide import DirectImageGuide
import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from pytti import named_rearrange, replace_grad, vram_usage_mode
from pytti.device import default_device
from pytti.image_models.differentiable_image import DifferentiableImage
from pytti.LossAug.HSVLossClass import HSVLoss


def break_tensor(tensor):
    """
    Given a tensor, break it into a tuple of four tensors:
    the floor of the tensor, the ceiling of the tensor,
    the rounded tensor, and the fractional part of the tensor

    :param tensor: the tensor to be broken down
    :return: 4 tensors:
        - floors: tensor of integer values that are the largest integer less than or equal to the
    corresponding element in the input tensor
        - ceils: tensor of integer values that are the smallest integer greater than or equal to the
    corresponding element in the input tensor
        - rounds: tensor of integer values that are the nearest integer
    """
    floors = tensor.floor().long()
    ceils = tensor.ceil().long()
    rounds = tensor.round().long()
    fracs = tensor - floors
    return floors, ceils, rounds, fracs


class PaletteLoss(nn.Module):
    """Palette normalization"""

    def __init__(self, n_palettes, weight=0.15, device=None):
        super().__init__()
        if device is None:
            device = default_device()
        self.device = device
        self.n_palettes = n_palettes
        self.register_buffer("weight", torch.as_tensor(weight).to(self.device))

    def forward(self, input: DifferentiableImage):
        """
        Given a pixel image, the function returns the mean of the loss of the softmax of the pixel image

        :param input: a PixelImage
        :return: The loss and the loss_raw.
        """
        if isinstance(input, PixelImage):
            tensor = (
                input.tensor.movedim(0, -1)
                .contiguous()
                .view(-1, self.n_palettes)
                .softmax(dim=-1)
            )
            N, n = tensor.shape
            mu = tensor.mean(dim=0, keepdim=True)
            sigma = tensor.std(dim=0, keepdim=True)
            tensor = tensor.sub(mu)
            # SVD
            S = (tensor.transpose(0, 1) @ tensor).div(sigma * sigma.transpose(0, 1) * N)
            # minimize correlation (anticorrelate palettes)
            S.sub_(torch.diag(S.diagonal()))
            loss_raw = S.mean()
            # maximze varience within each palette
            loss_raw.add_(sigma.mul(N).pow(-1).mean())
            return loss_raw * self.weight, loss_raw
        else:
            return 0, 0

    @torch.no_grad()
    def set_weight(self, weight, device=None):
        """
        Set the weight of the layer to the given value

        :param weight: The weight tensor
        :param device: The device to put the weights on
        """
        if device is None:
            device = self.device
        self.weight.set_(torch.as_tensor(weight, device=device))

    def __str__(self):
        return "Palette normalization"


class HdrLoss(nn.Module):
    def __init__(
        self,
        palette_size: int,
        n_palettes: int,
        gamma: float = 2.5,
        weight: float = 0.15,
        device=None,
    ):
        """
        Create a tensor of size (palette_size, n_palettes) and set the first row to be the palette_size
        values raised to the power of gamma

        :param palette_size: The number of colors in the palette
        :param n_palettes: The number of palettes in the warehouse
        :param gamma: The gamma parameter for the power law
        :param weight: The weight of the loss
        :param device: The device to run the model on
        """
        super().__init__()
        if device is None:
            device = default_device()
        self.device = device
        self.register_buffer(
            "comp",
            torch.linspace(0, 1, palette_size)
            .pow(gamma)
            .view(palette_size, 1)
            .repeat(1, n_palettes)
            .to(device),
        )
        self.register_buffer("weight", torch.as_tensor(weight).to(device))

    def forward(self, input: DifferentiableImage):
        """
        Given a Pixelimage and returns the loss.

        :param input: The input image
        :return: The loss and the loss itself.
        """
        if isinstance(input, PixelImage):
            palette = input.sort_palette()
            magic_color = palette.new_tensor([[[0.299, 0.587, 0.114]]])
            color_norms = torch.linalg.vector_norm(
                palette * (magic_color.sqrt()), dim=-1
            )
            loss_raw = F.mse_loss(color_norms, self.comp)
            return loss_raw * self.weight, loss_raw
        else:
            return 0, 0

    @torch.no_grad()
    def set_weight(self, weight, device=None):
        if device is None:
            device = self.device
        self.weight.set_(torch.as_tensor(weight, device=device))

    def __str__(self):
        return "HDR normalization"


def get_closest_color(a, b):
    """
    a: h1 x w1 x 3 pytorch tensor
    b: h2 x w2 x 3 pytorch tensor
    returns: h1 x w1 x 3 pytorch tensor containing the nearest color in b to the corresponding pixels in a"""
    a_flat = a.contiguous().view(1, -1, 3)
    b_flat = b.contiguous().view(-1, 1, 3)
    a_b = torch.norm(a_flat - b_flat, dim=2, keepdim=True)
    index = torch.argmin(a_b, dim=0)
    closest_color = b_flat[index]
    return closest_color.contiguous().view(a.shape)


class PixelImage(DifferentiableImage):
    """
    differentiable image format for pixel art images
    """

    @vram_usage_mode("Limited Palette Image")
    def __init__(
        self,
        width,
        height,
        scale,
        palette_size,
        n_palettes,
        gamma=1,
        hdr_weight=0.5,
        norm_weight=0.1,
        device=None,
    ):
        super().__init__(width * scale, height * scale)
        if device is None:
            device = default_device()
        self.device = device
        self.palette_inertia = 2
        palette = (
            torch.linspace(0, self.palette_inertia, palette_size)
            .pow(gamma)
            .view(palette_size, 1, 1)
            .repeat(1, n_palettes, 3)
        )
        # palette.set_(torch.rand_like(palette)*self.palette_inertia)
        self.palette = nn.Parameter(palette.to(self.device))

        self.palette_size = palette_size
        self.n_palettes = n_palettes
        self.value = nn.Parameter(torch.zeros(height, width).to(self.device))
        self.tensor = nn.Parameter(
            torch.zeros(n_palettes, height, width).to(self.device)
        )
        self.output_axes = ("n", "s", "y", "x")
        self.latent_strength = 0.1
        self.scale = scale
        self.hdr_loss = (
            HdrLoss(palette_size, n_palettes, gamma, hdr_weight)
            if hdr_weight != 0
            else None
        )
        self.loss = PaletteLoss(n_palettes, norm_weight)
        self.register_buffer("palette_target", torch.empty_like(self.palette))
        self.use_palette_target = False

    def clone(self):
        """
        Returns a new PixelImage object with the same parameters as the original, and copies the
        tensor and palette values from the original
        :return: A new PixelImage object with the same parameters as the
        original.
        """
        width, height = self.image_shape
        dummy = PixelImage(
            width // self.scale,
            height // self.scale,
            self.scale,
            self.palette_size,
            self.n_palettes,
            hdr_weight=0 if self.hdr_loss is None else float(self.hdr_loss.weight),
            norm_weight=float(self.loss.weight),
        )
        with torch.no_grad():
            dummy.value.set_(self.value.clone())
            dummy.tensor.set_(self.tensor.clone())
            dummy.palette.set_(self.palette.clone())
            dummy.palette_target.set_(self.palette_target.clone())
            dummy.use_palette_target = self.use_palette_target
        return dummy

    def set_palette_target(self, pil_image):
        """
        If the user provides a palette image, encode it and set it as the palette target

        :param pil_image: A PIL image (this might be wrong... maybe a DifferentiableImage?)
        :return: The return value is a tuple of the form (output, loss).
        """
        if pil_image is None:
            self.use_palette_target = False
            return
        dummy = self.clone()
        dummy.use_palette_target = False
        dummy.encode_image(pil_image)
        with torch.no_grad():
            self.palette_target.set_(dummy.sort_palette())
            self.palette.set_(self.palette_target.clone())
            self.use_palette_target = True

    @torch.no_grad()
    def lock_palette(self, lock=True):
        """
        If lock is True, set the palette_target attribute to the value of the sort_palette method

        :param lock: If True, the palette_target is locked to the current palette, defaults to True
        (optional)
        """
        if lock:
            self.palette_target.set_(self.sort_palette().clone())
        self.use_palette_target = lock

    def image_loss(self):
        """
        If the loss is not None, return it
        :return: A list of losses
        """
        return [x for x in [self.hdr_loss, self.loss] if x is not None]

    def sort_palette(self):
        """
        Given a palette of colors, sort the palette such that the colors are sorted by their brightness
        :return: The palette is being returned.
        """
        if self.use_palette_target:
            return self.palette_target
        palette = (self.palette / self.palette_inertia).clamp_(0, 1)
        # https://alienryderflex.com/hsp.html
        magic_color = palette.new_tensor([[[0.299, 0.587, 0.114]]])
        color_norms = palette.square().mul_(magic_color).sum(dim=-1)
        palette_indices = color_norms.argsort(dim=0).T
        palette = torch.stack(
            [palette[i][:, j] for j, i in enumerate(palette_indices)], dim=1
        )
        return palette

    def get_image_tensor(self):
        return torch.cat([self.value.unsqueeze(0), self.tensor])

    @torch.no_grad()
    def set_image_tensor(self, tensor):
        """
        Set the image tensor to the given tensor

        :param tensor: the tensor to be set
        """
        self.value.set_(tensor[0])
        self.tensor.set_(tensor[1:])

    def decode_tensor(self):
        """
        Given a tensor of shape (batch_size, n_palettes, n_values),
        returns a tensor of shape (batch_size, height, width, 3)
        where each pixel is a color from the palette
        :return: The image with the palette applied.
        """
        width, height = self.image_shape
        palette = self.sort_palette()

        # brightnes values of pixels
        values = self.value.clamp(0, 1) * (self.palette_size - 1)
        value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)
        value_fracs = value_fracs.unsqueeze(-1).unsqueeze(-1)

        palette_weights = self.tensor.movedim(0, 2)
        palettes = F.one_hot(palette_weights.argmax(dim=2), num_classes=self.n_palettes)

        palette_weights = palette_weights.softmax(dim=2).unsqueeze(-1)
        palettes = palettes.unsqueeze(-1)

        colors_disc = palette[value_rounds]
        colors_disc = (colors_disc * palettes).sum(dim=2)
        colors_disc = F.interpolate(
            colors_disc.movedim(2, 0)
            .unsqueeze(0)
            .to(self.device, memory_format=torch.channels_last),
            (height, width),
            mode="nearest",
        )

        colors_cont = (
            palette[value_floors] * (1 - value_fracs) + palette[value_ceils] * value_fracs
        )
        colors_cont = (colors_cont * palette_weights).sum(dim=2)
        colors_cont = F.interpolate(
            colors_cont.movedim(2, 0)
            .unsqueeze(0)
            .to(self.device, memory_format=torch.channels_last),
            (height, width),
            mode="nearest",
        )
        return replace_grad(colors_disc, colors_cont * 0.5 + colors_disc * 0.5)

    @torch.no_grad()
    def render_value_image(self):
        width, height = self.image_shape
        values = self.value.clamp(0, 1).unsqueeze(-1).repeat(1, 1, 3)
        array = np.array(
            values.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8)
        )[:, :, :]
        return Image.fromarray(array).resize((width, height), Image.NEAREST)

    @torch.no_grad()
    def render_palette(self):
        palette = self.sort_palette()
        width, height = self.n_palettes * 16, self.palette_size * 32
        array = np.array(
            palette.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8)
        )[:, :, :]
        return Image.fromarray(array).resize((width, height), Image.NEAREST)

    @torch.no_grad()
    def render_channel(self, palette_i):
        """
        Given a tensor of shape (batch_size, n_palettes, height, width),
        returns a tensor of shape (batch_size, height, width, n_palettes)

        :param palette_i: The index of the channel to render
        :return: The image.
        """
        width, height = self.image_shape
        palette = self.sort_palette()
        palette[:, :palette_i, :] = 0.5
        palette[:, palette_i + 1 :, :] = 0.5

        values = self.value.clamp(0, 1) * (self.palette_size - 1)
        value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)
        value_fracs = value_fracs.unsqueeze(-1).unsqueeze(-1)

        palette_weights = self.tensor.movedim(0, 2)
        # palettes = F.one_hot(palette_weights.argmax(dim=2), num_classes=self.n_palettes)
        palette_weights = palette_weights.softmax(dim=2).unsqueeze(-1)

        colors_cont = (
            palette[value_floors] * (1 - value_fracs) + palette[value_ceils] * value_fracs
        )
        colors_cont = (colors_cont * palette_weights).sum(dim=2)
        colors_cont = F.interpolate(
            colors_cont.movedim(2, 0).unsqueeze(0), (height, width), mode="nearest"
        )

        tensor = named_rearrange(colors_cont, self.output_axes, ("y", "x", "s"))
        array = np.array(
            tensor.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8)
        )[:, :, :]
        return Image.fromarray(array)

    @torch.no_grad()
    def update(self):
        """
        The palette is clamped to the palette inertia, the value is clamped to 1, and the tensor is
        clamped to infinity
        """
        self.palette.clamp_(0, self.palette_inertia)
        self.value.clamp_(0, 1)
        self.tensor.clamp_(0, float("inf"))
        # self.tensor.set_(self.tensor.softmax(dim = 0))

    def encode_image(self, pil_image, smart_encode=True, device=None):
        """
        Encodes the image into a tensor.

        :param pil_image: The image to encode
        :param smart_encode: If True, the palette will be optimized to match the image, defaults to True
        (optional)
        :param device: The device to run the model on
        """
        width, height = self.image_shape
        if device is None:
            device = self.device

        scale = self.scale
        color_ref = pil_image.resize((width // scale, height // scale), Image.LANCZOS)
        color_ref = TF.to_tensor(color_ref).to(device)
        # value_ref = ImageOps.grayscale(color_ref)
        with torch.no_grad():
            # https://alienryderflex.com/hsp.html
            magic_color = self.palette.new_tensor([[[0.299]], [[0.587]], [[0.114]]])
            value_ref = torch.linalg.vector_norm(
                color_ref * (magic_color.sqrt()), dim=0
            )
            self.value.set_(value_ref)

        # no embedder needed without any prompts
        if smart_encode:
            mse = HSVLoss.build("HSV loss", self.image_shape, pil_image=pil_image)

            if self.hdr_loss is not None:
                before_weight = self.hdr_loss.weight.detach()
                self.hdr_loss.set_weight(0.01)
            # no embedder, no prompts... we don't really need an "ImageGuide" class instance here, do we?
            # We could probably optimize a DifferentiableImage object directly here. I guess maybe
            # cutouts gets applied? Wait no, there's no embedder so there's no cutouts, right?
            from pytti.ImageGuide import DirectImageGuide

            guide = DirectImageGuide(
                self, None, optimizer=optim.Adam([self.palette, self.tensor], lr=0.1)
            )
            # why is there a magic number here?
            guide.run_steps(201, [], [], [mse])
            if self.hdr_loss is not None:
                self.hdr_loss.set_weight(before_weight)

    @torch.no_grad()
    def encode_random(self, random_palette=False):
        """
        Sets the value and palette to random values (uniform noise).

        :param random_palette: If True, the palette is initialized to random values, defaults to False
        (optional)
        """
        self.value.uniform_()
        self.tensor.uniform_()
        if random_palette:
            self.palette.uniform_(to=self.palette_inertia)
