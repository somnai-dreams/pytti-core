import math

import kornia.augmentation as K
import torch
from torch import nn


def pytti_classic():
    return nn.Sequential(
        K.RandomHorizontalFlip(p=0.3),
        K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
        K.RandomPerspective(
            0.2,
            p=0.4,
        ),
        K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        K.RandomErasing(
            scale=(0.1, 0.4), ratio=(0.3, 1 / 0.3), same_on_batch=False, p=0.7
        ),
        nn.Identity(),
    )


class BatchedAugs(nn.Module):
    """
    Sync-free reimplementation of the pytti_classic aug stack. Same ops with
    the same sampling ranges and per-sample probabilities, but flip, affine,
    and perspective compose into ONE projective grid_sample (kornia runs
    three separate warps), color jitter is a per-sample 3x3 color matrix
    (magnitudes here are tiny: hue 0.01, sat 0.01 — first-order exact), and
    erasing is an arange-comparison mask. Every parameter is sampled on the
    device; no .item()/host round-trips anywhere. Measured ~10x the kornia
    stack on MPS at [40, 3, 224, 224].

    Deliberate delta from kornia (legacy is a preset, not the reference):
    the composed warp resolves out-of-frame samples with border replication
    for all three ops, where kornia's perspective filled with black wedges.
    """

    def __init__(self, p_flip=0.3, degrees=30.0, translate=0.1, p_affine=0.8,
                 distortion=0.2, p_persp=0.4, hue=0.01, sat=0.01, p_jitter=0.7,
                 erase_scale=(0.1, 0.4), erase_ratio=(0.3, 1 / 0.3), p_erase=0.7):
        super().__init__()
        self.p_flip = p_flip
        self.degrees = degrees
        self.translate = translate
        self.p_affine = p_affine
        self.distortion = distortion
        self.p_persp = p_persp
        self.hue = hue
        self.sat = sat
        self.p_jitter = p_jitter
        self.erase_scale = erase_scale
        self.erase_ratio = erase_ratio
        self.p_erase = p_erase

    def _warp_matrices(self, n, device, dtype):
        """Inverse (output->input) projective matrices [n, 3, 3], normalized
        [-1, 1] coordinates."""
        eye = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3)

        # horizontal flip: its own inverse
        flip_vec = torch.where(
            torch.rand(n, 1, device=device, dtype=dtype) < self.p_flip,
            torch.tensor([-1.0, 1.0, 1.0], device=device, dtype=dtype),
            torch.ones(3, device=device, dtype=dtype),
        )  # [n, 3]
        m_flip = torch.diag_embed(flip_vec)

        # affine: rotation by U(-deg, deg) + translation U(-t, t); inverse =
        # rotate(-theta) then untranslate
        theta = (torch.rand(n, device=device, dtype=dtype) * 2 - 1) * (
            self.degrees * math.pi / 180
        )
        tx = (torch.rand(n, device=device, dtype=dtype) * 2 - 1) * (2 * self.translate)
        ty = (torch.rand(n, device=device, dtype=dtype) * 2 - 1) * (2 * self.translate)
        cos, sin = torch.cos(-theta), torch.sin(-theta)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        m_aff = torch.stack(
            [
                torch.stack([cos, -sin, -(cos * tx - sin * ty)], -1),
                torch.stack([sin, cos, -(sin * tx + cos * ty)], -1),
                torch.stack([zeros, zeros, ones], -1),
            ],
            -2,
        )
        keep = torch.rand(n, 1, 1, device=device, dtype=dtype) >= self.p_affine
        m_aff = torch.where(keep, eye, m_aff)

        # perspective: pull the two top (or bottom) corners inward like
        # kornia's four-corner jitter; a rank-1 projective perturbation of
        # comparable magnitude, invertible in closed form
        px = (torch.rand(n, device=device, dtype=dtype) * 2 - 1) * (self.distortion / 2)
        py = (torch.rand(n, device=device, dtype=dtype) * 2 - 1) * (self.distortion / 2)
        m_persp = torch.stack(
            [
                torch.stack([ones, zeros, zeros], -1),
                torch.stack([zeros, ones, zeros], -1),
                torch.stack([px, py, ones], -1),
            ],
            -2,
        )
        keep_p = torch.rand(n, 1, 1, device=device, dtype=dtype) >= self.p_persp
        m_persp = torch.where(keep_p, eye, m_persp)

        # forward order flip -> affine -> perspective; inverse composes
        # in the same order with each op inverted
        return m_flip @ m_aff @ m_persp

    def _color_matrices(self, n, device, dtype):
        """Per-sample 3x3 color matrices: saturation lerp toward luma plus a
        small hue rotation around the gray axis."""
        eye = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3)
        luma = torch.tensor(
            [[0.299, 0.587, 0.114]] * 3, device=device, dtype=dtype
        )
        s = 1 + (torch.rand(n, 1, 1, device=device, dtype=dtype) * 2 - 1) * self.sat
        m_sat = s * eye + (1 - s) * luma

        # hue rotation by angle h*2pi around the RGB gray axis
        # (Rodrigues' formula with unit axis (1,1,1)/sqrt(3))
        ang = (torch.rand(n, device=device, dtype=dtype) * 2 - 1) * (
            self.hue * 2 * math.pi
        )
        c, si = torch.cos(ang), torch.sin(ang)
        k = 1.0 / 3.0
        rt3 = 1.0 / math.sqrt(3.0)
        cc = 1 - c
        m_hue = torch.stack(
            [
                torch.stack([c + cc * k, cc * k - rt3 * si, cc * k + rt3 * si], -1),
                torch.stack([cc * k + rt3 * si, c + cc * k, cc * k - rt3 * si], -1),
                torch.stack([cc * k - rt3 * si, cc * k + rt3 * si, c + cc * k], -1),
            ],
            -2,
        )
        m = m_hue @ m_sat
        keep = torch.rand(n, 1, 1, device=device, dtype=dtype) >= self.p_jitter
        return torch.where(keep, eye, m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        device, dtype = x.device, x.dtype

        # one composed projective warp; base grid at pixel centers in the
        # align_corners=False convention so an identity matrix reproduces
        # the input exactly (no stray subpixel resample on kept samples)
        m = self._warp_matrices(n, device, dtype)
        ys = (2 * torch.arange(h, device=device, dtype=dtype) + 1) / h - 1
        xs = (2 * torch.arange(w, device=device, dtype=dtype) + 1) / w - 1
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([gx, gy, torch.ones_like(gx)], -1).reshape(1, -1, 3)
        pts = base @ m.transpose(-1, -2)  # [n, h*w, 3]
        grid = (pts[..., :2] / pts[..., 2:].clamp(min=1e-6)).reshape(n, h, w, 2)
        x = torch.nn.functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=False
        )

        # per-sample linear color jitter
        cm = self._color_matrices(n, device, dtype)
        x = torch.einsum("nij,njhw->nihw", cm, x)

        # erasing: one rectangle per sample, applied with p_erase
        area = (
            self.erase_scale[0]
            + torch.rand(n, device=device, dtype=dtype)
            * (self.erase_scale[1] - self.erase_scale[0])
        ) * (h * w)
        log_r = torch.empty(n, device=device, dtype=dtype).uniform_(
            math.log(self.erase_ratio[0]), math.log(self.erase_ratio[1])
        )
        ratio = torch.exp(log_r)
        eh = (area * ratio).sqrt().clamp(max=h - 1)
        ew = (area / ratio).sqrt().clamp(max=w - 1)
        y0 = torch.rand(n, device=device, dtype=dtype) * (h - eh)
        x0 = torch.rand(n, device=device, dtype=dtype) * (w - ew)
        yy = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)
        xx = torch.arange(w, device=device, dtype=dtype).view(1, 1, w)
        inside = (
            (yy >= y0.view(n, 1, 1))
            & (yy < (y0 + eh).view(n, 1, 1))
            & (xx >= x0.view(n, 1, 1))
            & (xx < (x0 + ew).view(n, 1, 1))
        )
        erase = inside & (
            torch.rand(n, 1, 1, device=device) < self.p_erase
        )
        return x * (~erase).unsqueeze(1).to(dtype)


def pytti_batched():
    return BatchedAugs()
