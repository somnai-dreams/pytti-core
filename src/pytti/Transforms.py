import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from loguru import logger
from PIL import Image

from pytti import parametric_eval
from pytti.LossAug.DepthLossClass import DepthLoss

PADDING_MODES = {
    "mirror": "reflection",
    "smear": "border",
    "black": "zeros",
    "wrap": "zeros",
}


@torch.no_grad()
def apply_grid(tensor, grid, border_mode, sampling_mode):
    height, width = tensor.shape[-2:]
    if border_mode == "wrap":
        max_offset = torch.max(grid.clamp(min=1))
        min_offset = torch.min(grid.clamp(max=-1))
        max_coord = max(max_offset, abs(min_offset))
        if max_coord > 1:
            mod_offset = int(math.ceil(max_coord))
            # make it odd for sure
            mod_offset += 1 - (mod_offset % 2)
            grid = grid.add(mod_offset).remainder(2.0001).sub(1)
    return F.grid_sample(
        tensor,
        grid,
        mode=sampling_mode,
        align_corners=True,
        padding_mode=PADDING_MODES[border_mode],
    )


@torch.no_grad()
def apply_flow(img, flow, border_mode="mirror", sampling_mode="bilinear", device=None):
    if device is None:
        device = img.device
    try:
        tensor = img.get_image_tensor().unsqueeze(0)
        fallback = False
    except NotImplementedError:
        tensor = TF.to_tensor(img.decode_image()).unsqueeze(0).to(device)
        fallback = True

    height, width = flow.shape[-2:]
    identity = torch.eye(3).to(device)
    identity = identity[0:2, :].unsqueeze(0)  # for batch
    uv = F.affine_grid(identity, tensor.shape, align_corners=True)
    flow = (
        TF.resize(flow, tensor.shape[-2:])
        .movedim(1, 3)
        .div(torch.tensor([[[[width / 2, height / 2]]]], device=device))
    )
    grid = uv - flow
    tensor = apply_grid(tensor, grid, border_mode, sampling_mode)
    if not fallback:
        img.set_image_tensor(tensor.squeeze(0))
        tensor_out = img.decode_tensor().detach()
    else:
        array = (
            tensor.squeeze()
            .movedim(0, -1)
            .mul(255)
            .clamp(0, 255)
            .cpu()
            .detach()
            .numpy()
            .astype(np.uint8)[:, :, :]
        )
        img.encode_image(Image.fromarray(array))
        tensor_out = tensor.detach()
    return tensor_out


@torch.no_grad()
def zoom_2d(
    img,
    translate=(0, 0),
    zoom=(0, 0),
    rotate=0,
    border_mode="mirror",
    sampling_mode="bilinear",
    device=None,
):
    if device is None:
        device = img.device
    try:
        tensor = img.get_image_tensor().unsqueeze(0)
        fallback = False
    except NotImplementedError:
        tensor = TF.to_tensor(img.decode_image()).unsqueeze(0).to(device)
        fallback = True
    height, width = tensor.shape[-2:]
    zy, zx = ((height - zoom[1]) / height, (width - zoom[0]) / width)
    ty, tx = (translate[1] * 2 / height, -translate[0] * 2 / width)
    theta = math.radians(rotate)
    affine = (
        torch.tensor(
            [
                [zx * math.cos(theta), -zy * math.sin(theta), tx],
                [zx * math.sin(theta), zy * math.cos(theta), ty],
            ]
        )
        .unsqueeze(0)
        .to(device)
    )
    grid = F.affine_grid(affine, tensor.shape, align_corners=True)
    tensor = apply_grid(tensor, grid, border_mode, sampling_mode)
    if not fallback:
        img.set_image_tensor(tensor.squeeze(0))
    else:
        array = (
            tensor.squeeze()
            .movedim(0, -1)
            .mul(255)
            .clamp(0, 255)
            .cpu()
            .detach()
            .numpy()
            .astype(np.uint8)[:, :, :]
        )
        img.encode_image(Image.fromarray(array))
    return img.decode_image()


def perspective_matrix(
    fov_rad: float, aspect: float, near: float = 0.1, far: float = 4.0
) -> torch.Tensor:
    """
    Right-handed OpenGL-style projection matrix (row-major math convention).
    Only the x/y/w rows influence the computed 2D flow — z is divided away —
    so near/far only shape the (unused) depth output.
    """
    g = 1.0 / math.tan(fov_rad / 2)
    m = torch.zeros(4, 4)
    m[0, 0] = g / aspect
    m[1, 1] = g
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def quaternion_matrix(w: float, x: float, y: float, z: float) -> torch.Tensor:
    """4x4 rotation matrix from a (w, x, y, z) quaternion (normalized here)."""
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("rotate_3d quaternion must be non-zero, got [0, 0, 0, 0]")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    m = torch.eye(4)
    m[0, 0] = 1 - 2 * (y * y + z * z)
    m[0, 1] = 2 * (x * y - w * z)
    m[0, 2] = 2 * (x * z + w * y)
    m[1, 0] = 2 * (x * y + w * z)
    m[1, 1] = 1 - 2 * (x * x + z * z)
    m[1, 2] = 2 * (y * z - w * x)
    m[2, 0] = 2 * (x * z - w * y)
    m[2, 1] = 2 * (y * z + w * x)
    m[2, 2] = 1 - 2 * (x * x + y * y)
    return m


def translation_matrix(tx: float, ty: float, tz: float) -> torch.Tensor:
    m = torch.eye(4)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


@torch.no_grad()
def render_image_3d(
    image,
    depth,
    P,
    T,
    border_mode,
    sampling_mode,
    stabilize,
    device=None,
):
    """
    image: n x h x w pytorch Tensor: the image tensor
    depth: h x w pytorch Tensor: the depth tensor
    P: 4 x 4 pytorch Tensor: the perspective matrix
    T: 4 x 4 pytorch Tensor: the camera move matrix
    """
    # create grid of points matching image
    h, w = image.shape[-2:]
    f = w / h
    image = image.unsqueeze(0)

    if device is None:
        device = image.device
    logger.debug(device)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-f, f, w), indexing="ij"
    )
    x = x.unsqueeze(0).unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(0)
    xy = torch.cat([x, y], dim=1).to(device)

    # v,u = torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w))
    # u = u.unsqueeze(0).unsqueeze(0)
    # v = v.unsqueeze(0).unsqueeze(0)
    # uv = torch.cat([u,v],dim=1).to(device)
    identity = torch.eye(3).to(device)
    identity = identity[0:2, :].unsqueeze(0)  # for batch
    uv = F.affine_grid(identity, image.shape, align_corners=True)
    # get the depth at each point
    depth = depth.unsqueeze(0).unsqueeze(0)
    # depth = depth.to(device)

    view_pos = torch.cat([xy, -depth, torch.ones_like(depth)], dim=1)
    # apply the camera move matrix (P and T are row-major math matrices, so
    # contract their column dim against the point channel dim: out = M @ v)
    next_view_pos = torch.tensordot(T.float(), view_pos.float(), ([1], [1])).movedim(
        0, 1
    )

    # apply the perspective matrix
    clip_pos = torch.tensordot(P.float(), view_pos.float(), ([1], [1])).movedim(0, 1)
    clip_pos = clip_pos / (clip_pos[:, 3, ...].unsqueeze(1))

    next_clip_pos = torch.tensordot(
        P.float(), next_view_pos.float(), ([1], [1])
    ).movedim(0, 1)
    next_clip_pos = next_clip_pos / (next_clip_pos[:, 3, ...].unsqueeze(1))

    # get the offset
    offset = (next_clip_pos - clip_pos)[:, 0:2, ...]
    # flow_forward = offset.mul(torch.tensor([w/2,h/(2*f)],device = device).view(1,2,1,1))
    # offset[:,1,...] *= -1
    # offset = offset.to(device)
    # render the image
    if stabilize:
        advection = offset.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        offset = offset - advection
    offset = offset.permute(0, 2, 3, 1)
    grid = uv - offset
    # grid = grid.permute(0,2,3,1)
    # grid = grid.to(device)

    return apply_grid(image, grid, border_mode, sampling_mode).squeeze(
        0
    ), offset.squeeze(0)


@torch.no_grad()
def zoom_3d(
    img,
    translate=(0, 0, 0),
    rotate=0,
    fov=45,
    near=180,
    far=15000,
    border_mode="mirror",
    sampling_mode="bilinear",
    stabilize=False,
    device=None,
):

    if device is None:
        device = img.device

    width, height = img.image_shape
    px = 2 / height
    alpha = math.radians(fov)

    pil_image = img.decode_image()
    f = width / height

    # convert depth map: AdaBins metric range (~1e-3..10m) rescaled into the
    # configured near/far pixel range
    depth_map, depth_resized = DepthLoss.get_depth(pil_image, device=device)
    depth_map = np.interp(depth_map, (1e-3, 10), (near * px, far * px))

    depth_median = np.median(depth_map.flatten())
    depth_mean = np.mean(depth_map)
    r = np.min(depth_map) / px
    R = np.max(depth_map) / px
    mu = (depth_mean + depth_median) / (2 * px)
    logger.debug(f"depth range: {r} (r) to {R} (R), mu = {mu}")
    translate = [parametric_eval(x, r=r, R=R, mu=mu) for x in translate]
    rotate = parametric_eval(rotate, r=r, R=R, mu=mu)
    logger.debug(f"moving: {translate}")
    try:
        image_tensor = img.get_image_tensor().to(device)
        depth_tensor = (
            TF.resize(
                torch.from_numpy(depth_map),
                image_tensor.shape[-2:],
                interpolation=TF.InterpolationMode.BICUBIC,
            )
            .squeeze()
            .to(device)
        )
        fallback = False
    except NotImplementedError:
        # fallback path
        image_tensor = TF.to_tensor(pil_image).to(device)
        if depth_resized:
            depth_tensor = (
                TF.resize(
                    torch.from_numpy(depth_map),
                    image_tensor.shape[-2:],
                    interpolation=TF.InterpolationMode.BICUBIC,
                )
                .squeeze()
                .to(device)
            )
        else:
            depth_tensor = torch.from_numpy(depth_map).squeeze().to(device)
        fallback = True
    p_matrix = perspective_matrix(alpha, f).to(device)
    tx, ty, tz = translate
    if not isinstance(rotate, (list, tuple)) or len(rotate) != 4:
        raise ValueError(
            f"rotate_3d must evaluate to a [w, x, y, z] quaternion, got {rotate!r}"
        )
    T_matrix = (
        quaternion_matrix(*rotate) @ translation_matrix(tx * px, -ty * px, tz * px)
    ).to(device)
    new_image, flow = render_image_3d(
        image_tensor,
        depth_tensor,
        p_matrix,
        T_matrix,
        border_mode=border_mode,
        sampling_mode=sampling_mode,
        stabilize=stabilize,
        device=device,
    )
    logger.debug(new_image.device)
    if not fallback:
        img.set_image_tensor(new_image)
    else:
        # fallback path
        array = (
            new_image.movedim(0, -1)
            .mul(255)
            .clamp(0, 255)
            .cpu()
            .detach()
            .numpy()
            .astype(np.uint8)[:, :, :]
        )
        img.encode_image(Image.fromarray(array))

    flow_out = flow.div(2).mul(torch.tensor([[[[width, height]]]], device=device))
    return flow_out, img.decode_image()


def animate_video_source(
    i,
    img,
    video_frames,
    optical_flows,
    base_name,
    pre_animation_steps,
    frame_stride,
    steps_per_frame,
    file_namespace,
    reencode_each_frame,
    lock_palette,
    save_every,
    ##
    infill_mode,
    sampling_mode,
    device=None,
):
    # ugh this is GROSSSSS....
    from pytti.image_models.pixel import PixelImage

    # current frame index
    frame_n = min(
        (i - pre_animation_steps) * frame_stride // steps_per_frame,
        len(video_frames) - 1,
    )

    # Next frame index (for forrward flow)
    next_frame_n = min(frame_n + frame_stride, len(video_frames) - 1)

    # get next frame of video
    # * will be used as init_image
    # * will be used to estimate forward flow
    # Q: what is type of video_frames[i]? np.array? list?
    #    ... probably some imageio class
    next_step_pil = (
        Image.fromarray(video_frames.get_data(next_frame_n))
        .convert("RGB")
        .resize(img.image_shape, Image.LANCZOS)
    )

    # Apply flows
    flow_im = None
    for j, optical_flow in enumerate(optical_flows):
        # This looks like something that we shouldn't have to recompute
        # but rather could be attached to the flow object as an attribute
        old_frame_n = frame_n - (2**j - 1) * frame_stride
        save_n = i // save_every - (2**j - 1)
        if old_frame_n < 0 or save_n < 1:
            break

        current_step_pil = (
            Image.fromarray(video_frames.get_data(old_frame_n))
            .convert("RGB")
            .resize(img.image_shape, Image.LANCZOS)
        )

        filename = f"backup/{file_namespace}/{base_name}_{save_n}.bak"
        filename = None if j == 0 else filename
        # `flow_im` isn't being used for anything.
        # Might be interesting to log it for monitoring
        # Q: why does this function take `_step_pil` objects
        #    as well as a filename? `set_flow` automatically writes a
        #    backup file? That seems like something that should be
        #    configurable.
        flow_im, mask_tensor = optical_flow.set_flow(
            current_step_pil,
            next_step_pil,
            img,
            filename,
            infill_mode,
            sampling_mode,
            device=device,
        )

        optical_flow.set_enabled(True)
        # first flow is previous frame
        if j == 0:
            mask_accum = mask_tensor.detach()
            valid = mask_tensor.mean()
            logger.debug("valid pixels:", valid)
            # what is this magic number here?
            if reencode_each_frame or valid < 0.03:
                if isinstance(img, PixelImage) and valid >= 0.03:
                    img.lock_palette()
                    img.encode_image(next_step_pil, smart_encode=False)
                    img.lock_palette(lock_palette)
                else:
                    img.encode_image(next_step_pil)
        else:
            with torch.no_grad():
                optical_flow.set_mask((mask_tensor - mask_accum).clamp(0, 1))
                mask_accum.add_(mask_tensor)

    return flow_im, next_step_pil
