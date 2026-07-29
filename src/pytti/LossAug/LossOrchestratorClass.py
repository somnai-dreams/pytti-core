from loguru import logger
from PIL import Image

from pytti import is_zero_weight
from pytti.image_models import PixelImage

# from pytti.LossAug import build_loss
from pytti.LossAug import OpticalFlowLoss, TargetFlowLoss, TVLoss
from pytti.LossAug.DepthLossClass import DepthLoss
from pytti.LossAug.EdgeLossClass import EdgeLoss
from pytti.Perceptor.Prompt import parse_prompt
from pytti.prompt_spec import parse_weight_spec

#################################


LOSS_DICT = {"edge": EdgeLoss, "depth": DepthLoss}


def build_loss(weight_name, weight, name, img, pil_target):
    category = weight_name.split("_", 1)[0]
    if category == "direct":
        loss_cls = type(img).get_preferred_loss()
    else:
        loss_cls = LOSS_DICT[category]
    # weight fields support the `weight_mask` syntax, e.g. "1_[mask.png]"
    weight_expr, mask, _ = parse_weight_spec(str(weight))
    out = loss_cls.build(
        f"{category} {name}",
        img.image_shape,
        weight=weight_expr,
        mask=mask,
        pil_image=pil_target,
    )
    out.set_enabled(pil_target is not None)
    return out


#################################


def configure_init_image(
    init_image_pil: Image.Image,
    restore: bool,
    img: PixelImage,
    params,
    loss_augs,
    embedder,
    prompts,
):

    if init_image_pil is not None:
        if not restore:
            # move these logging statements into .encode_image()
            logger.info("Encoding image...")
            img.encode_image(init_image_pil)
            logger.info("Image encoded.")
        # set up init image prompt
        init_augs = ["direct_init_weight"]
        init_augs = [
            build_loss(
                x,
                params[x],
                f"init image ({params.init_image})",
                img,
                init_image_pil,
            )
            for x in init_augs
            if not is_zero_weight(params[x])
        ]
        loss_augs.extend(init_augs)
        if params.semantic_init_weight not in ["", "0"]:
            semantic_init_prompt = parse_prompt(
                embedder,
                f"init image [{params.init_image}]:{params.semantic_init_weight}",
                init_image_pil,
            )
            prompts[0].append(semantic_init_prompt)
        else:
            semantic_init_prompt = None
    else:
        init_augs, semantic_init_prompt = [], None

    return init_augs, semantic_init_prompt, loss_augs, img, embedder, prompts


def configure_stabilization_augs(img, init_image_pil, params, loss_augs):
    ## NB: in loss orchestrator impl, this begins with an init_image override.
    ## possibly the source of the issue?
    stabilization_augs = [
        "direct_stabilization_weight",
        "depth_stabilization_weight",
        "edge_stabilization_weight",
    ]
    stabilization_augs = [
        build_loss(x, params[x], "stabilization", img, init_image_pil)
        for x in stabilization_augs
        if not is_zero_weight(params[x])
    ]
    loss_augs.extend(stabilization_augs)

    return loss_augs, img, init_image_pil, stabilization_augs


def configure_optical_flows(img, params, loss_augs):

    if params.animation_mode == "Video Source":
        if params.flow_stabilization_weight == "":
            params.flow_stabilization_weight = "0"
        flow_weight, flow_mask, _ = parse_weight_spec(
            str(params.flow_stabilization_weight)
        )
        optical_flows = [
            OpticalFlowLoss.build(
                f"optical flow stabilization (frame {-2**i})",
                img.image_shape,
                weight=flow_weight,
                mask=flow_mask,
            )
            for i in range(params.flow_long_term_samples + 1)
        ]
        for optical_flow in optical_flows:
            optical_flow.set_enabled(False)
        loss_augs.extend(optical_flows)
    elif params.animation_mode == "3D" and not is_zero_weight(
        params.flow_stabilization_weight
    ):
        flow_weight, flow_mask, _ = parse_weight_spec(
            str(params.flow_stabilization_weight)
        )
        optical_flows = [
            TargetFlowLoss.build(
                "optical flow stabilization",
                img.image_shape,
                weight=flow_weight,
                mask=flow_mask,
            )
        ]
        for optical_flow in optical_flows:
            optical_flow.set_enabled(False)
        loss_augs.extend(optical_flows)
    else:
        optical_flows = []
    # other loss augs
    if params.smoothing_weight != 0:
        loss_augs.append(TVLoss(weight=params.smoothing_weight))

    return img, loss_augs, optical_flows


#######################################
