#!/usr/bin/env python
"""
Held-out prompt-adherence judge for pytti still renders.

Scores final frames against a text prompt with a CLIP perceptor that was NOT
part of the optimization ensemble, giving cutout-strategy A/Bs an objective
adherence axis beyond LPIPS. (An image optimized against ViTB32+ViTB16 will
trivially score high on those; a held-out ViTL14 has no such circularity.)

Protocol (deterministic, no RNG): each image is embedded as 6 views —
the full frame squashed to the judge's input resolution, plus 4 corner crops
and 1 center crop at the inscribed-square size, resized to the input
resolution (bilinear, align_corners=False), normalized with the judge's own
stats. Score = mean over views of cosine(text embed, image embed).

Usage:
  .venv/bin/python scripts/judge_stills.py \\
      --images /tmp/pytti-ab/torch-be/run/images_out/torch-be final.png \\
      --prompt-from-conf golden-lp-still --judge ViTL14 --json scores.json

--images takes PNG paths and/or directories (for a directory: the newest
frame per pytti file_namespace pattern *_NNNN.png). --prompt-from-conf reads
the `scenes` field of a golden yaml (name resolved in tests/fixtures/golden/)
or preset yaml path — single scene only, weights/masks stripped — and refuses
a judge that the conf itself optimizes with. --self-test scores two synthetic
images to prove the pipeline end-to-end.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    import torch

    from pytti.Perceptor import LoadedPerceptor

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "golden"

# pytti file_namespace frame pattern, e.g. golden-lp-still_0042.png
_FRAME_RE = re.compile(r"^(?P<namespace>.+)_(?P<index>\d+)\.png$")

VIEW_NAMES = ("global", "top_left", "top_right", "bottom_left", "bottom_right", "center")


def die(msg: str) -> NoReturn:
    raise SystemExit(f"judge_stills: error: {msg}")


# ----------------------------------------------------------------------
# inputs: images and prompt
# ----------------------------------------------------------------------


def collect_images(specs: list[str]) -> list[Path]:
    """Expand path/dir specs; a dir contributes its newest frame per namespace."""
    images: list[Path] = []
    for spec in specs:
        path = Path(spec)
        if path.is_file():
            if path.suffix.lower() != ".png":
                die(f"{path} is not a .png")
            images.append(path.resolve())
        elif path.is_dir():
            newest: dict[str, tuple[int, Path]] = {}
            for candidate in path.iterdir():
                m = _FRAME_RE.match(candidate.name)
                if m is None:
                    continue
                namespace, index = m.group("namespace"), int(m.group("index"))
                if namespace not in newest or index > newest[namespace][0]:
                    newest[namespace] = (index, candidate.resolve())
            if not newest:
                die(f"directory {path} contains no *_NNNN.png frames")
            images.extend(newest[ns][1] for ns in sorted(newest))
        else:
            die(f"no such file or directory: {spec}")
    if not images:
        die("no images to score")
    return images


def resolve_conf(conf: str) -> Path:
    candidate = Path(conf)
    if candidate.suffix == ".yaml" or "/" in conf:
        if not candidate.is_file():
            die(f"config file not found: {candidate}")
        return candidate.resolve()
    golden = GOLDEN_DIR / f"{conf}.yaml"
    if golden.is_file():
        return golden
    available = sorted(p.stem for p in GOLDEN_DIR.glob("*.yaml"))
    die(f"unknown golden config {conf!r}. Available: {available}")


def prompt_from_conf(conf: str, judge_key: str) -> str:
    """Read the scenes field of a pytti conf and reduce it to plain judge text."""
    import yaml

    conf_path = resolve_conf(conf)
    data = yaml.safe_load(conf_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        die(f"{conf_path} did not parse to a mapping")
    if data.get(judge_key):
        die(
            f"judge {judge_key!r} is enabled in {conf_path.name} — it optimized "
            "the image, so it is not held out. Pick a perceptor the conf does "
            "not use (e.g. --judge ViTL14 for a ViTB32/ViTB16 render)."
        )
    scenes = data.get("scenes")
    if not isinstance(scenes, str) or not scenes.strip():
        die(f"{conf_path} has no usable 'scenes' field")
    return scene_to_text(scenes)


def scene_to_text(scenes: str) -> str:
    """
    Minimal pytti scene parse: single scene only, weights/masks stripped,
    prompt texts joined with ', '. Reuses the real grammar (pytti.prompt_spec)
    rather than re-guessing colon/bracket rules.
    """
    from pytti.prompt_spec import parse_prompt_spec, split_toplevel

    stages = scenes.split("||")
    if len(stages) != 1:
        die(
            f"scenes field has {len(stages)} scenes ('||' separator) — the "
            "judge scores one prompt. Pass --prompt with the text you mean."
        )
    texts: list[str] = []
    for chunk in split_toplevel(stages[0], "|", maxsplit=len(stages[0])):
        try:
            spec = parse_prompt_spec(chunk.strip())
        except ValueError as exc:
            die(str(exc))
        if spec.image_path() is not None:
            die(
                f"scene contains a direct image prompt {spec.text!r} — "
                "no text to judge. Pass --prompt instead."
            )
        texts.append(spec.text)
    return ", ".join(texts)


# ----------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------


@dataclass
class ImageScore:
    path: Path
    views: list[float]  # cosine per view, VIEW_NAMES order (global first)

    @property
    def score(self) -> float:
        return sum(self.views) / len(self.views)

    @property
    def global_view(self) -> float:
        return self.views[0]

    @property
    def crop_mean(self) -> float:
        crops = self.views[1:]
        return sum(crops) / len(crops)


def load_judge(key: str, device: torch.device) -> LoadedPerceptor:
    import torch

    from pytti.Perceptor import PERCEPTOR_REGISTRY, _load_perceptor

    if key not in PERCEPTOR_REGISTRY:
        die(f"unknown judge {key!r}. Available: {sorted(PERCEPTOR_REGISTRY)}")
    with torch.no_grad():
        return _load_perceptor(key, device)


def build_views(path: Path, cut_size: int) -> torch.Tensor:
    """(6, 3, cut_size, cut_size) in [0,1]: global squash + 4 corners + center."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    frame = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    _, h, w = frame.shape
    s = min(h, w)
    y_mid, x_mid = (h - s) // 2, (w - s) // 2
    views = [
        frame,  # global (squashed: aspect not preserved)
        frame[:, :s, :s],  # top_left
        frame[:, :s, w - s :],  # top_right
        frame[:, h - s :, :s],  # bottom_left
        frame[:, h - s :, w - s :],  # bottom_right
        frame[:, y_mid : y_mid + s, x_mid : x_mid + s],  # center
    ]
    return torch.cat(
        [
            F.interpolate(
                v.unsqueeze(0),
                size=(cut_size, cut_size),
                mode="bilinear",
                align_corners=False,
            )
            for v in views
        ],
        dim=0,
    )


def score_images(
    judge: LoadedPerceptor, images: list[Path], prompt: str, device: torch.device
) -> list[ImageScore]:
    import torch
    import torch.nn.functional as F

    results: list[ImageScore] = []
    with torch.no_grad():
        text = F.normalize(judge.embed_text(prompt, device), dim=-1)  # (1, D)
        for path in images:
            batch = judge.normalize(build_views(path, judge.cut_size)).to(device)
            embeds = F.normalize(judge.encode_image(batch).float(), dim=-1)  # (6, D)
            views = [float(v) for v in (embeds @ text.T).squeeze(1).cpu()]
            for name, value in zip(VIEW_NAMES, views, strict=True):
                if not math.isfinite(value) or not -1.0001 <= value <= 1.0001:
                    die(f"{path} view {name}: cosine {value} outside [-1, 1]")
            results.append(ImageScore(path=path, views=views))
    return results


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------


def print_table(
    scores: list[ImageScore], judge: LoadedPerceptor, prompt: str, device: torch.device
) -> None:
    print(
        f"judge={judge.key}  device={device}  "
        f"views=6 (1 global + 5 crops @ {judge.cut_size}px)"
    )
    print(f"prompt: {prompt}")
    print()
    print(f"{'score':>8}  {'global':>8}  {'crops':>8}  image")
    for s in scores:
        print(f"{s.score:8.4f}  {s.global_view:8.4f}  {s.crop_mean:8.4f}  {s.path}")


def write_json(
    out_path: Path,
    scores: list[ImageScore],
    judge: LoadedPerceptor,
    prompt: str,
    device: torch.device,
) -> None:
    payload = {
        "judge": judge.key,
        "cut_size": judge.cut_size,
        "device": str(device),
        "prompt": prompt,
        "view_names": list(VIEW_NAMES),
        "images": [
            {
                "path": str(s.path),
                "score": s.score,
                "global_view": s.global_view,
                "crop_mean": s.crop_mean,
                "views": s.views,
            }
            for s in scores
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\njson: {out_path}")


# ----------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------


def self_test(judge_key: str, device: torch.device) -> None:
    """Score two synthetic images end-to-end; assert sane, distinct scores."""
    import numpy as np
    from PIL import Image

    rng = np.random.RandomState(0)  # fixed seed: the only RNG in this script
    noise_path = Path("/tmp/judge_stills_selftest_noise.png")
    solid_path = Path("/tmp/judge_stills_selftest_solid.png")
    Image.fromarray(
        rng.randint(0, 256, size=(256, 256, 3)).astype(np.uint8)
    ).save(noise_path)
    Image.new("RGB", (256, 256), (40, 90, 160)).save(solid_path)

    prompt = "a photograph of an astronaut riding a horse on the moon"
    judge = load_judge(judge_key, device)
    scores = score_images(judge, [noise_path, solid_path], prompt, device)
    print_table(sorted(scores, key=lambda s: s.score, reverse=True), judge, prompt, device)

    noise, solid = scores[0].score, scores[1].score
    if noise == solid:
        die(f"self-test: noise and solid scored identically ({noise}) — pipeline suspect")
    print(f"\nself-test OK: noise={noise:.4f} solid={solid:.4f} — finite, in [-1,1], distinct")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=[],
        metavar="PATH",
        help="PNG files and/or directories (dir: newest *_NNNN.png per namespace)",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", help="text to judge adherence against")
    prompt_group.add_argument(
        "--prompt-from-conf",
        metavar="CONF",
        help="golden name (tests/fixtures/golden/) or preset yaml; reads 'scenes'",
    )
    parser.add_argument(
        "--judge",
        default="ViTL14",
        help="PERCEPTOR_REGISTRY key; must be held out of the render's ensemble",
    )
    parser.add_argument("--device", default="auto", help="auto | mps | cuda[:N] | cpu")
    parser.add_argument("--json", type=Path, default=None, help="also write scores here")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="score two synthetic images (no render) to prove the pipeline",
    )
    args = parser.parse_args()

    if args.self_test:
        if args.images or args.prompt or args.prompt_from_conf:
            parser.error("--self-test takes no --images/--prompt (only --judge/--device)")
    else:
        if not args.images:
            parser.error("--images is required (or use --self-test)")
        if not args.prompt and not args.prompt_from_conf:
            parser.error("one of --prompt / --prompt-from-conf is required")
    return args


def main() -> None:
    args = parse_args()

    from pytti.device import resolve_device

    device = resolve_device(args.device)

    if args.self_test:
        self_test(args.judge, device)
        return

    images = collect_images(args.images)
    prompt = (
        args.prompt
        if args.prompt is not None
        else prompt_from_conf(args.prompt_from_conf, args.judge)
    )
    judge = load_judge(args.judge, device)
    scores = sorted(
        score_images(judge, images, prompt, device),
        key=lambda s: s.score,
        reverse=True,
    )
    print_table(scores, judge, prompt, device)
    if args.json is not None:
        write_json(args.json, scores, judge, prompt, device)


if __name__ == "__main__":
    main()
