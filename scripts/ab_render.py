#!/usr/bin/env python
"""
A/B render harness (plan Slice 3 / §4): render the SAME fixed-seed config
under two variants and emit everything needed for a human + metric verdict:

  - contact sheet PNG (row A over row B, every saved frame)
  - final-frame diptych PNG
  - LPIPS(final_a, final_b) + per-frame LPIPS curve PNG
  - pixel MAE
  - wall-clock and s/step comparison parsed from the render logs
  - a one-screen markdown report (+ metrics.json)

Usage:
  .venv/bin/python scripts/ab_render.py --conf golden-lp-still \\
      --label-a classic --label-b batched \\
      --overrides-a cutout_sampler=classic \\
      --overrides-b cutout_sampler=batched \\
      --steps 24

--conf takes a golden name (resolved in tests/fixtures/golden/) or a path to
a hydra preset yaml (`# @package _global_` header). Each variant renders via
`python -m pytti.workhorse` in its own scratch workspace under
/tmp/pytti-ab/<label>/ (wiped per invocation). Any crash is fatal.

Reading the numbers — the measured determinism floor (2026-07-30, torch
2.13.0, M-series Mac): the pipeline is bit-deterministic per seed, but
parallel-reduction order in MPS and multi-threaded CPU kernels is not, and
the optimization loop amplifies those last-bit differences chaotically.
Same-seed same-config null pairs measured: MPS 512px 24 steps -> final
LPIPS ~0.027 (frame 1 exactly 0, growing monotonically); CPU multi-thread
128px 12 steps -> ~0.0008-0.0016; CPU with OMP_NUM_THREADS=1 -> exactly
0.000000 on every frame. So for "is B identical to A?" gates, either run
both variants under OMP_NUM_THREADS=1 device=cpu, or judge the A/B against
a same-vs-same null pair at the same step count on the same device.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "golden"
AB_ROOT = Path("/tmp/pytti-ab")

_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_FRAME_RE = re.compile(r"_(\d+)\.png$")
# a tqdm bar state: "  12/24 [00:15<00:15,  1.29s/it]"
_TQDM_RE = re.compile(r"(\d+)/(\d+) \[([\d:]+)<[^\]]*,\s*([\d.]+)\s*(s/it|it/s)\]")


def die(msg: str) -> NoReturn:
    raise SystemExit(f"ab_render: error: {msg}")


# ----------------------------------------------------------------------
# run one variant
# ----------------------------------------------------------------------


@dataclass
class Timing:
    steps: int
    elapsed_s: float  # summed from completed tqdm bars
    s_per_step: float  # steps-weighted from tqdm rates


@dataclass
class RunResult:
    label: str
    workdir: Path
    run_dir: Path
    log_path: Path
    wall_s: float
    frames: dict[int, Path]  # frame index -> png path
    timing: Timing


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


def pytti_assets_dir() -> Path:
    import pytti

    assets = Path(pytti.__file__).parent / "assets"
    if not (assets / "default.yaml").is_file():
        die(f"pytti assets/default.yaml not found under {assets}")
    return assets


def prepare_workspace(label: str, conf_path: Path) -> Path:
    """Scratch cwd for one variant: local hydra config dir + run dir."""
    workdir = AB_ROOT / label
    if workdir.exists():
        shutil.rmtree(workdir)
    conf_dir = workdir / "config" / "conf"
    conf_dir.mkdir(parents=True)
    shutil.copyfile(pytti_assets_dir() / "default.yaml", workdir / "config" / "default.yaml")
    (conf_dir / "_empty.yaml").write_text("\n", encoding="utf-8")
    shutil.copyfile(conf_path, conf_dir / conf_path.name)
    return workdir


def parse_elapsed(stamp: str) -> float:
    parts = [int(p) for p in stamp.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    die(f"unparseable tqdm elapsed stamp: {stamp!r}")


def parse_timing(log_text: str, log_path: Path) -> Timing:
    """Sum the completed tqdm training bars in a render log."""
    # segment the match stream into bars: a counter that goes backwards means
    # a new bar started; otherwise it's a later render of the same bar (tqdm
    # re-renders the final state on close, often with a different rate)
    last_per_bar: list[re.Match] = []
    prev_n = None
    for m in _TQDM_RE.finditer(log_text):
        n = int(m.group(1))
        if prev_n is None or n < prev_n:
            last_per_bar.append(m)
        else:
            last_per_bar[-1] = m
        prev_n = n

    complete: list[tuple[int, float, float]] = []  # (steps, elapsed_s, s_per_step)
    for m in last_per_bar:
        n, total = int(m.group(1)), int(m.group(2))
        if n != total or total < 2:
            continue
        rate = float(m.group(4))
        s_per_step = rate if m.group(5) == "s/it" else 1.0 / rate
        complete.append((total, parse_elapsed(m.group(3)), s_per_step))
    if not complete:
        die(f"no completed training progress bar found in {log_path}")
    steps = sum(c[0] for c in complete)
    elapsed = sum(c[1] for c in complete)
    s_per_step = sum(c[0] * c[2] for c in complete) / steps
    return Timing(steps=steps, elapsed_s=elapsed, s_per_step=s_per_step)


def run_variant(
    label: str, conf_path: Path, overrides: list[str], steps: int | None
) -> RunResult:
    workdir = prepare_workspace(label, conf_path)
    run_dir = workdir / "run"
    log_path = workdir / "render.log"

    cmd = [
        sys.executable,
        "-m",
        "pytti.workhorse",
        f"conf={conf_path.stem}",
        f"file_namespace={label}",
        "allow_overwrite=true",
        f"hydra.run.dir={run_dir}",
    ]
    if steps is not None:
        # keep ~8 saved frames regardless of the shortened run
        cmd += [f"steps_per_scene={steps}", f"save_every={max(1, steps // 8)}"]
    cmd += overrides

    print(f"[{label}] rendering: {' '.join(cmd)}")
    t0 = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    wall_s = time.monotonic() - t0
    log_text = log_path.read_text(encoding="utf-8")
    if proc.returncode != 0:
        tail = "".join(log_text.replace("\r", "\n").splitlines(keepends=True)[-30:])
        die(
            f"render for {label!r} failed with exit code {proc.returncode}.\n"
            f"--- last log lines ({log_path}) ---\n{tail}"
        )

    frames_dir = run_dir / "images_out" / label
    frames: dict[int, Path] = {}
    if frames_dir.is_dir():
        for p in sorted(frames_dir.glob("*.png")):
            m = _FRAME_RE.search(p.name)
            if m:
                frames[int(m.group(1))] = p
    if not frames:
        die(f"render for {label!r} exited 0 but saved no frames in {frames_dir}")

    print(f"[{label}] done in {wall_s:.1f}s, {len(frames)} frames saved")
    return RunResult(
        label=label,
        workdir=workdir,
        run_dir=run_dir,
        log_path=log_path,
        wall_s=wall_s,
        frames=frames,
        timing=parse_timing(log_text, log_path),
    )


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------


@dataclass
class FrameMetric:
    frame: int
    lpips: float
    mae: float


def paired_frames(a: RunResult, b: RunResult) -> list[int]:
    if set(a.frames) != set(b.frames):
        die(
            "frame indices differ between runs (different save cadence?): "
            f"A={sorted(a.frames)} B={sorted(b.frames)}"
        )
    return sorted(a.frames)


def compute_metrics(a: RunResult, b: RunResult, net: str) -> list[FrameMetric]:
    import lpips
    import numpy as np
    import torch
    from PIL import Image

    loss_fn = lpips.LPIPS(net=net, verbose=False)
    loss_fn.eval()

    def to_tensor(path: Path) -> torch.Tensor:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    out: list[FrameMetric] = []
    with torch.no_grad():
        for idx in paired_frames(a, b):
            ta, tb = to_tensor(a.frames[idx]), to_tensor(b.frames[idx])
            if ta.shape != tb.shape:
                die(f"frame {idx}: size mismatch {tuple(ta.shape)} vs {tuple(tb.shape)}")
            d = float(loss_fn(ta * 2 - 1, tb * 2 - 1).item())
            mae = float((ta - tb).abs().mean().item())
            out.append(FrameMetric(frame=idx, lpips=d, mae=mae))
    return out


# ----------------------------------------------------------------------
# image artifacts (PIL only — no plotting deps)
# ----------------------------------------------------------------------

_GAP = 4
_MARGIN = 10


def _font(size: int):
    from PIL import ImageFont

    return ImageFont.load_default(size=size)


def contact_sheet(
    a: RunResult, b: RunResult, out_path: Path, thumb_w: int, max_cols: int
) -> None:
    from PIL import Image, ImageDraw

    indices = paired_frames(a, b)
    if len(indices) > max_cols:
        picks = sorted(
            {round(i * (len(indices) - 1) / (max_cols - 1)) for i in range(max_cols)}
        )
        indices = [indices[i] for i in picks]

    first = Image.open(a.frames[indices[0]])
    thumb_h = round(thumb_w * first.height / first.width)
    cols = len(indices)
    header_h, label_h = 20, 22
    width = 2 * _MARGIN + cols * thumb_w + (cols - 1) * _GAP
    height = 2 * _MARGIN + header_h + 2 * (label_h + thumb_h) + _GAP

    sheet = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    small, med = _font(12), _font(14)

    for col, idx in enumerate(indices):
        x = _MARGIN + col * (thumb_w + _GAP)
        draw.text((x, _MARGIN + 3), f"frame {idx:04d}", font=small, fill=(180, 180, 180))

    for row, run in enumerate((a, b)):
        y_label = _MARGIN + header_h + row * (label_h + thumb_h + _GAP)
        tag = "A" if row == 0 else "B"
        draw.text(
            (_MARGIN, y_label + 3), f"{tag}: {run.label}", font=med, fill=(255, 255, 255)
        )
        for col, idx in enumerate(indices):
            x = _MARGIN + col * (thumb_w + _GAP)
            thumb = Image.open(run.frames[idx]).resize(
                (thumb_w, thumb_h), Image.LANCZOS
            )
            sheet.paste(thumb, (x, y_label + label_h))
    sheet.save(out_path)


def final_diptych(a: RunResult, b: RunResult, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    idx = paired_frames(a, b)[-1]
    im_a, im_b = Image.open(a.frames[idx]), Image.open(b.frames[idx])
    caption_h = 30
    width = 2 * _MARGIN + im_a.width + _GAP + im_b.width
    height = 2 * _MARGIN + max(im_a.height, im_b.height) + caption_h

    sheet = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    med = _font(15)
    sheet.paste(im_a, (_MARGIN, _MARGIN))
    sheet.paste(im_b, (_MARGIN + im_a.width + _GAP, _MARGIN))
    y_caption = _MARGIN + max(im_a.height, im_b.height) + 6
    draw.text((_MARGIN, y_caption), f"A: {a.label} (frame {idx:04d})", font=med, fill=(255, 255, 255))
    draw.text(
        (_MARGIN + im_a.width + _GAP, y_caption),
        f"B: {b.label} (frame {idx:04d})",
        font=med,
        fill=(255, 255, 255),
    )
    sheet.save(out_path)


def lpips_curve(metrics: list[FrameMetric], out_path: Path) -> None:
    """Minimal PIL line chart — matplotlib is deliberately not a dep."""
    from PIL import Image, ImageDraw

    w, h = 820, 320
    left, right, top, bottom = 70, 20, 34, 44
    plot_w, plot_h = w - left - right, h - top - bottom
    y_max = max(1e-6, max(m.lpips for m in metrics) * 1.15)

    im = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    small, med = _font(12), _font(14)
    draw.text((left, 8), "per-frame LPIPS (A vs B)", font=med, fill=(0, 0, 0))

    # axes + y ticks
    draw.line([(left, top), (left, top + plot_h), (left + plot_w, top + plot_h)], fill=(0, 0, 0))
    for i in range(5):
        frac = i / 4
        y = top + plot_h - frac * plot_h
        draw.line([(left - 4, y), (left, y)], fill=(0, 0, 0))
        draw.line([(left, y), (left + plot_w, y)], fill=(230, 230, 230))
        draw.text((6, y - 7), f"{frac * y_max:.4f}", font=small, fill=(0, 0, 0))

    def xy(i: int, value: float) -> tuple[float, float]:
        x_frac = i / max(1, len(metrics) - 1)
        return (left + x_frac * plot_w, top + plot_h - (value / y_max) * plot_h)

    points = [xy(i, m.lpips) for i, m in enumerate(metrics)]
    if len(points) > 1:
        draw.line(points, fill=(31, 119, 180), width=2)
    for (x, y), m in zip(points, metrics, strict=True):
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(31, 119, 180))
        draw.text((x - 12, top + plot_h + 8), f"{m.frame:04d}", font=small, fill=(0, 0, 0))
    draw.text((left + plot_w // 2 - 20, h - 18), "frame", font=small, fill=(0, 0, 0))
    im.save(out_path)


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------


def write_report(
    out_dir: Path,
    conf_path: Path,
    args: argparse.Namespace,
    a: RunResult,
    b: RunResult,
    metrics: list[FrameMetric],
) -> Path:
    import torch

    final = metrics[-1]
    lines = [
        "# pytti A/B render report",
        "",
        f"- **conf:** `{conf_path.stem}` ({conf_path})",
        f"- **when/where:** {time.strftime('%Y-%m-%d %H:%M:%S')} · {platform.platform()} · "
        f"torch {torch.__version__} · mps={torch.backends.mps.is_available()}",
        f"- **steps:** {a.timing.steps}"
        + (f" (override --steps {args.steps})" if args.steps is not None else " (from conf)"),
        f"- **A** `{a.label}` — overrides: `{' '.join(args.overrides_a) or '(none)'}`",
        f"- **B** `{b.label}` — overrides: `{' '.join(args.overrides_b) or '(none)'}`",
        "",
        "## Timing",
        "",
        "| variant | wall-clock (s) | s/step (tqdm) | steps |",
        "|---|---|---|---|",
        f"| A {a.label} | {a.wall_s:.1f} | {a.timing.s_per_step:.3f} | {a.timing.steps} |",
        f"| B {b.label} | {b.wall_s:.1f} | {b.timing.s_per_step:.3f} | {b.timing.steps} |",
        f"| B/A | {b.wall_s / a.wall_s:.3f}x | {b.timing.s_per_step / a.timing.s_per_step:.3f}x | |",
        "",
        "## Similarity (A vs B, same frame index)",
        "",
        f"- **LPIPS(final):** {final.lpips:.6f}",
        f"- **Pixel MAE(final):** {final.mae:.6f} (0-1 scale)",
        f"- **mean per-frame LPIPS:** {sum(m.lpips for m in metrics) / len(metrics):.6f}",
        "",
        "| frame | LPIPS | MAE |",
        "|---|---|---|",
        *(f"| {m.frame:04d} | {m.lpips:.6f} | {m.mae:.6f} |" for m in metrics),
        "",
        "## Artifacts",
        "",
        f"- contact sheet (row A over row B): {out_dir / 'contact_sheet.png'}",
        f"- final-frame diptych: {out_dir / 'final_diptych.png'}",
        f"- LPIPS curve: {out_dir / 'lpips_curve.png'}",
        f"- metrics: {out_dir / 'metrics.json'}",
        f"- runs/logs: {a.workdir} · {b.workdir}",
        "",
    ]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_metrics_json(
    out_dir: Path,
    conf_path: Path,
    args: argparse.Namespace,
    a: RunResult,
    b: RunResult,
    metrics: list[FrameMetric],
) -> None:
    payload = {
        "conf": str(conf_path),
        "steps_override": args.steps,
        "lpips_net": args.lpips_net,
        "a": {
            "label": a.label,
            "overrides": args.overrides_a,
            "wall_s": a.wall_s,
            "s_per_step": a.timing.s_per_step,
            "steps": a.timing.steps,
        },
        "b": {
            "label": b.label,
            "overrides": args.overrides_b,
            "wall_s": b.wall_s,
            "s_per_step": b.timing.s_per_step,
            "steps": b.timing.steps,
        },
        "final_lpips": metrics[-1].lpips,
        "final_mae": metrics[-1].mae,
        "per_frame": [
            {"frame": m.frame, "lpips": m.lpips, "mae": m.mae} for m in metrics
        ],
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--conf",
        required=True,
        help="golden config name (tests/fixtures/golden/) or path to a preset yaml",
    )
    parser.add_argument("--label-a", required=True, help="name for variant A")
    parser.add_argument("--label-b", required=True, help="name for variant B")
    parser.add_argument(
        "--overrides-a",
        nargs="*",
        default=[],
        metavar="K=V",
        help="hydra overrides applied only to variant A",
    )
    parser.add_argument(
        "--overrides-b",
        nargs="*",
        default=[],
        metavar="K=V",
        help="hydra overrides applied only to variant B",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override steps_per_scene for both runs (save_every rescales to keep ~8 frames)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="report directory (default /tmp/pytti-ab/<conf>__<a>_vs_<b>)",
    )
    parser.add_argument(
        "--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"]
    )
    parser.add_argument("--thumb-width", type=int, default=192)
    parser.add_argument("--max-columns", type=int, default=12)
    args = parser.parse_args()

    for label in (args.label_a, args.label_b):
        if not _LABEL_RE.match(label):
            parser.error(f"label {label!r} must match {_LABEL_RE.pattern}")
    if args.label_a == args.label_b:
        parser.error("--label-a and --label-b must differ (they name the run dirs)")
    if args.steps is not None and args.steps < 2:
        parser.error("--steps must be >= 2")
    return args


def main() -> None:
    args = parse_args()
    conf_path = resolve_conf(args.conf)
    out_dir = args.out or AB_ROOT / f"{conf_path.stem}__{args.label_a}_vs_{args.label_b}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    run_a = run_variant(args.label_a, conf_path, args.overrides_a, args.steps)
    run_b = run_variant(args.label_b, conf_path, args.overrides_b, args.steps)

    print(f"computing LPIPS ({args.lpips_net}) + MAE ...")
    metrics = compute_metrics(run_a, run_b, net=args.lpips_net)

    contact_sheet(
        run_a, run_b, out_dir / "contact_sheet.png", args.thumb_width, args.max_columns
    )
    final_diptych(run_a, run_b, out_dir / "final_diptych.png")
    lpips_curve(metrics, out_dir / "lpips_curve.png")
    write_metrics_json(out_dir, conf_path, args, run_a, run_b, metrics)
    report_path = write_report(out_dir, conf_path, args, run_a, run_b, metrics)

    final = metrics[-1]
    print()
    print(f"LPIPS(final) = {final.lpips:.6f}   MAE(final) = {final.mae:.6f}")
    print(
        f"wall-clock A={run_a.wall_s:.1f}s B={run_b.wall_s:.1f}s   "
        f"s/step A={run_a.timing.s_per_step:.3f} B={run_b.timing.s_per_step:.3f}"
    )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
