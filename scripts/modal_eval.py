#!/usr/bin/env python
"""
Run eval_matrix campaigns with cells rendered on Modal GPUs in parallel.

This is a thin scheduling + transport layer over scripts/eval_matrix.py:
plan construction, cell identity, conf generation, resumability, judging,
and every report artifact are eval_matrix's own functions, imported and
called — never reimplemented. What this script adds:

  * a Modal app (ephemeral, auto-torn-down) whose image is debian-slim +
    python 3.10 + torch (CUDA wheel) + pytti-core pip-installed from the
    PINNED commit, so campaigns are reproducible against an exact engine;
  * a render_cell Modal function: given a generated cell conf (text), it
    rebuilds eval_matrix's cell workspace inside the container, runs
    `python -m pytti.workhorse conf=<cell>`, and returns the final frame
    bytes + a log tail + timing. Model weights download from HF at first
    use into a Modal Volume mounted at /cache (reached from /root/.cache
    via a container-start symlink), so they download once per volume, not
    once per cell;
  * dependency WAVES: wave 1 = legs with no init_from, rendered in
    parallel via .starmap; wave n+1 = chained legs, each cell carrying its
    source cell's final frame BYTES (the conf's init_image is rewritten to
    the container-side path; the LOCAL conf keeps eval_matrix's canonical
    text, so cell identity and resumability are unchanged);
  * download of finals into eval_matrix's exact local cell layout under
    /tmp/pytti-eval-modal/<tier>/<leg>/<cell>/run/images_out/..., after
    which judging + reports run through the local eval_matrix path and
    come out identical in shape. The root is SEPARATE from local
    /tmp/pytti-eval campaigns — CUDA and MPS cells never share a pool.

Backend constraint: every leg must render with perceptor_backend=torch
(the schema default). mlx / mlx_full are Metal-only and fail loud here at
plan time, naming the constraint.

Usage (mirrors eval_matrix; plan first with --dry-run, always — and
nothing is ever billed without the explicit --launch flag):
  .venv/bin/python scripts/modal_eval.py \\
      --tier full --subset 5 \\
      --baseline pixel:image_model=Limited\\ Palette \\
      --legs lg:image_model=LlamaGen \\
      --legs 'up-tight:init_from=lg,image_model=Limited Palette,direct_init_weight=2' \\
      --gpu L4 --max-parallel 10 --dry-run

The extra flags over eval_matrix: --launch (required to submit paid
renders; without it the plan + cost estimate print and nothing is
spent), --gpu (priced choices below), --max-parallel (container fan-out
cap, default 10), --timeout-min (per cell, default 20),
--est-cell-minutes (cost-estimate override; per-tier defaults are
calibrated from the 2026-08 paid smoke).

The 'smoke' tier (this script only — eval_matrix's own CLI is untouched)
renders 128px/20-step cells for the paid integration smoke; pair it with
tests/fixtures/eval/battery_smoke.yaml (1 prompt, 1 seed).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent


def _import_eval_matrix():
    """scripts/ is not a package; make `import eval_matrix` resolvable."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import eval_matrix

    return eval_matrix


em = _import_eval_matrix()

# ----------------------------------------------------------------------
# pinned image definition — campaigns are reproducible against this exact
# engine; a pytti-core fix means: commit, push, update PINNED_COMMIT
# ----------------------------------------------------------------------

PINNED_COMMIT = "2868ca8e670e60cc55118708c2fc01fd5192b66e"
PYTTI_GIT_URL = f"git+https://github.com/somnai-dreams/pytti-core@{PINNED_COMMIT}"
# match the local venv exactly: the render engine is version-pinned, and the
# remote function is cloudpickled from this interpreter (3.10 <-> 3.10)
PYTHON_VERSION = "3.10"
TORCH_PIN = "2.13.0"  # local venv's torch; linux wheel ships CUDA

APP_NAME = "pytti-eval"
VOLUME_NAME = "pytti-eval-cache"
# one volume covers BOTH weight caches: the HF hub cache (~/.cache/
# huggingface — FARE4/SigLIP2/LlamaGen/open_clip towers) and pytti's
# models_parent_dir="${user_cache:}" (~/.cache/pytti/vqgan). It cannot
# mount AT /root/.cache — Modal appends its own client-dependency layer
# after every user layer and that leaves pip artifacts there, and Modal
# refuses to mount a volume over a non-empty path — so it mounts at
# /cache and render_cell symlinks /root/.cache -> /cache at container
# start, which routes every ~/.cache consumer into the volume.
CACHE_MOUNT = "/cache"
CONTAINER_WORK_ROOT = "/work"

# local root for Modal-rendered campaigns — deliberately NOT eval_matrix's
# /tmp/pytti-eval: CUDA and MPS cells must never mix in one pool
MODAL_EVAL_ROOT = Path("/tmp/pytti-eval-modal")

# modal.com/pricing, 2026-08 (per-second billing, shown per hour)
GPU_DOLLARS_PER_HOUR = {
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
}

# per-cell wall-clock estimates (minutes, incl. per-cell overhead; excludes
# one-time image build + first-ever weight downloads). CALIBRATED from the
# 2026-08-10 paid smoke on L4: full-tier modern-stack cell (512px/200 steps,
# FARE4+SigLIP2B16) = 2.65 min incl. its first SigLIP2B16 download; smoke
# cell (128px/20 steps) = 0.3-0.5 min, ~15-20s of it fixed model-load
# overhead. screening (256px/150 steps) is interpolated from those two
# (fixed ~0.3 min + ~0.19x full-tier compute), not yet measured directly.
# Override with --est-cell-minutes for unusually heavy leg sets.
DEFAULT_EST_MINUTES = {"screening": 1.0, "full": 2.7, "smoke": 0.5}

# the paid-smoke tier: tiny cells, real end-to-end plumbing. Lives here
# only — eval_matrix's TIERS and CLI stay byte-identical.
SMOKE_TIER = em.Tier(
    name="smoke",
    dims={"1:1": (128, 128), "3:4": (120, 160)},
    steps=20,
    save_every=10,
)
TIERS = {**em.TIERS, "smoke": SMOKE_TIER}

LOG_TAIL_LINES = 60


def die(msg: str) -> NoReturn:
    raise SystemExit(f"modal_eval: error: {msg}")


# ----------------------------------------------------------------------
# plan reuse (verbatim eval_matrix machinery) + modal-only validation
# ----------------------------------------------------------------------


def validate_backend_torch(leg_set) -> None:
    """Modal cells render on linux CUDA: only perceptor_backend=torch runs
    there. mlx/mlx_full are Metal-only — fail loud at plan time, per leg."""
    for leg in leg_set.legs:
        overrides = leg.overrides_dict()
        backend = yaml.safe_load(overrides.get("perceptor_backend", "torch"))
        if backend != "torch":
            die(
                f"leg {leg.name!r}: perceptor_backend={backend!r} cannot render "
                "on Modal — cells run on linux CUDA and mlx/mlx_full are "
                "Metal-only. Use perceptor_backend=torch (the default)."
            )
        device = overrides.get("device")
        if device is not None and yaml.safe_load(device) not in ("cuda", "auto"):
            die(
                f"leg {leg.name!r}: device={device!r} would not resolve on a "
                "Modal GPU container. Drop the override (auto resolves to cuda)."
            )


@dataclass(frozen=True)
class Campaign:
    plan: object  # em.Plan
    leg_set: object  # em.LegSet
    sweeps: list  # list[em.SweepField]
    tier: object  # em.Tier


def build_campaign(args: argparse.Namespace, root: Path = MODAL_EVAL_ROOT) -> Campaign:
    """eval_matrix's own parse/assemble/plan path, verbatim — the only
    additions are the torch-backend gate and the modal-scoped root."""
    explicit = [em.parse_leg_spec(spec) for spec in args.legs]
    baseline = (
        em.parse_leg_spec(args.baseline, source="baseline") if args.baseline else None
    )
    sweeps = [em.parse_sweep_spec(spec) for spec in args.sweep]
    leg_set = em.assemble_legs(explicit, baseline, sweeps)
    validate_backend_torch(leg_set)
    battery = em.parse_battery(args.battery)
    tier = TIERS[args.tier]
    plan = em.build_plan(battery, leg_set, args.subset, tier, root=root)
    return Campaign(plan=plan, leg_set=leg_set, sweeps=sweeps, tier=tier)


# ----------------------------------------------------------------------
# waves: dependency-partitioned parallelism
# ----------------------------------------------------------------------


def leg_depth(leg, by_name: dict) -> int:
    """Chain depth: 0 = no init_from; a dependent is 1 + its source's depth.
    Cycles/unknown sources were already rejected by assemble_legs."""
    depth = 0
    current = leg
    while current.init_from is not None:
        current = by_name[current.init_from]
        depth += 1
    return depth


def waves(plan) -> list[list]:
    """plan.cells partitioned by chain depth, plan (render) order preserved
    within each wave. Wave n only needs finals from waves < n, so every
    wave maps to Modal in parallel."""
    by_name = {leg.name: leg for leg in plan.legs.legs}
    buckets: dict[int, list] = {}
    for cell in plan.cells:
        buckets.setdefault(leg_depth(cell.leg, by_name), []).append(cell)
    return [buckets[d] for d in sorted(buckets)]


# ----------------------------------------------------------------------
# conf transport: the container renders from the SAME conf text, except a
# chained cell's init_image is rewritten to the container-side init path.
# The LOCAL conf on disk stays eval_matrix's canonical cell_conf_text —
# cell identity and resumability semantics are untouched.
# ----------------------------------------------------------------------


def cell_slug(cell) -> str:
    return f"{cell.leg.name}__{cell.name}"


def container_workdir(cell) -> str:
    return f"{CONTAINER_WORK_ROOT}/{cell_slug(cell)}"


def container_init_path(cell) -> str:
    return f"{container_workdir(cell)}/init.png"


def final_rel(cell) -> str:
    """Final frame path relative to the cell workdir (same layout local and
    container — eval_matrix's _final_frame_path minus the workdir prefix)."""
    return f"run/images_out/{cell.name}/{cell.name}_{cell.final_index:04d}.png"


def rewrite_conf_init_image(conf_text: str, new_path: str) -> str:
    """Swap init_image in a generated cell conf, preserving the comment
    header (hydra needs the `# @package _global_` line) and key order."""
    lines = conf_text.splitlines(keepends=True)
    n_header = 0
    while n_header < len(lines) and lines[n_header].startswith("#"):
        n_header += 1
    header, body = "".join(lines[:n_header]), "".join(lines[n_header:])
    data = yaml.safe_load(body)
    if not isinstance(data, dict) or "init_image" not in data:
        die(
            "rewrite_conf_init_image called on a conf without init_image — "
            "only chained cells carry one (driver bug)"
        )
    data["init_image"] = new_path
    return header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


# ----------------------------------------------------------------------
# the Modal app (built lazily so plan/tests never import modal)
# ----------------------------------------------------------------------


def build_remote(gpu: str, timeout_s: int, max_parallel: int):
    """Ephemeral Modal app + the render_cell function.

    render_cell is fully self-contained (stdlib + the image's pinned pytti
    only) and registered with serialized=True, so it cloudpickles by value —
    no source mount, no module-identity coupling to this script.

    Error contract: a RENDER failure (nonzero exit, or exit 0 with no final
    frame) RETURNS {"ok": False, ...} — the driver dies loud with the log
    tail and modal never retries it. An INFRA failure (container crash,
    preemption, OOM-kill) raises, and retries=1 covers exactly that class.
    """
    import modal

    app = modal.App(APP_NAME)
    image = (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("git")
        .pip_install(f"torch=={TORCH_PIN}", "torchvision")
        .pip_install(PYTTI_GIT_URL)
    )
    cache = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    def render_cell(
        cell_slug: str,
        conf_name: str,
        conf_text: str,
        final_rel: str,
        init_png: bytes | None,
    ) -> dict:
        import os
        import shutil
        import subprocess
        import sys
        import time
        from pathlib import Path

        t0 = time.monotonic()

        # route every ~/.cache consumer (HF hub, pytti's ${user_cache:})
        # into the weights volume at /cache — the volume cannot mount at
        # /root/.cache itself (Modal's client layer leaves pip artifacts
        # there and volumes refuse non-empty mount paths)
        cache_root = Path("/root/.cache")
        if not cache_root.is_symlink():
            shutil.rmtree(cache_root, ignore_errors=True)
            cache_root.symlink_to("/cache")

        work = Path("/work") / cell_slug
        if work.exists():
            shutil.rmtree(work)  # container reuse: never render into leftovers
        conf_dir = work / "config" / "conf"
        conf_dir.mkdir(parents=True)

        # mirror eval_matrix.prepare_cell_workspace, from the image's own
        # pinned pytti (NOT the driver's venv copy)
        import pytti

        assets = Path(pytti.__file__).parent / "assets"
        shutil.copyfile(assets / "default.yaml", work / "config" / "default.yaml")
        (conf_dir / "_empty.yaml").write_text("\n", encoding="utf-8")
        (conf_dir / f"{conf_name}.yaml").write_text(conf_text, encoding="utf-8")
        if init_png is not None:
            # the driver rewrote this conf's init_image to exactly this path
            (work / "init.png").write_bytes(init_png)

        try:
            gpu_name = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            gpu_name = "unknown"  # telemetry only — never fails a paid render

        log_path = work / "render.log"
        cmd = [
            sys.executable,
            "-m",
            "pytti.workhorse",
            f"conf={conf_name}",
            f"hydra.run.dir={work / 'run'}",
        ]
        with open(log_path, "w", encoding="utf-8") as log_fh:
            proc = subprocess.run(
                cmd,
                cwd=work,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                # inherit the container env (CUDA vars, HOME->/root for the
                # mounted weight caches); only force unbuffered logging
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        text = log_path.read_text(encoding="utf-8", errors="replace")
        tail = "".join(text.replace("\r", "\n").splitlines(keepends=True)[-60:])
        wall_s = time.monotonic() - t0

        if proc.returncode != 0:
            return {
                "ok": False,
                "reason": f"workhorse exit code {proc.returncode}",
                "log_tail": tail,
                "wall_s": wall_s,
                "gpu_name": gpu_name,
            }
        final = work / final_rel
        if not final.is_file():
            return {
                "ok": False,
                "reason": f"exit 0 but final frame missing at {final}",
                "log_tail": tail,
                "wall_s": wall_s,
                "gpu_name": gpu_name,
            }
        return {
            "ok": True,
            "final_png": final.read_bytes(),
            "log_tail": tail,
            "wall_s": wall_s,
            "gpu_name": gpu_name,
        }

    remote = app.function(
        image=image,
        gpu=gpu,
        timeout=timeout_s,
        volumes={CACHE_MOUNT: cache},
        retries=modal.Retries(max_retries=1, initial_delay=5.0),
        max_containers=max_parallel,
        serialized=True,
    )(render_cell)
    return app, remote


# ----------------------------------------------------------------------
# driver: resumability partition + wave execution
# ----------------------------------------------------------------------


def partition_pending(plan) -> tuple[list, list]:
    """(done, pending) cells — eval_matrix's own resumability rule: final
    exists AND the local conf matches this plan (mismatch dies loud there)."""
    done, pending = [], []
    for cell in plan.cells:
        (done if em.cell_is_done(plan, cell) else pending).append(cell)
    return done, pending


def uniform_value(path: Path) -> int | None:
    """A uniform (single-value) final frame is a dead render — the classic
    all-black CUDA failure. Returns the flat luma value, else None."""
    from PIL import Image

    lo, hi = Image.open(path).convert("L").getextrema()
    return lo if lo == hi else None


# results in job order — a lazy iterator (Modal .starmap) or a list (tests)
Submit = Callable[[list[tuple]], Iterable[dict]]


def run_waves(
    plan,
    pending: list,
    submit: Submit,
    echo: Callable[[str], None] = print,
) -> dict:
    """Render pending cells wave by wave through `submit` (production: a
    Modal .starmap; tests: a fake). Returns cell_key -> result meta.

    Partial-failure contract: results stream in job order and every
    SUCCESSFUL final is persisted to the local layout the moment it
    arrives — a failed cell mid-wave never discards its wave-mates' paid
    renders (they resume as done). Failures (including uniform dead
    frames, whose finals are deleted so resumability re-renders them) are
    collected and the campaign dies at the end of the wave naming ALL of
    them; later waves never run, so a dependent never chains onto a
    failed source."""
    pending_keys = {em.cell_key(c) for c in pending}
    metas: dict = {}
    for wave_no, wave in enumerate(waves(plan), start=1):
        todo = [c for c in wave if em.cell_key(c) in pending_keys]
        if not todo:
            continue
        leg_names = sorted({c.leg.name for c in todo})
        echo(
            f"[wave {wave_no}] {len(todo)} cells -> Modal "
            f"(legs: {', '.join(leg_names)})"
        )
        jobs: list[tuple] = []
        for cell in todo:
            # a chained cell's source final must exist LOCALLY by now
            # (previous wave downloaded it, or it was resumed) — this is
            # eval_matrix's own never-render-from-a-missing-init guard
            em.require_source_final(cell)
            em.prepare_cell_workspace(plan, cell)  # canonical local conf
            conf_text = em.cell_conf_text(cell, plan.tier)
            init_png = None
            if cell.init_image is not None:
                init_png = cell.init_image.read_bytes()
                conf_text = rewrite_conf_init_image(conf_text, container_init_path(cell))
            jobs.append((cell_slug(cell), cell.name, conf_text, final_rel(cell), init_png))

        t0 = time.monotonic()
        failures: list[str] = []
        # strict zip fails loud if Modal returns a different result count
        for cell, result in zip(todo, submit(jobs), strict=True):
            tag = f"{cell.leg.name}/{cell.name}"
            workdir = em.cell_workdir(plan, cell)
            log_path = workdir / "render.log"
            log_path.write_text(result["log_tail"], encoding="utf-8")
            if not result["ok"]:
                failures.append(
                    f"{tag}: {result['reason']}\n"
                    f"--- last log lines ({log_path}) ---\n{result['log_tail']}"
                )
                continue
            final = em.final_frame_path(plan, cell)
            final.parent.mkdir(parents=True, exist_ok=True)
            final.write_bytes(result["final_png"])
            flat = uniform_value(final)
            if flat is not None:
                # a dead frame must never satisfy cell_is_done on resume
                final.unlink()
                failures.append(
                    f"{tag}: final frame is uniform (value {flat}) — dead "
                    f"render. Inspect {log_path}"
                )
                continue
            meta = {
                "wall_s": result["wall_s"],
                "gpu_name": result["gpu_name"],
                "final": str(final),
            }
            (workdir / "modal_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            metas[em.cell_key(cell)] = meta
            echo(f"[{tag}] done in {result['wall_s']:.1f}s on {result['gpu_name']}")
        echo(f"[wave {wave_no}] wall {time.monotonic() - t0:.1f}s")
        if failures:
            die(
                f"[wave {wave_no}] {len(failures)}/{len(todo)} renders failed; "
                "every successful final in this wave was saved and resumes as "
                "done — fix the cause and rerun to render only the failures:\n\n"
                + "\n\n".join(failures)
            )
    return metas


# ----------------------------------------------------------------------
# cost estimate
# ----------------------------------------------------------------------


def launch_gate(pending: list, launch: bool) -> None:
    """Spend discipline: submitting paid renders requires the explicit
    --launch flag. Planning (--dry-run) and resumed judge-only runs
    (nothing pending) are free and need no flag."""
    if pending and not launch:
        die(
            f"{len(pending)} cells would render on paid Modal GPUs — pass "
            "--launch to spend (or --dry-run to plan). Nothing was submitted."
        )


def gpu_price(gpu: str) -> float:
    if gpu not in GPU_DOLLARS_PER_HOUR:
        die(
            f"no price on file for gpu {gpu!r} — add it to "
            f"GPU_DOLLARS_PER_HOUR (known: {sorted(GPU_DOLLARS_PER_HOUR)})"
        )
    return GPU_DOLLARS_PER_HOUR[gpu]


def estimate_dollars(n_cells: int, minutes_per_cell: float, gpu: str) -> float:
    return n_cells * minutes_per_cell / 60.0 * gpu_price(gpu)


def format_modal_summary(
    plan,
    done: list,
    pending: list,
    gpu: str,
    est_minutes: float,
    max_parallel: int,
    timeout_min: float,
    est_note: str = "per-tier default, calibrated from the 2026-08 paid smoke",
) -> str:
    wave_list = waves(plan)
    pending_keys = {em.cell_key(c) for c in pending}
    lines = [
        "modal execution plan:",
        f"  app {APP_NAME} (ephemeral) · image: debian-slim py{PYTHON_VERSION} "
        f"+ torch=={TORCH_PIN} + pytti-core@{PINNED_COMMIT[:7]}",
        f"  weights volume {VOLUME_NAME!r} at {CACHE_MOUNT}",
        f"  gpu {gpu} (${gpu_price(gpu):.2f}/h) · max parallel {max_parallel} · "
        f"timeout {timeout_min:g} min/cell",
        f"  local root {plan.root} (never mixed with local-MPS {em.EVAL_ROOT})",
    ]
    for i, wave in enumerate(wave_list, start=1):
        todo = [c for c in wave if em.cell_key(c) in pending_keys]
        legs = sorted({c.leg.name for c in wave})
        lines.append(
            f"  wave {i}: {len(todo)}/{len(wave)} cells to render "
            f"(legs: {', '.join(legs)})"
        )
    cost = estimate_dollars(len(pending), est_minutes, gpu)
    lines.append(
        f"  cells: {len(plan.cells)} total, {len(done)} done (skipped), "
        f"{len(pending)} to render"
    )
    lines.append(
        f"  cost estimate: {len(pending)} cells x {est_minutes:g} min x "
        f"${gpu_price(gpu):.2f}/h = ${cost:.2f} ({est_note})"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # mirrored eval_matrix flags (same semantics, same parsing functions)
    parser.add_argument(
        "--legs",
        action="append",
        default=[],
        metavar="NAME[:K=V,K=V...]",
        help="a named leg of overrides (repeatable); init_from=SOURCELEG "
        "chains this leg onto another leg's final frames",
    )
    parser.add_argument("--baseline", default=None, metavar="NAME[:K=V,...]")
    parser.add_argument(
        "--sweep", action="append", default=[], metavar="FIELD=V1,V2,..."
    )
    parser.add_argument(
        "--tier",
        choices=sorted(TIERS),
        default="screening",
        help="render budget; 'smoke' (modal_eval-only) = 128px/20 steps "
        "for the paid integration smoke",
    )
    parser.add_argument("--battery", type=Path, default=em.DEFAULT_BATTERY)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"report directory (default {MODAL_EVAL_ROOT}/report-<tier>; wiped)",
    )
    parser.add_argument("--subset", type=int, default=None, metavar="N")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan, waves and cost estimate; render nothing",
    )
    # modal-only flags
    parser.add_argument(
        "--launch",
        action="store_true",
        help="actually submit paid renders to Modal — without this (or "
        "--dry-run) the plan + cost estimate print and nothing is spent",
    )
    parser.add_argument(
        "--gpu",
        choices=sorted(GPU_DOLLARS_PER_HOUR),
        default="L4",
        help="Modal GPU type (default L4 — cheapest current-gen part; the "
        "smoke calibrates whether A10G buys its 1.4x price)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=10,
        metavar="N",
        help="max concurrent Modal containers (default 10)",
    )
    parser.add_argument(
        "--timeout-min",
        type=float,
        default=20.0,
        metavar="MIN",
        help="per-cell Modal timeout in minutes (default 20)",
    )
    parser.add_argument(
        "--est-cell-minutes",
        type=float,
        default=None,
        metavar="MIN",
        help="wall minutes per cell for the cost estimate (default: "
        f"per-tier guesses {DEFAULT_EST_MINUTES})",
    )
    args = parser.parse_args(argv)
    if args.max_parallel < 1:
        die(f"--max-parallel {args.max_parallel} must be >= 1")
    if args.timeout_min <= 0:
        die(f"--timeout-min {args.timeout_min} must be > 0")
    return args


def main() -> None:
    args = parse_args()
    campaign = build_campaign(args)
    plan = campaign.plan
    if args.est_cell_minutes is not None:
        est_minutes, est_note = args.est_cell_minutes, "--est-cell-minutes"
    else:
        est_minutes = DEFAULT_EST_MINUTES[campaign.tier.name]
        est_note = "per-tier default, calibrated from the 2026-08 paid smoke"

    done, pending = partition_pending(plan)

    if args.dry_run:
        print(em.format_plan(plan, sys.executable))
        print(
            format_modal_summary(
                plan, done, pending, args.gpu, est_minutes,
                args.max_parallel, args.timeout_min, est_note,
            )
        )
        return

    for cell in done:
        print(f"[{cell.leg.name}/{cell.name}] final frame exists — skipping")

    metas: dict = {}
    if pending:
        print(
            format_modal_summary(
                plan, done, pending, args.gpu, est_minutes,
                args.max_parallel, args.timeout_min, est_note,
            )
        )
        launch_gate(pending, args.launch)
        import modal

        app, remote = build_remote(
            gpu=args.gpu,
            timeout_s=int(args.timeout_min * 60),
            max_parallel=args.max_parallel,
        )
        with modal.enable_output(), app.run():
            metas = run_waves(
                plan,
                pending,
                # NOT list()-ed: results stream so run_waves persists each
                # success the moment it lands (partial-failure contract)
                submit=remote.starmap,
            )
        total_s = sum(m["wall_s"] for m in metas.values())
        billed = total_s / 3600.0 * gpu_price(args.gpu)
        print(
            f"modal renders: {len(metas)} cells, GPU wall {total_s / 60:.1f} min "
            f"(mean {total_s / len(metas) / 60:.2f} min/cell), "
            f"est billed ${billed:.2f} on {args.gpu} "
            "(excludes container boot + first-ever weight downloads)"
        )
    print(f"renders: {len(metas)} run, {len(done)} skipped (resumed)")

    # judging + artifacts: eval_matrix's local path, verbatim
    all_scores = em.judge_all(plan)
    sweep_legs = [leg for leg in campaign.leg_set.legs if leg.source == "sweep"]
    trends = [
        (sweep, em.trend_table(all_scores, sweep, sweep_legs, campaign.leg_set.baseline))
        for sweep in campaign.sweeps
    ]
    rows = em.prompt_rows(all_scores, plan)

    out_dir: Path = (
        args.out if args.out is not None else MODAL_EVAL_ROOT / f"report-{plan.tier.name}"
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    compare_dir = out_dir / "compare"
    compare_dir.mkdir()
    for prompt in plan.prompts:
        em.compare_sheet(plan, prompt, all_scores, compare_dir / f"{prompt.id}.png")
    for leg in campaign.leg_set.legs:
        em.contact_grid(plan, leg, all_scores, out_dir / f"grid_{leg.name}.png")
    em.write_metrics_json(plan, all_scores, trends, rows, out_dir)
    report_path = em.write_report(plan, all_scores, trends, rows, out_dir)

    for judge in em.JUDGES:
        summary = "   ".join(
            f"{leg.name}={em.leg_stats(all_scores[judge], leg.name).mean:.4f}"
            for leg in campaign.leg_set.legs
        )
        print(f"{judge}: {summary}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
