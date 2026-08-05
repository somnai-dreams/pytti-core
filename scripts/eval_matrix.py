#!/usr/bin/env python
"""
Prompt-battery eval matrix: render a battery of prompts under N config
"legs" and judge every final frame with BOTH held-out judge families,
reported per judge — replacing single-prompt single-seed judgment with a
real basis.

Why two judges, always: judge-family bias is established — ViT-L/14 and
SigLIP2-SO400M disagree across looks — so a verdict from one family is a
coin with a thumb on it. The report never averages across judges; the
per-prompt disagreement table is signal, not noise to smooth away. The
golden parity fixtures (tests/fixtures/golden/) have NO quality authority
here — the battery (tests/fixtures/eval/battery.yaml) is the basis.

Tier (--tier): screening (default) — 256-class dims scaled per aspect
('1:1' -> 256x256, '3:4' -> 240x320), 150 steps; full — production dims
('1:1' -> 512x512, '3:4' -> 448x576), 200 steps. Both save_every 25,
backups 0, schema defaults otherwise. Screening ranks legs cheaply; full
certifies at production dims. Cell workspaces live under
/tmp/pytti-eval/<tier>/ so the tiers coexist; dims are tier-owned per
aspect and never leg-overridable.

Usage (leg mode — named legs, each a set of overrides on the shared base):
  .venv/bin/python scripts/eval_matrix.py \\
      --legs default \\
      --legs full16:cutout_sampler=full,cutouts=16 \\
      --out /tmp/pytti-eval/report

Usage (sweep mode — cross-product legs + a TREND table per swept field):
  .venv/bin/python scripts/eval_matrix.py \\
      --baseline default --sweep cutouts=8,16,40 --subset 3

--sweep field=v1,v2,... (repeatable) auto-generates legs named
field-value[_field2-value2] as the cross-product across swept fields.
Win-rates anchor to --baseline (or, when sweeping, to the sole --legs leg).
--subset N keeps the first N battery prompts for cheap sweeps. Always plan
first: --dry-run prints the full cell matrix + commands and renders nothing.

Per cell: `python -m pytti.workhorse` renders in its own scratch workspace
under /tmp/pytti-eval/<tier>/<leg>/<prompt>_s<seed>/ (detached process, log
polled; a crash is fatal with the log tail). Resumable: a cell whose final
frame already exists — and whose generated conf matches this plan — is
skipped. Judging reuses scripts/judge_stills.py machinery in-process (one
tower load per family, not one subprocess per image).

Artifacts: per-leg contact grids, metrics.json, report.md, and — the
primary human-comparison artifact — prompt-major compare sheets in
<out>/compare/: one PNG per prompt, columns = legs, rows = seeds, every
tile labeled with both judges' scores, missing cells marked.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_BATTERY = REPO_ROOT / "tests" / "fixtures" / "eval" / "battery.yaml"
EVAL_ROOT = Path("/tmp/pytti-eval")

# both, always — the families disagree across looks; report per judge only
JUDGES = ("ViTL14", "SigLIP2SO400M")
TIE_THRESHOLD = 0.002

# the aspect vocabulary the battery may use; every tier maps each aspect
ASPECTS = ("1:1", "3:4")


@dataclass(frozen=True)
class Tier:
    """Render budget for one campaign: dims per aspect + step/save schedule.
    Dims are tier-owned cell identity — never leg-overridable."""

    name: str
    dims: dict[str, tuple[int, int]]  # aspect -> (width, height)
    steps: int
    save_every: int

    def describe(self) -> str:
        dims = ", ".join(f"{a} -> {w}x{h}" for a, (w, h) in self.dims.items())
        return (
            f"{self.name} — {dims}, {self.steps} steps, "
            f"save_every {self.save_every}, backups 0, schema defaults otherwise"
        )


TIERS = {
    # screening ranks legs cheaply; it does not certify a look
    "screening": Tier(
        name="screening",
        dims={"1:1": (256, 256), "3:4": (240, 320)},
        steps=150,
        save_every=25,
    ),
    # full certifies at production dims
    "full": Tier(
        name="full",
        dims={"1:1": (512, 512), "3:4": (448, 576)},
        steps=200,
        save_every=25,
    ),
}
assert all(
    tuple(tier.dims) == ASPECTS for tier in TIERS.values()
), f"every tier must map exactly the battery aspects {ASPECTS}"

_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")
# progress heartbeat only: "  12/150 [" from a tqdm bar
_TQDM_RE = re.compile(r"(\d+)/(\d+) \[")
POLL_S = 5.0

BATTERY_KEYS = frozenset({"seeds", "prompts"})
PROMPT_KEYS = ("id", "category", "aspect", "scenes", "rationale")

# cell identity comes from the battery + harness; a leg must not fork it
FORBIDDEN_OVERRIDES = frozenset(
    {"scenes", "seed", "width", "height", "file_namespace", "allow_overwrite", "restore"}
)


def die(msg: str) -> NoReturn:
    raise SystemExit(f"eval_matrix: error: {msg}")


def _judge_stills():
    """scripts/ is not a package; make `import judge_stills` resolvable."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import judge_stills

    return judge_stills


_SCHEMA_FIELDS: frozenset[str] | None = None


def schema_field_names() -> frozenset[str]:
    global _SCHEMA_FIELDS
    if _SCHEMA_FIELDS is None:
        import attrs

        from pytti.config.structured_config import ConfigSchema

        _SCHEMA_FIELDS = frozenset(f.name for f in attrs.fields(ConfigSchema))
    return _SCHEMA_FIELDS


# ----------------------------------------------------------------------
# battery
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BatteryPrompt:
    id: str
    category: str
    aspect: str
    scenes: str
    rationale: str
    judge_text: str  # scenes reduced to plain judge text (weights stripped)


@dataclass(frozen=True)
class Battery:
    path: Path
    seeds: tuple[int, ...]
    prompts: tuple[BatteryPrompt, ...]


def parse_battery(path: Path) -> Battery:
    if not path.is_file():
        die(f"battery file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        die(f"{path} did not parse to a mapping")
    unknown = set(data) - BATTERY_KEYS
    if unknown:
        die(f"{path}: unknown top-level keys {sorted(unknown)}")

    seeds = data.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        die(f"{path}: 'seeds' must be a non-empty list of ints")
    for s in seeds:
        if not isinstance(s, int) or isinstance(s, bool):
            die(f"{path}: seed {s!r} is not an int")
    if len(set(seeds)) != len(seeds):
        die(f"{path}: duplicate seeds {seeds}")

    raw_prompts = data.get("prompts")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        die(f"{path}: 'prompts' must be a non-empty list")

    scene_to_text = _judge_stills().scene_to_text
    prompts: list[BatteryPrompt] = []
    seen: set[str] = set()
    for i, item in enumerate(raw_prompts):
        where = f"{path}: prompts[{i}]"
        if not isinstance(item, dict):
            die(f"{where} is not a mapping")
        unknown = set(item) - set(PROMPT_KEYS)
        if unknown:
            die(f"{where}: unknown keys {sorted(unknown)}")
        for key in PROMPT_KEYS:
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                die(f"{where}: {key!r} must be a non-empty string")
        pid = item["id"]
        if not _LABEL_RE.match(pid):
            die(f"{where}: id {pid!r} must match {_LABEL_RE.pattern}")
        if pid in seen:
            die(f"{where}: duplicate id {pid!r}")
        seen.add(pid)
        if item["aspect"] not in ASPECTS:
            die(f"{where}: aspect {item['aspect']!r} not in {sorted(ASPECTS)}")
        prompts.append(
            BatteryPrompt(
                id=pid,
                category=item["category"],
                aspect=item["aspect"],
                scenes=item["scenes"],
                rationale=item["rationale"],
                # single scene, no image prompts — scene_to_text enforces both
                judge_text=scene_to_text(item["scenes"]),
            )
        )
    return Battery(path=path, seeds=tuple(seeds), prompts=tuple(prompts))


# ----------------------------------------------------------------------
# legs and sweeps
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LegSpec:
    name: str
    overrides: tuple[tuple[str, str], ...]  # ordered (key, raw value string)
    source: str  # "explicit" | "baseline" | "sweep"
    sweep: tuple[tuple[str, str], ...] = ()  # swept (field, value) assignment

    def overrides_dict(self) -> dict[str, str]:
        return dict(self.overrides)

    def overrides_text(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.overrides) or "(schema defaults)"


def _validate_override_key(key: str, context: str) -> None:
    if key not in schema_field_names():
        die(f"{context}: {key!r} is not a ConfigSchema field")
    if key in FORBIDDEN_OVERRIDES:
        die(
            f"{context}: {key!r} is cell identity (owned by the battery/"
            "harness), not a leg override"
        )


def _validate_judge_held_out(key: str, raw: str, context: str) -> None:
    if key in JUDGES and yaml.safe_load(raw):
        die(
            f"{context}: {key}={raw} would put judge {key!r} into the "
            "optimization ensemble — judges must stay held out"
        )


def parse_overrides(text: str, context: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in text.split(","):
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            die(f"{context}: override {item!r} must be key=value")
        _validate_override_key(key, context)
        _validate_judge_held_out(key, value, context)
        if key in seen:
            die(f"{context}: duplicate override key {key!r}")
        seen.add(key)
        pairs.append((key, value))
    return tuple(pairs)


def parse_leg_spec(spec: str, source: str = "explicit") -> LegSpec:
    name, sep, rest = spec.partition(":")
    if not _LABEL_RE.match(name):
        die(f"leg spec {spec!r}: name {name!r} must match {_LABEL_RE.pattern}")
    if sep and not rest.strip():
        die(f"leg spec {spec!r}: empty overrides after ':'")
    overrides = parse_overrides(rest, f"leg {name!r}") if sep else ()
    return LegSpec(name=name, overrides=overrides, source=source)


@dataclass(frozen=True)
class SweepField:
    field: str
    values: tuple[str, ...]


def parse_sweep_spec(spec: str) -> SweepField:
    context = f"--sweep {spec!r}"
    field, sep, rest = spec.partition("=")
    field = field.strip()
    if not sep or not field or not rest.strip():
        die(f"{context}: must be field=v1,v2,...")
    _validate_override_key(field, context)
    values = tuple(v.strip() for v in rest.split(","))
    if any(not v for v in values):
        die(f"{context}: empty value")
    if len(values) < 2:
        die(f"{context}: a sweep needs at least 2 values")
    if len(set(values)) != len(values):
        die(f"{context}: duplicate values")
    for v in values:
        _validate_judge_held_out(field, v, context)
    return SweepField(field=field, values=values)


def expand_sweeps(sweeps: list[SweepField]) -> list[LegSpec]:
    """Cross-product swept fields into legs named field-value[_field2-value2]."""
    if not sweeps:
        return []
    fields = [s.field for s in sweeps]
    if len(set(fields)) != len(fields):
        die(f"duplicate swept fields in {fields}")
    axes = [[(s.field, v) for v in s.values] for s in sweeps]
    legs: list[LegSpec] = []
    for combo in itertools.product(*axes):
        name = "_".join(f"{f}-{v}" for f, v in combo)
        if not _LABEL_RE.match(name):
            die(f"sweep leg name {name!r} is not filesystem-safe ({_LABEL_RE.pattern})")
        legs.append(
            LegSpec(name=name, overrides=tuple(combo), source="sweep", sweep=tuple(combo))
        )
    return legs


@dataclass(frozen=True)
class LegSet:
    legs: tuple[LegSpec, ...]
    baseline: str | None  # leg name that win-rate-vs-baseline anchors to


def assemble_legs(
    explicit: list[LegSpec], baseline: LegSpec | None, sweeps: list[SweepField]
) -> LegSet:
    sweep_legs = expand_sweeps(sweeps)
    ordered = ([baseline] if baseline else []) + explicit + sweep_legs
    if not ordered:
        die("no legs: pass --legs and/or --sweep (and optionally --baseline)")
    names = [leg.name for leg in ordered]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        die(f"duplicate leg names {dupes}")
    if baseline is not None:
        base_name = baseline.name
    elif sweeps and len(explicit) == 1:
        # the sole explicit leg anchors sweep win-rates (named in the report)
        base_name = explicit[0].name
    else:
        base_name = None
    return LegSet(legs=tuple(ordered), baseline=base_name)


# ----------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    leg: LegSpec
    prompt: BatteryPrompt
    seed: int
    width: int
    height: int
    steps: int
    save_every: int

    @property
    def name(self) -> str:
        return f"{self.prompt.id}_s{self.seed}"

    @property
    def final_index(self) -> int:
        return self.steps // self.save_every


@dataclass(frozen=True)
class Plan:
    battery: Battery
    prompts: tuple[BatteryPrompt, ...]  # after --subset
    legs: LegSet
    tier: Tier
    root: Path  # tier-scoped: <eval root>/<tier>
    cells: tuple[Cell, ...]


CellKey = tuple[str, str, int]  # (leg name, prompt id, seed)


def cell_key(cell: Cell) -> CellKey:
    return (cell.leg.name, cell.prompt.id, cell.seed)


def leg_int(leg: LegSpec, key: str, default: int) -> int:
    raw = leg.overrides_dict().get(key)
    if raw is None:
        return default
    value = yaml.safe_load(raw)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        die(f"leg {leg.name!r}: {key}={raw!r} must be a positive int")
    return value


def build_plan(
    battery: Battery,
    leg_set: LegSet,
    subset: int | None,
    tier: Tier,
    root: Path = EVAL_ROOT,
) -> Plan:
    prompts = battery.prompts
    if subset is not None:
        if not 1 <= subset <= len(prompts):
            die(f"--subset {subset} out of range 1..{len(prompts)}")
        prompts = prompts[:subset]
    cells: list[Cell] = []
    for leg in leg_set.legs:
        steps = leg_int(leg, "steps_per_scene", tier.steps)
        save_every = leg_int(leg, "save_every", tier.save_every)
        if steps % save_every:
            die(
                f"leg {leg.name!r}: steps_per_scene {steps} is not a multiple of "
                f"save_every {save_every} — the final frame index would be ambiguous"
            )
        for prompt in prompts:
            width, height = tier.dims[prompt.aspect]
            for seed in battery.seeds:
                cells.append(
                    Cell(
                        leg=leg,
                        prompt=prompt,
                        seed=seed,
                        width=width,
                        height=height,
                        steps=steps,
                        save_every=save_every,
                    )
                )
    return Plan(
        battery=battery,
        prompts=prompts,
        legs=leg_set,
        tier=tier,
        # tier-scoped so screening and full campaigns coexist (and a tier
        # switch can never trip the conf-mismatch resumability check)
        root=root / tier.name,
        cells=tuple(cells),
    )


def cell_workdir(plan: Plan, cell: Cell) -> Path:
    return plan.root / cell.leg.name / cell.name


def cell_conf_path(plan: Plan, cell: Cell) -> Path:
    return cell_workdir(plan, cell) / "config" / "conf" / f"{cell.name}.yaml"


def final_frame_path(plan: Plan, cell: Cell) -> Path:
    # pytti.ImageGuide.frame_filename: {namespace}_{n:04d}.png, n = step/save_every
    return (
        cell_workdir(plan, cell)
        / "run"
        / "images_out"
        / cell.name
        / f"{cell.name}_{cell.final_index:04d}.png"
    )


def cell_conf_text(cell: Cell, tier: Tier) -> str:
    cfg: dict[str, object] = {
        "scenes": cell.prompt.scenes,
        "width": cell.width,
        "height": cell.height,
        "seed": cell.seed,
        "steps_per_scene": tier.steps,
        "save_every": tier.save_every,
        "backups": 0,
        "file_namespace": cell.name,
    }
    for key, raw in cell.leg.overrides:
        cfg[key] = yaml.safe_load(raw)
    header = (
        "# @package _global_\n"
        f"# generated by scripts/eval_matrix.py — leg {cell.leg.name!r}, "
        f"cell {cell.name!r}\n"
        f"# category: {cell.prompt.category} · aspect {cell.prompt.aspect} · "
        f"{tier.describe()}\n"
    )
    return header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


def cell_command(plan: Plan, cell: Cell, python_exe: str) -> list[str]:
    workdir = cell_workdir(plan, cell)
    return [
        python_exe,
        "-m",
        "pytti.workhorse",
        f"conf={cell.name}",
        f"hydra.run.dir={workdir / 'run'}",
    ]


def format_plan(plan: Plan, python_exe: str) -> str:
    lines = ["pytti eval matrix — dry-run plan"]
    subset_note = (
        f"; subset -> first {len(plan.prompts)}"
        if len(plan.prompts) != len(plan.battery.prompts)
        else ""
    )
    lines.append(
        f"battery: {plan.battery.path} — {len(plan.battery.prompts)} prompts, "
        f"seeds {list(plan.battery.seeds)}{subset_note}"
    )
    lines.append(f"tier: {plan.tier.describe()}")
    lines.append(f"judges: {' + '.join(JUDGES)} (always both; reported per judge)")
    base = plan.legs.baseline
    lines.append(f"baseline: {base if base else '(none — no win-rate-vs-baseline columns)'}")
    lines.append("")

    leg_w = max(len(leg.name) for leg in plan.legs.legs)
    src_w = max(len(leg.source) for leg in plan.legs.legs)
    lines.append(f"legs ({len(plan.legs.legs)}):")
    for leg in plan.legs.legs:
        mark = " (baseline)" if leg.name == base else ""
        lines.append(
            f"  {leg.name:<{leg_w}}  [{leg.source:<{src_w}}]  "
            f"{leg.overrides_text()}{mark}"
        )
    lines.append("")

    n_legs, n_prompts, n_seeds = (
        len(plan.legs.legs),
        len(plan.prompts),
        len(plan.battery.seeds),
    )
    lines.append(
        f"cells ({len(plan.cells)} = {n_legs} legs x {n_prompts} prompts x {n_seeds} seeds):"
    )
    cell_w = max(len("cell"), *(len(c.name) for c in plan.cells))
    cat_w = max(len("category"), *(len(c.prompt.category) for c in plan.cells))
    lw = max(len("leg"), leg_w)
    lines.append(
        f"  {'leg':<{lw}}  {'cell':<{cell_w}}  {'category':<{cat_w}}  "
        f"{'size':<8}  {'steps':<5}  final"
    )
    for cell in plan.cells:
        size = f"{cell.width}x{cell.height}"
        lines.append(
            f"  {cell.leg.name:<{lw}}  {cell.name:<{cell_w}}  "
            f"{cell.prompt.category:<{cat_w}}  {size:<8}  {cell.steps:<5}  "
            f"{final_frame_path(plan, cell)}"
        )
    lines.append("")

    lines.append("commands (per cell; cwd = the cell workspace):")
    for cell in plan.cells:
        cmd = shlex.join(cell_command(plan, cell, python_exe))
        lines.append(f"  cd {cell_workdir(plan, cell)} && {cmd}")
    lines.append("")
    lines.append("dry run: nothing rendered, nothing written.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# rendering (real runs only — never on battery power)
# ----------------------------------------------------------------------


def pytti_assets_dir() -> Path:
    import pytti

    assets = Path(pytti.__file__).parent / "assets"
    if not (assets / "default.yaml").is_file():
        die(f"pytti assets/default.yaml not found under {assets}")
    return assets


def prepare_cell_workspace(plan: Plan, cell: Cell) -> Path:
    """Scratch cwd for one cell: local hydra config dir + run dir."""
    workdir = cell_workdir(plan, cell)
    if workdir.exists():
        shutil.rmtree(workdir)  # stale partial — only complete cells are skipped
    conf_dir = workdir / "config" / "conf"
    conf_dir.mkdir(parents=True)
    shutil.copyfile(pytti_assets_dir() / "default.yaml", workdir / "config" / "default.yaml")
    (conf_dir / "_empty.yaml").write_text("\n", encoding="utf-8")
    cell_conf_path(plan, cell).write_text(cell_conf_text(cell, plan.tier), encoding="utf-8")
    return workdir


def cell_is_done(plan: Plan, cell: Cell) -> bool:
    """Resumability check: final frame exists AND its conf matches this plan."""
    final = final_frame_path(plan, cell)
    if not final.is_file():
        return False
    conf = cell_conf_path(plan, cell)
    if not conf.is_file() or conf.read_text(encoding="utf-8") != cell_conf_text(cell, plan.tier):
        die(
            f"{cell.leg.name}/{cell.name}: final frame exists but its conf does "
            f"not match this plan (leg redefined under the same name?). "
            f"Wipe {cell_workdir(plan, cell)} and rerun."
        )
    return True


def _last_progress(log_path: Path) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    matches = _TQDM_RE.findall(text)
    if not matches:
        return ""
    n, total = matches[-1]
    return f"step {n}/{total}"


def run_cell(plan: Plan, cell: Cell) -> None:
    workdir = prepare_cell_workspace(plan, cell)
    log_path = workdir / "render.log"
    cmd = cell_command(plan, cell, sys.executable)
    tag = f"{cell.leg.name}/{cell.name}"
    print(f"[{tag}] rendering: {shlex.join(cmd)} (cwd {workdir})")
    t0 = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log_fh:
        # nohup-style: detached session, output to the log, poll for progress
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        last = ""
        while proc.poll() is None:
            time.sleep(POLL_S)
            progress = _last_progress(log_path)
            if progress and progress != last:
                print(f"[{tag}] {progress}")
                last = progress
    if proc.returncode != 0:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        tail = "".join(text.replace("\r", "\n").splitlines(keepends=True)[-30:])
        die(
            f"render for {tag} failed with exit code {proc.returncode}.\n"
            f"--- last log lines ({log_path}) ---\n{tail}"
        )
    final = final_frame_path(plan, cell)
    if not final.is_file():
        die(f"render for {tag} exited 0 but final frame {final} is missing")
    print(f"[{tag}] done in {time.monotonic() - t0:.1f}s")


# ----------------------------------------------------------------------
# judging (reuses scripts/judge_stills.py in-process)
# ----------------------------------------------------------------------

Scores = dict[str, dict[CellKey, float]]  # judge -> cell -> mean-view cosine


def judge_all(plan: Plan) -> Scores:
    import gc

    import torch

    from pytti.device import resolve_device

    js = _judge_stills()
    device = resolve_device("auto")
    all_scores: Scores = {}
    for judge_key in JUDGES:
        print(f"[judge {judge_key}] loading on {device} ...")
        judge = js.load_judge(judge_key, device)
        scores: dict[CellKey, float] = {}
        for prompt in plan.prompts:
            cells = [c for c in plan.cells if c.prompt.id == prompt.id]
            paths = [final_frame_path(plan, c) for c in cells]
            for c, s in zip(
                cells, js.score_images(judge, paths, prompt.judge_text, device), strict=True
            ):
                scores[cell_key(c)] = s.score
        all_scores[judge_key] = scores
        print(f"[judge {judge_key}] scored {len(scores)} finals")
        del judge
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return all_scores


# ----------------------------------------------------------------------
# aggregation (pure — unit-tested on synthetic scores)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class WinRecord:
    wins: int
    losses: int
    ties: int

    @property
    def n(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        if self.n == 0:
            die("win_rate over zero comparisons")
        return self.wins / self.n

    def text(self) -> str:
        return f"{self.wins}W/{self.losses}L/{self.ties}T ({self.win_rate:.2f})"


def leg_scores(scores_j: dict[CellKey, float], leg: str) -> dict[tuple[str, int], float]:
    return {(p, s): v for (name, p, s), v in scores_j.items() if name == leg}


@dataclass(frozen=True)
class LegStats:
    mean: float
    median: float
    n: int


def leg_stats(scores_j: dict[CellKey, float], leg: str) -> LegStats:
    values = sorted(leg_scores(scores_j, leg).values())
    if not values:
        die(f"no scores for leg {leg!r}")
    return LegStats(
        mean=sum(values) / len(values), median=statistics.median(values), n=len(values)
    )


def compare_legs(
    scores_j: dict[CellKey, float],
    leg_a: str,
    leg_b: str,
    tie_threshold: float = TIE_THRESHOLD,
) -> WinRecord:
    """A-vs-B over shared (prompt, seed) cells; |diff| < tie_threshold is a tie."""
    cells_a, cells_b = leg_scores(scores_j, leg_a), leg_scores(scores_j, leg_b)
    if not cells_a:
        die(f"no scores for leg {leg_a!r}")
    if set(cells_a) != set(cells_b):
        die(
            f"cell sets differ between legs {leg_a!r} and {leg_b!r}: "
            f"{sorted(set(cells_a) ^ set(cells_b))}"
        )
    wins = losses = ties = 0
    for key in sorted(cells_a):
        diff = cells_a[key] - cells_b[key]
        if abs(diff) < tie_threshold:
            ties += 1
        elif diff > 0:
            wins += 1
        else:
            losses += 1
    return WinRecord(wins=wins, losses=losses, ties=ties)


def _avg_ranks(values: dict[str, float]) -> dict[str, float]:
    """Rank 1 = highest score; exactly-equal scores share the average rank."""
    ordered = sorted(values.items(), key=lambda kv: -kv[1])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        avg = (i + 1 + j) / 2  # average of ranks i+1 .. j
        for k in range(i, j):
            ranks[ordered[k][0]] = avg
        i = j
    return ranks


def rank_disagreement(means_a: dict[str, float], means_b: dict[str, float]) -> float:
    """
    Normalized Spearman-footrule distance between two judges' leg rankings:
    0 = identical order, 1 = fully reversed. Scale-free, so the judges'
    different cosine ranges cannot masquerade as (dis)agreement.
    """
    if set(means_a) != set(means_b):
        die(f"leg sets differ: {sorted(set(means_a) ^ set(means_b))}")
    n = len(means_a)
    if n < 2:
        return 0.0
    ranks_a, ranks_b = _avg_ranks(means_a), _avg_ranks(means_b)
    footrule = sum(abs(ranks_a[k] - ranks_b[k]) for k in means_a)
    return footrule / (n * n // 2)


def prompt_leg_mean(scores_j: dict[CellKey, float], leg: str, prompt_id: str) -> float:
    values = [
        v for (name, p, _), v in scores_j.items() if name == leg and p == prompt_id
    ]
    if not values:
        die(f"no scores for leg {leg!r} prompt {prompt_id!r}")
    return sum(values) / len(values)


@dataclass(frozen=True)
class TrendRow:
    value: str
    legs: tuple[str, ...]  # sweep legs carrying this value
    means: dict[str, float]  # judge -> mean over all their cells
    vs_baseline: dict[str, WinRecord] | None  # judge -> record; None if no baseline


def trend_table(
    all_scores: Scores,
    sweep: SweepField,
    legs: list[LegSpec],
    baseline: str | None,
) -> list[TrendRow]:
    """Per swept value (pooled across cross-product combos): per-judge mean
    + win-rate vs the baseline leg over matched (prompt, seed) cells."""
    rows: list[TrendRow] = []
    for value in sweep.values:
        value_legs = [
            leg.name for leg in legs if dict(leg.sweep).get(sweep.field) == value
        ]
        if not value_legs:
            die(f"sweep {sweep.field!r}: no legs carry value {value!r}")
        means: dict[str, float] = {}
        vs: dict[str, WinRecord] | None = {} if baseline is not None else None
        for judge, scores_j in all_scores.items():
            pooled = [v for leg in value_legs for v in leg_scores(scores_j, leg).values()]
            if not pooled:
                die(f"sweep {sweep.field!r}={value!r}: no scores under judge {judge!r}")
            means[judge] = sum(pooled) / len(pooled)
            if vs is not None:
                assert baseline is not None
                wins = losses = ties = 0
                for leg in value_legs:
                    record = compare_legs(scores_j, leg, baseline)
                    wins, losses, ties = (
                        wins + record.wins,
                        losses + record.losses,
                        ties + record.ties,
                    )
                vs[judge] = WinRecord(wins=wins, losses=losses, ties=ties)
        rows.append(TrendRow(value=value, legs=tuple(value_legs), means=means, vs_baseline=vs))
    return rows


@dataclass(frozen=True)
class PromptRow:
    prompt: BatteryPrompt
    means: dict[str, dict[str, float]]  # judge -> leg -> mean over seeds
    disagreement: float


def prompt_rows(all_scores: Scores, plan: Plan) -> list[PromptRow]:
    leg_names = [leg.name for leg in plan.legs.legs]
    rows: list[PromptRow] = []
    for prompt in plan.prompts:
        means = {
            judge: {
                leg: prompt_leg_mean(scores_j, leg, prompt.id) for leg in leg_names
            }
            for judge, scores_j in all_scores.items()
        }
        rows.append(
            PromptRow(
                prompt=prompt,
                means=means,
                disagreement=rank_disagreement(means[JUDGES[0]], means[JUDGES[1]]),
            )
        )
    rows.sort(key=lambda r: -r.disagreement)  # stable: battery order within ties
    return rows


# ----------------------------------------------------------------------
# artifacts (PIL only — no plotting deps; matches ab_render conventions)
# ----------------------------------------------------------------------

_GAP = 4
_MARGIN = 10


def _font(size: int):
    from PIL import ImageFont

    return ImageFont.load_default(size=size)


def contact_grid(
    plan: Plan, leg: LegSpec, all_scores: Scores, out_path: Path, thumb_w: int = 192
) -> None:
    """All cells of one leg tiled: rows = prompts, cols = seeds, labeled with
    per-judge scores."""
    from PIL import Image, ImageDraw

    seeds = plan.battery.seeds
    label_h, header_h = 18, 26
    row_heights = [
        round(thumb_w * plan.tier.dims[p.aspect][1] / plan.tier.dims[p.aspect][0])
        for p in plan.prompts
    ]
    width = 2 * _MARGIN + len(seeds) * thumb_w + (len(seeds) - 1) * _GAP
    height = (
        2 * _MARGIN
        + header_h
        + sum(label_h + h for h in row_heights)
        + (len(plan.prompts) - 1) * _GAP
    )
    sheet = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    small, med = _font(11), _font(14)
    draw.text(
        (_MARGIN, _MARGIN),
        f"leg: {leg.name} — {leg.overrides_text()}   "
        f"[scores: {JUDGES[0]}/{JUDGES[1]}]",
        font=med,
        fill=(255, 255, 255),
    )
    cells_by_key = {cell_key(c): c for c in plan.cells}
    y = _MARGIN + header_h
    for prompt, row_h in zip(plan.prompts, row_heights, strict=True):
        for col, seed in enumerate(seeds):
            x = _MARGIN + col * (thumb_w + _GAP)
            key = (leg.name, prompt.id, seed)
            label = (
                f"{prompt.id} s{seed}  "
                f"{all_scores[JUDGES[0]][key]:.3f}/{all_scores[JUDGES[1]][key]:.3f}"
            )
            draw.text((x, y + 2), label, font=small, fill=(200, 200, 200))
            frame = Image.open(final_frame_path(plan, cells_by_key[key]))
            sheet.paste(frame.resize((thumb_w, row_h), Image.LANCZOS), (x, y + label_h))
        y += label_h + row_h + _GAP
    sheet.save(out_path)


def compare_sheet(
    plan: Plan,
    prompt: BatteryPrompt,
    all_scores: Scores,
    out_path: Path,
    tile_w: int = 256,
) -> None:
    """
    Prompt-major comparison sheet — the primary human-comparison artifact:
    one PNG per prompt, title = scenes text + category, columns = legs with
    big name headers, rows = seeds, every tile labeled
    's<seed>  <ViTL14>/<SO400M>'. A missing final frame or score renders as
    a marked placeholder instead of dying: the sheet is for eyes, and
    partial evidence beats none.
    """
    from PIL import Image, ImageDraw

    legs = plan.legs.legs
    seeds = plan.battery.seeds
    w, h = plan.tier.dims[prompt.aspect]
    tile_h = round(tile_w * h / w)
    title_h, header_h, label_h = 46, 30, 18
    width = 2 * _MARGIN + len(legs) * tile_w + (len(legs) - 1) * _GAP
    height = (
        2 * _MARGIN
        + title_h
        + header_h
        + len(seeds) * (label_h + tile_h)
        + (len(seeds) - 1) * _GAP
    )
    sheet = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    # drawn text sticks to glyphs PIL's default font has (no em-dash: tofu)
    small, med, big = _font(11), _font(15), _font(19)
    draw.text(
        (_MARGIN, _MARGIN),
        f"{prompt.scenes} · {prompt.category}",
        font=med,
        fill=(255, 255, 255),
    )
    draw.text(
        (_MARGIN, _MARGIN + 24),
        f"tier {plan.tier.name} · rows = seeds · "
        f"tile scores: {JUDGES[0]}/{JUDGES[1]}",
        font=small,
        fill=(170, 170, 170),
    )
    cells_by_key = {cell_key(c): c for c in plan.cells}
    for col, leg in enumerate(legs):
        draw.text(
            (_MARGIN + col * (tile_w + _GAP), _MARGIN + title_h),
            leg.name,
            font=big,
            fill=(255, 255, 255),
        )
    y = _MARGIN + title_h + header_h
    for seed in seeds:
        for col, leg in enumerate(legs):
            x = _MARGIN + col * (tile_w + _GAP)
            key = (leg.name, prompt.id, seed)
            frame_path = final_frame_path(plan, cells_by_key[key])
            scores = "/".join(
                f"{s:.3f}" if s is not None else "-"
                for s in (all_scores[judge].get(key) for judge in JUDGES)
            )
            if frame_path.is_file():
                label = f"s{seed}  {scores}"
                frame = Image.open(frame_path).resize((tile_w, tile_h), Image.LANCZOS)
                sheet.paste(frame, (x, y + label_h))
            else:
                label = f"s{seed}  MISSING  {scores}"
                draw.rectangle(
                    (x, y + label_h, x + tile_w - 1, y + label_h + tile_h - 1),
                    outline=(120, 60, 60),
                )
                draw.text(
                    (x + 8, y + label_h + tile_h // 2 - 8),
                    "missing frame",
                    font=med,
                    fill=(200, 90, 90),
                )
            draw.text((x, y + 2), label, font=small, fill=(200, 200, 200))
        y += label_h + tile_h + _GAP
    sheet.save(out_path)


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------


def write_report(
    plan: Plan,
    all_scores: Scores,
    trends: list[tuple[SweepField, list[TrendRow]]],
    rows: list[PromptRow],
    out_dir: Path,
) -> Path:
    import torch

    leg_names = [leg.name for leg in plan.legs.legs]
    base = plan.legs.baseline
    lines = [
        "# pytti eval matrix report",
        "",
        f"- **when/where:** {time.strftime('%Y-%m-%d %H:%M:%S')} · {platform.platform()} · "
        f"torch {torch.__version__} · mps={torch.backends.mps.is_available()}",
        f"- **tier:** {plan.tier.describe()}",
        f"- **battery:** `{plan.battery.path}` — {len(plan.battery.prompts)} prompts, "
        f"seeds {list(plan.battery.seeds)}"
        + (
            f" (subset: first {len(plan.prompts)})"
            if len(plan.prompts) != len(plan.battery.prompts)
            else ""
        ),
        f"- **judges:** {', '.join(JUDGES)} — always both, reported per judge, never "
        "averaged across families (family bias is established; disagreement is signal)",
        f"- **tie threshold:** {TIE_THRESHOLD}",
        f"- **baseline:** {base if base else '(none)'}",
        "",
        "## Legs",
        "",
        "| leg | source | overrides |",
        "|---|---|---|",
        *(
            f"| {leg.name}{' (baseline)' if leg.name == base else ''} "
            f"| {leg.source} | `{leg.overrides_text()}` |"
            for leg in plan.legs.legs
        ),
        "",
        "## Per-leg adherence (held-out cosine, mean over 6 views)",
        "",
        "| judge | leg | mean | median | n |",
        "|---|---|---|---|---|",
    ]
    for judge in JUDGES:
        for leg in leg_names:
            stats = leg_stats(all_scores[judge], leg)
            lines.append(
                f"| {judge} | {leg} | {stats.mean:.4f} | {stats.median:.4f} | {stats.n} |"
            )
    lines += ["", f"## Win-rate matrices (row beats column; tie < {TIE_THRESHOLD})", ""]
    for judge in JUDGES:
        lines.append(f"### {judge}")
        lines.append("")
        lines.append("| vs | " + " | ".join(leg_names) + " |")
        lines.append("|---|" + "---|" * len(leg_names))
        for leg_a in leg_names:
            row = [f"| {leg_a} "]
            for leg_b in leg_names:
                if leg_a == leg_b:
                    row.append("| — ")
                else:
                    row.append(f"| {compare_legs(all_scores[judge], leg_a, leg_b).text()} ")
            lines.append("".join(row) + "|")
        lines.append("")
    for sweep, trend in trends:
        lines.append(f"## Trend: {sweep.field} (baseline: {base if base else 'none'})")
        lines.append("")
        header = f"| {sweep.field} | legs |"
        divider = "|---|---|"
        for judge in JUDGES:
            header += f" {judge} mean |"
            divider += "---|"
            if base is not None:
                header += f" {judge} vs baseline |"
                divider += "---|"
        lines += [header, divider]
        for row in trend:
            text = f"| {row.value} | {', '.join(row.legs)} |"
            for judge in JUDGES:
                text += f" {row.means[judge]:.4f} |"
                if row.vs_baseline is not None:
                    text += f" {row.vs_baseline[judge].text()} |"
            lines.append(text)
        lines.append("")
    lines += [
        "## Per-prompt (sorted by judge disagreement on leg ranking)",
        "",
        "disagreement = normalized rank-footrule between the two judges' leg",
        "orderings for that prompt (0 = same order, 1 = reversed).",
        "",
        "| prompt | category | disagreement | "
        + " | ".join(f"{judge} best leg" for judge in JUDGES)
        + " |",
        "|---|---|---|" + "---|" * len(JUDGES),
    ]
    for row in rows:
        best = [
            max(sorted(row.means[judge]), key=lambda leg: row.means[judge][leg])
            for judge in JUDGES
        ]
        lines.append(
            f"| {row.prompt.id} | {row.prompt.category} | {row.disagreement:.2f} | "
            + " | ".join(
                f"{leg} ({row.means[judge][leg]:.4f})"
                for judge, leg in zip(JUDGES, best, strict=True)
            )
            + " |"
        )
    disagreeing = [row.prompt.id for row in rows if row.disagreement > 0][:3]
    lines += [
        "",
        (
            f"**Largest-disagreement prompts:** {', '.join(disagreeing)} — "
            "eyeball these grids first; the judges are voting for different looks."
            if disagreeing
            else "**Judges fully agree on leg ordering for every prompt.**"
        ),
        "",
        "## Artifacts",
        "",
        "Compare sheets (one per prompt: columns = legs, rows = seeds, both",
        "judges on every tile) are the primary human-comparison artifact —",
        "eyeball these before the tables:",
        "",
        *(
            f"- compare sheet `{p.id}`: {out_dir / 'compare' / f'{p.id}.png'}"
            for p in plan.prompts
        ),
        "",
        *(f"- contact grid `{leg}`: {out_dir / f'grid_{leg}.png'}" for leg in leg_names),
        f"- metrics: {out_dir / 'metrics.json'}",
        f"- cell workspaces/logs: {plan.root}/<leg>/<cell>/",
        "",
    ]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_metrics_json(
    plan: Plan,
    all_scores: Scores,
    trends: list[tuple[SweepField, list[TrendRow]]],
    rows: list[PromptRow],
    out_dir: Path,
) -> None:
    def record(win: WinRecord) -> dict:
        return {
            "wins": win.wins,
            "losses": win.losses,
            "ties": win.ties,
            "win_rate": win.win_rate,
        }

    leg_names = [leg.name for leg in plan.legs.legs]
    payload = {
        "tier": {
            "name": plan.tier.name,
            "dims": {aspect: list(dims) for aspect, dims in plan.tier.dims.items()},
            "steps": plan.tier.steps,
            "save_every": plan.tier.save_every,
        },
        "battery": str(plan.battery.path),
        "seeds": list(plan.battery.seeds),
        "prompts_used": [p.id for p in plan.prompts],
        "judges": list(JUDGES),
        "tie_threshold": TIE_THRESHOLD,
        "baseline": plan.legs.baseline,
        "legs": [
            {
                "name": leg.name,
                "source": leg.source,
                "overrides": leg.overrides_dict(),
                "sweep": dict(leg.sweep),
            }
            for leg in plan.legs.legs
        ],
        "cells": [
            {
                "leg": cell.leg.name,
                "prompt": cell.prompt.id,
                "seed": cell.seed,
                "final": str(final_frame_path(plan, cell)),
                "scores": {judge: all_scores[judge][cell_key(cell)] for judge in JUDGES},
            }
            for cell in plan.cells
        ],
        "leg_stats": {
            judge: {
                leg: {
                    "mean": leg_stats(all_scores[judge], leg).mean,
                    "median": leg_stats(all_scores[judge], leg).median,
                    "n": leg_stats(all_scores[judge], leg).n,
                }
                for leg in leg_names
            }
            for judge in JUDGES
        },
        "win_matrix": {
            judge: [
                {
                    "a": leg_a,
                    "b": leg_b,
                    **record(compare_legs(all_scores[judge], leg_a, leg_b)),
                }
                for leg_a in leg_names
                for leg_b in leg_names
                if leg_a != leg_b
            ]
            for judge in JUDGES
        },
        "trends": [
            {
                "field": sweep.field,
                "baseline": plan.legs.baseline,
                "rows": [
                    {
                        "value": row.value,
                        "legs": list(row.legs),
                        "means": row.means,
                        "vs_baseline": (
                            {j: record(w) for j, w in row.vs_baseline.items()}
                            if row.vs_baseline is not None
                            else None
                        ),
                    }
                    for row in trend
                ],
            }
            for sweep, trend in trends
        ],
        "prompt_disagreement": [
            {
                "prompt": row.prompt.id,
                "category": row.prompt.category,
                "disagreement": row.disagreement,
                "means": row.means,
            }
            for row in rows
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--legs",
        action="append",
        default=[],
        metavar="NAME[:K=V,K=V...]",
        help="a named leg of overrides on the screening base (repeatable)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        metavar="NAME[:K=V,...]",
        help="baseline leg that win-rate-vs-baseline columns anchor to",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="FIELD=V1,V2,...",
        help="sweep a field (repeatable; cross-products into auto-named legs)",
    )
    parser.add_argument(
        "--tier",
        choices=sorted(TIERS),
        default="screening",
        help="render budget: screening ranks legs cheaply at small dims; "
        "full certifies at production dims (default: screening)",
    )
    parser.add_argument("--battery", type=Path, default=DEFAULT_BATTERY)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="report directory (default /tmp/pytti-eval/report-<tier>; wiped)",
    )
    parser.add_argument(
        "--subset", type=int, default=None, metavar="N", help="first N battery prompts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the full cell matrix + commands, render nothing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explicit = [parse_leg_spec(spec) for spec in args.legs]
    baseline = (
        parse_leg_spec(args.baseline, source="baseline") if args.baseline else None
    )
    sweeps = [parse_sweep_spec(spec) for spec in args.sweep]
    leg_set = assemble_legs(explicit, baseline, sweeps)
    battery = parse_battery(args.battery)
    tier = TIERS[args.tier]
    plan = build_plan(battery, leg_set, args.subset, tier)

    if args.dry_run:
        print(format_plan(plan, sys.executable))
        return

    ran = skipped = 0
    for cell in plan.cells:
        if cell_is_done(plan, cell):
            print(f"[{cell.leg.name}/{cell.name}] final frame exists — skipping")
            skipped += 1
        else:
            run_cell(plan, cell)
            ran += 1
    print(f"renders: {ran} run, {skipped} skipped (resumed)")

    all_scores = judge_all(plan)
    sweep_legs = [leg for leg in leg_set.legs if leg.source == "sweep"]
    trends = [
        (sweep, trend_table(all_scores, sweep, sweep_legs, leg_set.baseline))
        for sweep in sweeps
    ]
    rows = prompt_rows(all_scores, plan)

    out_dir: Path = args.out if args.out is not None else EVAL_ROOT / f"report-{tier.name}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    compare_dir = out_dir / "compare"
    compare_dir.mkdir()
    for prompt in plan.prompts:
        compare_sheet(plan, prompt, all_scores, compare_dir / f"{prompt.id}.png")
    for leg in leg_set.legs:
        contact_grid(plan, leg, all_scores, out_dir / f"grid_{leg.name}.png")
    write_metrics_json(plan, all_scores, trends, rows, out_dir)
    report_path = write_report(plan, all_scores, trends, rows, out_dir)

    for judge in JUDGES:
        summary = "   ".join(
            f"{leg}={leg_stats(all_scores[judge], leg).mean:.4f}"
            for leg in (leg.name for leg in leg_set.legs)
        )
        print(f"{judge}: {summary}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
