"""
Unit tests for scripts/eval_matrix.py: battery parse/validation, leg + sweep
spec parsing (incl. cross-products and name generation), tier plumbing,
aggregation math on synthetic scores, compare sheets, and the dry-run plan
snapshots (one per tier). Pure CPU, no downloads, no renders.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATHS = {
    "screening": REPO_ROOT / "tests" / "fixtures" / "eval" / "dryrun_plan.snapshot.txt",
    "full": REPO_ROOT / "tests" / "fixtures" / "eval" / "dryrun_plan_full.snapshot.txt",
}

ROBOT_PROMPT = (
    "a full shot of a metallic, silver colored robotic knight standing in a desert"
)


def _load_eval_matrix():
    spec = importlib.util.spec_from_file_location(
        "eval_matrix", REPO_ROOT / "scripts" / "eval_matrix.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # py3.10 dataclasses resolve cls.__module__ through sys.modules at class
    # creation; exec_module without registration crashes
    sys.modules["eval_matrix"] = module
    spec.loader.exec_module(module)
    return module


em = _load_eval_matrix()


# ----------------------------------------------------------------------
# battery parse + validation
# ----------------------------------------------------------------------


def test_battery_parses_and_covers_the_use_space():
    battery = em.parse_battery(em.DEFAULT_BATTERY)
    assert battery.seeds == (2, 3)
    assert len(battery.prompts) == 10
    ids = [p.id for p in battery.prompts]
    assert len(set(ids)) == 10
    categories = [p.category for p in battery.prompts]
    assert len(set(categories)) == 10  # each prompt covers a distinct slot
    # aspect alternates 1:1 <-> 3:4 (a full cross doubles cost for no signal)
    assert [p.aspect for p in battery.prompts] == ["1:1", "3:4"] * 5
    # the user-authentic prompt is verbatim from a real pytti-able session
    assert any(p.scenes == ROBOT_PROMPT for p in battery.prompts)
    assert all(p.rationale.strip() for p in battery.prompts)


def test_battery_judge_text_flattens_piped_prompts():
    battery = em.parse_battery(em.DEFAULT_BATTERY)
    market = next(p for p in battery.prompts if p.id == "neon-market")
    assert "|" not in market.judge_text
    assert market.judge_text == (
        "cyberpunk street market at night #artstation, "
        "neon reflections, trending on artstation"
    )


VALID_BATTERY = {
    "seeds": [2, 3],
    "prompts": [
        {
            "id": "temple",
            "category": "composed-scene",
            "aspect": "1:1",
            "scenes": "an ancient temple, dramatic lighting",
            "rationale": "why",
        }
    ],
}


def _write_battery(tmp_path: Path, data) -> Path:
    path = tmp_path / "battery.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _mutated(**top_level):
    data = {
        "seeds": list(VALID_BATTERY["seeds"]),
        "prompts": [dict(VALID_BATTERY["prompts"][0])],
    }
    data.update(top_level)
    return data


def test_battery_valid_minimal_roundtrips(tmp_path):
    battery = em.parse_battery(_write_battery(tmp_path, _mutated()))
    assert battery.prompts[0].judge_text == "an ancient temple, dramatic lighting"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"bonus": 1}, "unknown top-level keys"),
        ({"seeds": []}, "non-empty list"),
        ({"seeds": [2, "x"]}, "not an int"),
        ({"seeds": [2, 2]}, "duplicate seeds"),
        ({"prompts": []}, "non-empty list"),
    ],
)
def test_battery_top_level_validation(tmp_path, mutation, match):
    with pytest.raises(SystemExit, match=match):
        em.parse_battery(_write_battery(tmp_path, _mutated(**mutation)))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("aspect", "16:9", "aspect"),
        ("id", "bad name", "must match"),
        ("rationale", "  ", "non-empty string"),
        ("scenes", "scene one || scene two", "scenes"),
        ("extra", "x", "unknown keys"),
    ],
)
def test_battery_prompt_validation(tmp_path, field, value, match):
    data = _mutated()
    data["prompts"][0][field] = value
    with pytest.raises(SystemExit, match=match):
        em.parse_battery(_write_battery(tmp_path, data))


def test_battery_duplicate_prompt_ids_rejected(tmp_path):
    data = _mutated()
    data["prompts"].append(dict(data["prompts"][0]))
    with pytest.raises(SystemExit, match="duplicate id"):
        em.parse_battery(_write_battery(tmp_path, data))


# ----------------------------------------------------------------------
# leg + sweep spec parsing
# ----------------------------------------------------------------------


def test_leg_spec_without_overrides():
    leg = em.parse_leg_spec("default")
    assert leg.name == "default"
    assert leg.overrides == ()
    assert leg.source == "explicit"


def test_leg_spec_with_overrides_preserves_order():
    leg = em.parse_leg_spec("fast:cutouts=16,cut_pow=1.5")
    assert leg.name == "fast"
    assert leg.overrides == (("cutouts", "16"), ("cut_pow", "1.5"))


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("we$t", "must match"),
        ("x:", "empty overrides"),
        ("x:nonsense_key=1", "not a ConfigSchema field"),
        ("x:seed=5", "cell identity"),
        ("x:scenes=hijack", "cell identity"),
        ("x:cutouts=8,cutouts=9", "duplicate override key"),
        ("x:cutouts", "must be key=value"),
        ("x:ViTL14=true", "held out"),
        ("x:SigLIP2SO400M=1", "held out"),
    ],
)
def test_leg_spec_rejections(spec, match):
    with pytest.raises(SystemExit, match=match):
        em.parse_leg_spec(spec)


def test_leg_spec_judge_disabled_is_allowed():
    # explicitly turning a judge OFF keeps it held out — fine
    leg = em.parse_leg_spec("x:ViTL14=false")
    assert leg.overrides == (("ViTL14", "false"),)


def test_sweep_spec_parses():
    sweep = em.parse_sweep_spec("cutouts=8,16,40")
    assert sweep.field == "cutouts"
    assert sweep.values == ("8", "16", "40")


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("cutouts", "must be field="),
        ("cutouts=8", "at least 2 values"),
        ("cutouts=8,8", "duplicate values"),
        ("nonsense=1,2", "not a ConfigSchema field"),
        ("seed=1,2", "cell identity"),
        ("ViTL14=false,true", "held out"),
    ],
)
def test_sweep_spec_rejections(spec, match):
    with pytest.raises(SystemExit, match=match):
        em.parse_sweep_spec(spec)


def test_sweep_expansion_single_field():
    legs = em.expand_sweeps([em.parse_sweep_spec("cutouts=8,16")])
    assert [leg.name for leg in legs] == ["cutouts-8", "cutouts-16"]
    assert legs[0].overrides == (("cutouts", "8"),)
    assert legs[0].sweep == (("cutouts", "8"),)
    assert all(leg.source == "sweep" for leg in legs)


def test_sweep_expansion_cross_product_names():
    legs = em.expand_sweeps(
        [em.parse_sweep_spec("cutouts=8,16"), em.parse_sweep_spec("cut_pow=1,2")]
    )
    assert [leg.name for leg in legs] == [
        "cutouts-8_cut_pow-1",
        "cutouts-8_cut_pow-2",
        "cutouts-16_cut_pow-1",
        "cutouts-16_cut_pow-2",
    ]
    assert legs[3].overrides == (("cutouts", "16"), ("cut_pow", "2"))


def test_sweep_duplicate_field_rejected():
    with pytest.raises(SystemExit, match="duplicate swept fields"):
        em.expand_sweeps(
            [em.parse_sweep_spec("cutouts=8,16"), em.parse_sweep_spec("cutouts=1,2")]
        )


def test_assemble_implicit_baseline_is_the_sole_explicit_leg():
    leg_set = em.assemble_legs(
        [em.parse_leg_spec("default")], None, [em.parse_sweep_spec("cutouts=8,16")]
    )
    assert leg_set.baseline == "default"
    assert [leg.name for leg in leg_set.legs] == ["default", "cutouts-8", "cutouts-16"]


def test_assemble_no_baseline_with_two_explicit_legs():
    leg_set = em.assemble_legs(
        [em.parse_leg_spec("a"), em.parse_leg_spec("b")],
        None,
        [em.parse_sweep_spec("cutouts=8,16")],
    )
    assert leg_set.baseline is None


def test_assemble_explicit_baseline_comes_first():
    leg_set = em.assemble_legs(
        [em.parse_leg_spec("other")],
        em.parse_leg_spec("base", source="baseline"),
        [],
    )
    assert leg_set.baseline == "base"
    assert [leg.name for leg in leg_set.legs] == ["base", "other"]


def test_assemble_duplicate_names_rejected():
    with pytest.raises(SystemExit, match="duplicate leg names"):
        em.assemble_legs([em.parse_leg_spec("a"), em.parse_leg_spec("a")], None, [])


def test_assemble_empty_rejected():
    with pytest.raises(SystemExit, match="no legs"):
        em.assemble_legs([], None, [])


# ----------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------


def _plan(legs=("default",), sweeps=(), subset=2, tier="screening", root=None):
    leg_set = em.assemble_legs(
        [em.parse_leg_spec(spec) for spec in legs],
        None,
        [em.parse_sweep_spec(spec) for spec in sweeps],
    )
    battery = em.parse_battery(em.DEFAULT_BATTERY)
    kwargs = {} if root is None else {"root": root}
    return em.build_plan(battery, leg_set, subset, em.TIERS[tier], **kwargs)


def test_plan_cell_count_and_layout():
    plan = _plan(legs=("default",), sweeps=("cutouts=8,16,40",), subset=3)
    assert len(plan.legs.legs) == 4
    assert len(plan.cells) == 4 * 3 * 2  # legs x prompts x seeds
    cell = plan.cells[0]
    assert cell.name == "temple-lit_s2"
    assert (cell.width, cell.height) == (256, 256)
    assert em.final_frame_path(plan, cell) == Path(
        "/tmp/pytti-eval/screening/default/temple-lit_s2/run/images_out/"
        "temple-lit_s2/temple-lit_s2_0006.png"
    )
    portrait = next(c for c in plan.cells if c.prompt.aspect == "3:4")
    assert (portrait.width, portrait.height) == (240, 320)


# ----------------------------------------------------------------------
# tiers
# ----------------------------------------------------------------------


def test_tiers_cover_exactly_the_battery_aspects():
    assert set(em.TIERS) == {"screening", "full"}
    for tier in em.TIERS.values():
        assert tuple(tier.dims) == em.ASPECTS


def test_full_tier_dims_steps_and_root():
    plan = _plan(subset=3, tier="full")
    assert plan.tier.name == "full"
    cell = plan.cells[0]
    assert (cell.width, cell.height) == (512, 512)
    assert cell.steps == 200 and cell.save_every == 25
    assert cell.final_index == 8
    # tier-scoped workspace root: screening and full campaigns coexist
    assert em.final_frame_path(plan, cell) == Path(
        "/tmp/pytti-eval/full/default/temple-lit_s2/run/images_out/"
        "temple-lit_s2/temple-lit_s2_0008.png"
    )
    portrait = next(c for c in plan.cells if c.prompt.aspect == "3:4")
    assert (portrait.width, portrait.height) == (448, 576)


def test_full_tier_conf_text_carries_tier_budget():
    plan = _plan(subset=1, tier="full")
    conf = yaml.safe_load(em.cell_conf_text(plan.cells[0], plan.tier))
    assert (conf["width"], conf["height"]) == (512, 512)
    assert conf["steps_per_scene"] == 200 and conf["save_every"] == 25


def test_tier_describe_names_dims_and_budget():
    assert em.TIERS["screening"].describe() == (
        "screening — 1:1 -> 256x256, 3:4 -> 240x320, "
        "150 steps, save_every 25, backups 0, schema defaults otherwise"
    )
    assert em.TIERS["full"].describe() == (
        "full — 1:1 -> 512x512, 3:4 -> 448x576, "
        "200 steps, save_every 25, backups 0, schema defaults otherwise"
    )


def test_leg_step_override_still_beats_tier_default():
    plan = _plan(legs=("long:steps_per_scene=400",), subset=1, tier="full")
    assert plan.cells[0].steps == 400
    assert plan.cells[0].final_index == 16


def test_plan_subset_out_of_range_rejected():
    with pytest.raises(SystemExit, match="out of range"):
        _plan(subset=11)


def test_plan_leg_steps_override_moves_final_frame():
    plan = _plan(legs=("long:steps_per_scene=300",), subset=1)
    assert plan.cells[0].final_index == 12


@pytest.mark.parametrize(
    ("leg", "match"),
    [
        ("bad:steps_per_scene=140", "not a multiple"),
        ("bad:save_every=0", "positive int"),
        ("bad:steps_per_scene=nope", "positive int"),
    ],
)
def test_plan_step_save_validation(leg, match):
    with pytest.raises(SystemExit, match=match):
        _plan(legs=(leg,), subset=1)


def test_cell_conf_text_types_overrides():
    plan = _plan(legs=("full16:cutout_sampler=full,cutouts=16",), subset=1)
    cell = plan.cells[0]
    conf = yaml.safe_load(em.cell_conf_text(cell, plan.tier))
    assert conf["scenes"] == cell.prompt.scenes
    assert conf["seed"] == 2
    assert (conf["width"], conf["height"]) == (256, 256)
    assert conf["steps_per_scene"] == 150 and conf["save_every"] == 25
    assert conf["cutouts"] == 16  # yaml-typed, not the string "16"
    assert conf["cutout_sampler"] == "full"
    assert conf["file_namespace"] == cell.name
    assert em.cell_conf_text(cell, plan.tier).startswith("# @package _global_\n")


# ----------------------------------------------------------------------
# aggregation math (synthetic scores)
# ----------------------------------------------------------------------


def _scores(**by_leg):
    """{(leg, prompt, seed): score} from {leg: {(prompt, seed): score}}."""
    return {
        (leg, prompt, seed): value
        for leg, cells in by_leg.items()
        for (prompt, seed), value in cells.items()
    }


CELLS = [("p1", 2), ("p1", 3), ("p2", 2), ("p2", 3)]


def test_leg_stats_mean_median():
    scores = _scores(a={("p1", 2): 0.1, ("p1", 3): 0.2, ("p2", 2): 0.6, ("p2", 3): 0.3})
    stats = em.leg_stats(scores, "a")
    assert stats.mean == pytest.approx(0.3)
    assert stats.median == pytest.approx(0.25)
    assert stats.n == 4


def test_leg_stats_missing_leg_dies():
    with pytest.raises(SystemExit, match="no scores for leg"):
        em.leg_stats(_scores(a={("p1", 2): 0.1}), "b")


def test_compare_legs_wins_losses_ties():
    scores = _scores(
        a={("p1", 2): 0.30, ("p1", 3): 0.10, ("p2", 2): 0.5000, ("p2", 3): 0.4019},
        b={("p1", 2): 0.20, ("p1", 3): 0.15, ("p2", 2): 0.5020, ("p2", 3): 0.4000},
    )
    record = em.compare_legs(scores, "a", "b")
    # p1s2: +0.10 win; p1s3: -0.05 loss; p2s2: -0.0020 loss (== threshold,
    # NOT a tie); p2s3: +0.0019 tie (< threshold)
    assert (record.wins, record.losses, record.ties) == (1, 2, 1)
    assert record.win_rate == pytest.approx(0.25)
    inverse = em.compare_legs(scores, "b", "a")
    assert (inverse.wins, inverse.losses, inverse.ties) == (2, 1, 1)


def test_compare_legs_mismatched_cells_die():
    scores = _scores(a={("p1", 2): 0.1}, b={("p1", 3): 0.1})
    with pytest.raises(SystemExit, match="cell sets differ"):
        em.compare_legs(scores, "a", "b")


def test_rank_disagreement_identical_and_reversed():
    assert em.rank_disagreement({"a": 0.3, "b": 0.2}, {"a": 0.9, "b": 0.1}) == 0.0
    assert em.rank_disagreement({"a": 0.3, "b": 0.2}, {"a": 0.1, "b": 0.9}) == 1.0
    # 3 legs fully reversed: footrule |1-3|+|2-2|+|3-1| = 4, max = 4
    assert (
        em.rank_disagreement(
            {"a": 0.3, "b": 0.2, "c": 0.1}, {"a": 0.1, "b": 0.2, "c": 0.3}
        )
        == 1.0
    )


def test_rank_disagreement_ties_use_average_ranks():
    # judge 2 scores all legs identically -> every rank is 2.0;
    # footrule = |1-2|+|2-2|+|3-2| = 2, normalized by 4 -> 0.5
    value = em.rank_disagreement(
        {"a": 0.3, "b": 0.2, "c": 0.1}, {"a": 0.5, "b": 0.5, "c": 0.5}
    )
    assert value == pytest.approx(0.5)


def test_rank_disagreement_single_leg_is_zero():
    assert em.rank_disagreement({"a": 0.3}, {"a": 0.1}) == 0.0


def test_trend_table_means_and_win_rates():
    sweep = em.parse_sweep_spec("cutouts=8,16")
    sweep_legs = em.expand_sweeps([sweep])
    base = {("p1", 2): 0.30, ("p1", 3): 0.20}
    scores_j1 = _scores(
        default=base,
        **{
            "cutouts-8": {("p1", 2): 0.20, ("p1", 3): 0.10},  # loses both
            "cutouts-16": {("p1", 2): 0.40, ("p1", 3): 0.2001},  # win + tie
        },
    )
    # second judge disagrees: cutouts-8 wins everywhere
    scores_j2 = _scores(
        default=base,
        **{
            "cutouts-8": {("p1", 2): 0.90, ("p1", 3): 0.90},
            "cutouts-16": {("p1", 2): 0.10, ("p1", 3): 0.10},
        },
    )
    all_scores = {"ViTL14": scores_j1, "SigLIP2SO400M": scores_j2}
    rows = em.trend_table(all_scores, sweep, sweep_legs, "default")
    assert [row.value for row in rows] == ["8", "16"]
    assert rows[0].legs == ("cutouts-8",)
    assert rows[0].means["ViTL14"] == pytest.approx(0.15)
    assert rows[0].means["SigLIP2SO400M"] == pytest.approx(0.90)
    r8, r16 = rows[0].vs_baseline["ViTL14"], rows[1].vs_baseline["ViTL14"]
    assert (r8.wins, r8.losses, r8.ties) == (0, 2, 0)
    assert (r16.wins, r16.losses, r16.ties) == (1, 0, 1)
    assert rows[0].vs_baseline["SigLIP2SO400M"].wins == 2  # per-judge disagreement


def test_trend_table_without_baseline_has_no_win_rates():
    sweep = em.parse_sweep_spec("cutouts=8,16")
    sweep_legs = em.expand_sweeps([sweep])
    scores = _scores(
        **{
            "cutouts-8": {("p1", 2): 0.1},
            "cutouts-16": {("p1", 2): 0.2},
        }
    )
    rows = em.trend_table({"ViTL14": scores, "SigLIP2SO400M": scores}, sweep, sweep_legs, None)
    assert all(row.vs_baseline is None for row in rows)


def test_trend_table_pools_cross_product_combos():
    sweeps = [em.parse_sweep_spec("cutouts=8,16"), em.parse_sweep_spec("cut_pow=1,2")]
    sweep_legs = em.expand_sweeps(sweeps)
    cells = {("p1", 2): 0.0}
    scores = _scores(
        **{
            "cutouts-8_cut_pow-1": {("p1", 2): 0.1},
            "cutouts-8_cut_pow-2": {("p1", 2): 0.3},
            "cutouts-16_cut_pow-1": {("p1", 2): 0.5},
            "cutouts-16_cut_pow-2": {("p1", 2): 0.7},
        },
        default=cells,
    )
    all_scores = {"ViTL14": scores, "SigLIP2SO400M": scores}
    rows = em.trend_table(all_scores, sweeps[0], sweep_legs, None)
    assert rows[0].legs == ("cutouts-8_cut_pow-1", "cutouts-8_cut_pow-2")
    assert rows[0].means["ViTL14"] == pytest.approx(0.2)
    assert rows[1].means["ViTL14"] == pytest.approx(0.6)


# ----------------------------------------------------------------------
# compare sheets (synthetic frames + scores; no renders)
# ----------------------------------------------------------------------


def _fake_frame(path: Path, size, color) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _sheet_scores(plan, skip_key=None):
    """Synthetic per-judge scores for every cell (optionally minus one)."""
    scores = {judge: {} for judge in em.JUDGES}
    for i, cell in enumerate(plan.cells):
        key = em.cell_key(cell)
        if key == skip_key:
            continue
        for j, judge in enumerate(em.JUDGES):
            scores[judge][key] = 0.1 * (i + 1) + 0.01 * j
    return scores


def test_compare_sheet_writes_one_png_per_prompt_layout(tmp_path):
    from PIL import Image

    plan = _plan(legs=("a", "b:cutouts=16"), subset=1, root=tmp_path)
    for cell in plan.cells:
        _fake_frame(
            em.final_frame_path(plan, cell), (cell.width, cell.height), (10, 60, 110)
        )
    prompt = plan.prompts[0]
    out = tmp_path / "compare" / f"{prompt.id}.png"
    out.parent.mkdir()
    em.compare_sheet(plan, prompt, _sheet_scores(plan), out, tile_w=64)

    with Image.open(out) as sheet:
        # 2 leg columns x 2 seed rows of 64px-wide tiles at the prompt aspect
        tile_h = round(64 * 256 / 256)
        assert sheet.width == 2 * 10 + 2 * 64 + 4
        assert sheet.height == 2 * 10 + 46 + 30 + 2 * (18 + tile_h) + 4


def test_compare_sheet_marks_missing_frame_and_score(tmp_path):
    plan = _plan(legs=("a", "b:cutouts=16"), subset=1, root=tmp_path)
    missing = em.cell_key(plan.cells[-1])  # leg b, seed 3: no frame, no score
    for cell in plan.cells:
        if em.cell_key(cell) != missing:
            _fake_frame(
                em.final_frame_path(plan, cell), (cell.width, cell.height), (90, 20, 20)
            )
    prompt = plan.prompts[0]
    out = tmp_path / f"{prompt.id}.png"
    # must not die: the sheet marks the hole instead
    em.compare_sheet(plan, prompt, _sheet_scores(plan, skip_key=missing), out)
    assert out.is_file()


def test_compare_sheet_uses_tier_aspect_for_tile_height(tmp_path):
    from PIL import Image

    plan = _plan(legs=("a",), subset=2, root=tmp_path, tier="full")
    portrait = plan.prompts[1]
    assert portrait.aspect == "3:4"
    for cell in plan.cells:
        _fake_frame(em.final_frame_path(plan, cell), (16, 16), (0, 0, 0))
    out = tmp_path / f"{portrait.id}.png"
    em.compare_sheet(plan, portrait, _sheet_scores(plan), out, tile_w=60)
    with Image.open(out) as sheet:
        tile_h = round(60 * 576 / 448)
        assert sheet.height == 2 * 10 + 46 + 30 + 2 * (18 + tile_h) + 4


# ----------------------------------------------------------------------
# dry-run plan snapshots (2 legs + 1 sweep, one snapshot per tier)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("tier", sorted(em.TIERS))
def test_dry_run_plan_snapshot(tier):
    leg_set = em.assemble_legs(
        [em.parse_leg_spec("default"), em.parse_leg_spec("full16:cutout_sampler=full,cutouts=16")],
        None,
        [em.parse_sweep_spec("cut_pow=1,2")],
    )
    battery = em.parse_battery(em.DEFAULT_BATTERY)
    plan = em.build_plan(battery, leg_set, 2, em.TIERS[tier])
    expected = (
        SNAPSHOT_PATHS[tier]
        .read_text(encoding="utf-8")
        .replace("{battery}", str(em.DEFAULT_BATTERY))
        .rstrip("\n")
    )
    assert em.format_plan(plan, "python") == expected
