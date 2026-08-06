"""
Unit tests for eval_matrix chained legs — the init_from pseudo-override that
renders one leg FROM another leg's final frames (underpainting experiments,
e.g. LlamaGen structure under a pixel-space finish). Covers: spec parsing and
init_from extraction, unknown-source + cycle fail-loud at assemble time,
topological render order (depth-2 chain + diamond), init_image path injection
into generated cell confs, the missing-source-final guard, dependent-cell
resumability, and chain self-documentation in report/metrics. Pure CPU, no
renders.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def _assemble(*specs: str):
    return em.assemble_legs([em.parse_leg_spec(s) for s in specs], None, [])


def _plan(*specs: str, subset: int = 1, root: Path, tier: str = "screening"):
    battery = em.parse_battery(em.DEFAULT_BATTERY)
    return em.build_plan(battery, _assemble(*specs), subset, em.TIERS[tier], root=root)


def _cell(plan, leg: str, seed: int):
    return next(c for c in plan.cells if c.leg.name == leg and c.seed == seed)


def _fake_frame(path: Path, size=(8, 8), color=(10, 60, 110)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


# ----------------------------------------------------------------------
# leg-spec parsing: init_from extraction
# ----------------------------------------------------------------------


def test_init_from_extracted_and_never_an_override():
    leg = em.parse_leg_spec("finish:init_from=base,direct_init_weight=1")
    assert leg.init_from == "base"
    assert leg.overrides == (("direct_init_weight", "1"),)


def test_init_from_position_independent_and_alone():
    mid = em.parse_leg_spec("finish:direct_init_weight=1,init_from=base,cutouts=16")
    assert mid.init_from == "base"
    assert mid.overrides == (("direct_init_weight", "1"), ("cutouts", "16"))
    alone = em.parse_leg_spec("finish:init_from=base")
    assert alone.init_from == "base"
    assert alone.overrides == ()


def test_overrides_text_shows_the_chain():
    leg = em.parse_leg_spec("finish:init_from=lg-under,direct_init_weight=1")
    assert leg.overrides_text() == "init_from: lg-under, direct_init_weight=1"
    assert em.parse_leg_spec("finish:init_from=base").overrides_text() == "init_from: base"
    # non-chained legs are untouched (byte-compat)
    assert em.parse_leg_spec("plain").overrides_text() == "(schema defaults)"


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("x:init_from=bad name", "must be a leg name"),
        ("x:init_from=a,init_from=b", "duplicate override key"),
        ("x:init_from=a,init_image=/tmp/foo.png", "conflict"),
        ("x:init_from=", "must be key=value"),
    ],
)
def test_init_from_spec_rejections(spec, match):
    with pytest.raises(SystemExit, match=match):
        em.parse_leg_spec(spec)


def test_sweep_cannot_carry_init_from():
    with pytest.raises(SystemExit, match="not a ConfigSchema field"):
        em.parse_sweep_spec("init_from=a,b")


# ----------------------------------------------------------------------
# assemble-time fail-loud: unknown source + cycles
# ----------------------------------------------------------------------


def test_unknown_source_dies_at_assemble_time():
    with pytest.raises(SystemExit, match="init_from source 'ghost' is not a leg"):
        _assemble("base", "finish:init_from=ghost")


def test_two_leg_cycle_dies_at_assemble_time():
    with pytest.raises(SystemExit, match="init_from cycle: a -> b -> a"):
        _assemble("a:init_from=b", "b:init_from=a")


def test_self_cycle_dies_at_assemble_time():
    with pytest.raises(SystemExit, match="init_from cycle: a -> a"):
        _assemble("a:init_from=a")


# ----------------------------------------------------------------------
# topological render order
# ----------------------------------------------------------------------


def test_render_order_depth2_chain_given_reversed():
    leg_set = _assemble("finish:init_from=mid", "mid:init_from=base", "base")
    # CLI order is preserved for reports/sheets ...
    assert [leg.name for leg in leg_set.legs] == ["finish", "mid", "base"]
    # ... while rendering follows the dependency chain
    assert [leg.name for leg in em.render_order(leg_set.legs)] == ["base", "mid", "finish"]


def test_render_order_diamond():
    leg_set = _assemble("d:init_from=l", "r:init_from=base", "l:init_from=base", "base")
    order = [leg.name for leg in em.render_order(leg_set.legs)]
    assert order == ["base", "l", "d", "r"]
    assert order.index("base") < order.index("l") < order.index("d")
    assert order.index("base") < order.index("r")


def test_render_order_is_identity_for_unchained_legs():
    leg_set = _assemble("a", "b:cutouts=16", "c")
    assert em.render_order(leg_set.legs) == leg_set.legs


def test_baseline_leg_works_as_source_and_as_dependent(tmp_path):
    battery = em.parse_battery(em.DEFAULT_BATTERY)
    n = len(battery.seeds)

    def plan_with_baseline(legs: list[str], baseline: str):
        leg_set = em.assemble_legs(
            [em.parse_leg_spec(s) for s in legs],
            em.parse_leg_spec(baseline, source="baseline"),
            [],
        )
        return em.build_plan(battery, leg_set, 1, em.TIERS["screening"], root=tmp_path)

    # baseline as SOURCE: dependents chain onto it, win-rates still anchor to it
    plan = plan_with_baseline(["finish:init_from=base"], "base")
    assert plan.legs.baseline == "base"
    assert [c.leg.name for c in plan.cells] == ["base"] * n + ["finish"] * n
    for src, dep in zip(plan.cells[:n], plan.cells[n:], strict=True):
        assert dep.init_image == em.final_frame_path(plan, src)

    # baseline as DEPENDENT: renders after its source, reports keep CLI order
    plan = plan_with_baseline(["base"], "finish:init_from=base")
    assert plan.legs.baseline == "finish"
    assert [leg.name for leg in plan.legs.legs] == ["finish", "base"]
    assert [c.leg.name for c in plan.cells] == ["base"] * n + ["finish"] * n


# ----------------------------------------------------------------------
# plan: init_image injection + cell render order
# ----------------------------------------------------------------------


def test_plan_orders_cells_by_dependency_and_injects_init(tmp_path):
    plan = _plan("finish:init_from=base,direct_init_weight=1", "base", root=tmp_path)
    # report order stays as given; cells render source-first
    assert [leg.name for leg in plan.legs.legs] == ["finish", "base"]
    assert [c.leg.name for c in plan.cells] == ["base", "base", "finish", "finish"]
    for seed in (2, 3):
        dep, src = _cell(plan, "finish", seed), _cell(plan, "base", seed)
        assert src.init_image is None
        assert dep.init_image == em.final_frame_path(plan, src)


def test_conf_text_carries_source_final_frame_path(tmp_path):
    plan = _plan("finish:init_from=base,direct_init_weight=1", "base", root=tmp_path)
    dep, src = _cell(plan, "finish", 2), _cell(plan, "base", 2)
    conf = yaml.safe_load(em.cell_conf_text(dep, plan.tier))
    assert conf["init_image"] == str(em.final_frame_path(plan, src))
    assert conf["direct_init_weight"] == 1
    # the pseudo-override never reaches the render config
    assert "init_from" not in conf
    # unchained cells' confs are byte-identical to before (resumability)
    assert "init_image" not in yaml.safe_load(em.cell_conf_text(src, plan.tier))


def test_depth2_chain_init_paths_hop_leg_by_leg(tmp_path):
    plan = _plan("finish:init_from=mid", "mid:init_from=base", "base", root=tmp_path)
    base, mid, fin = (_cell(plan, name, 2) for name in ("base", "mid", "finish"))
    assert mid.init_image == em.final_frame_path(plan, base)
    assert fin.init_image == em.final_frame_path(plan, mid)
    assert [c.leg.name for c in plan.cells[::2]] == ["base", "mid", "finish"]


# ----------------------------------------------------------------------
# missing-source guard + resumability
# ----------------------------------------------------------------------


def test_missing_source_final_fails_loud_naming_the_source_cell(tmp_path):
    plan = _plan("finish:init_from=base", "base", root=tmp_path)
    dep = _cell(plan, "finish", 2)
    with pytest.raises(
        SystemExit,
        match=r"finish/temple-lit_s2: source cell base/temple-lit_s2 has no final frame",
    ):
        em.require_source_final(dep)


def test_present_source_final_passes_the_guard(tmp_path):
    plan = _plan("finish:init_from=base", "base", root=tmp_path)
    dep, src = _cell(plan, "finish", 2), _cell(plan, "base", 2)
    _fake_frame(em.final_frame_path(plan, src))
    em.require_source_final(dep)  # must not die
    em.require_source_final(src)  # unchained: nothing to check


def test_dependent_cell_skips_when_its_final_frame_exists(tmp_path):
    plan = _plan("finish:init_from=base", "base", root=tmp_path)
    dep = _cell(plan, "finish", 2)
    _fake_frame(em.final_frame_path(plan, dep))
    conf_path = em.cell_conf_path(plan, dep)
    conf_path.parent.mkdir(parents=True)
    conf_path.write_text(em.cell_conf_text(dep, plan.tier), encoding="utf-8")
    # skips like any other cell — even though the SOURCE final is absent
    assert em.cell_is_done(plan, dep) is True


def test_dependent_cell_without_final_frame_is_not_done(tmp_path):
    plan = _plan("finish:init_from=base", "base", root=tmp_path)
    assert em.cell_is_done(plan, _cell(plan, "finish", 2)) is False


def test_dependent_cell_with_stale_conf_dies(tmp_path):
    # same leg name, different chain -> the conf under the final frame no
    # longer matches this plan; a silent re-skip would judge stale pixels
    old = _plan("finish:init_from=base", "base", root=tmp_path)
    dep = _cell(old, "finish", 2)
    _fake_frame(em.final_frame_path(old, dep))
    conf_path = em.cell_conf_path(old, dep)
    conf_path.parent.mkdir(parents=True)
    conf_path.write_text(em.cell_conf_text(dep, old.tier), encoding="utf-8")
    new = _plan("finish:init_from=base,direct_init_weight=1", "base", root=tmp_path)
    with pytest.raises(SystemExit, match="does not match this plan"):
        em.cell_is_done(new, _cell(new, "finish", 2))


# ----------------------------------------------------------------------
# report + metrics self-documentation
# ----------------------------------------------------------------------


def _synthetic_scores(plan):
    scores = {judge: {} for judge in em.JUDGES}
    for i, cell in enumerate(plan.cells):
        for j, judge in enumerate(em.JUDGES):
            scores[judge][em.cell_key(cell)] = 0.1 * (i + 1) + 0.01 * j
    return scores


def test_report_leg_table_shows_the_chain(tmp_path):
    plan = _plan("finish:init_from=base,direct_init_weight=1", "base", root=tmp_path)
    scores = _synthetic_scores(plan)
    report = em.write_report(
        plan, scores, [], em.prompt_rows(scores, plan), tmp_path
    ).read_text(encoding="utf-8")
    assert "| finish | explicit | `init_from: base, direct_init_weight=1` |" in report
    assert "| base | explicit | `(schema defaults)` |" in report


def test_metrics_json_records_init_from(tmp_path):
    plan = _plan("finish:init_from=base", "base", root=tmp_path)
    scores = _synthetic_scores(plan)
    em.write_metrics_json(plan, scores, [], em.prompt_rows(scores, plan), tmp_path)
    legs = {
        leg["name"]: leg
        for leg in json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["legs"]
    }
    assert legs["finish"]["init_from"] == "base"
    assert legs["base"]["init_from"] is None
    # init_from is consumed, not an override
    assert legs["finish"]["overrides"] == {}
