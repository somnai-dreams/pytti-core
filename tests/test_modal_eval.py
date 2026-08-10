"""
Unit tests for scripts/modal_eval.py — the Modal scheduling/transport layer
over eval_matrix. Covers: plan-reuse equivalence (modal_eval's plan for a
given CLI == eval_matrix's own machinery), wave partitioning (flat, diamond,
depth-2 chain), conf init_image rewrite for byte-carried inits, the
torch-backend gate, resumability skip, cost-estimator arithmetic, and the
wave driver end-to-end with the Modal call mocked (the paid smoke is the
real integration test). Pure CPU, no modal import, no downloads, no renders.
"""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # py3.10 dataclasses resolve cls.__module__ through sys.modules at class
    # creation; exec_module without registration crashes
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


me = _load_script("modal_eval")
em = me.em  # the eval_matrix module modal_eval itself imported


def _png_bytes(size=(8, 8)) -> bytes:
    """A tiny NON-uniform png (the driver rejects uniform finals)."""
    from PIL import Image

    img = Image.new("RGB", size)
    img.putdata(
        [(x * 30 % 256, y * 40 % 256, (x + y) * 20 % 256)
         for y in range(size[1]) for x in range(size[0])]
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _args(*argv: str):
    return me.parse_args(list(argv))


def _campaign(*argv: str, root: Path):
    return me.build_campaign(_args(*argv), root=root)


# ----------------------------------------------------------------------
# plan-reuse equivalence: modal_eval must produce eval_matrix's plan
# ----------------------------------------------------------------------


def test_plan_equals_eval_matrix_plan_simple(tmp_path):
    campaign = _campaign(
        "--legs", "default", "--legs", "full16:cutout_sampler=full,cutouts=16",
        "--baseline", "base:cutouts=8", "--subset", "3", "--tier", "full",
        root=tmp_path,
    )
    leg_set = em.assemble_legs(
        [
            em.parse_leg_spec("default"),
            em.parse_leg_spec("full16:cutout_sampler=full,cutouts=16"),
        ],
        em.parse_leg_spec("base:cutouts=8", source="baseline"),
        [],
    )
    battery = em.parse_battery(em.DEFAULT_BATTERY)
    expected = em.build_plan(battery, leg_set, 3, em.TIERS["full"], root=tmp_path)
    assert campaign.plan == expected
    assert campaign.leg_set == leg_set


def test_plan_equals_eval_matrix_plan_chained_and_swept(tmp_path):
    campaign = _campaign(
        "--legs", "src", "--legs", "dep:init_from=src,direct_init_weight=1",
        "--sweep", "cutouts=8,16", "--baseline", "src2", "--subset", "1",
        root=tmp_path,
    )
    leg_set = em.assemble_legs(
        [
            em.parse_leg_spec("src"),
            em.parse_leg_spec("dep:init_from=src,direct_init_weight=1"),
        ],
        em.parse_leg_spec("src2", source="baseline"),
        [em.parse_sweep_spec("cutouts=8,16")],
    )
    battery = em.parse_battery(em.DEFAULT_BATTERY)
    expected = em.build_plan(battery, leg_set, 1, em.TIERS["screening"], root=tmp_path)
    assert campaign.plan == expected
    assert [s.field for s in campaign.sweeps] == ["cutouts"]


def test_default_root_is_separate_from_eval_matrix():
    assert me.MODAL_EVAL_ROOT != em.EVAL_ROOT


def test_smoke_tier_exists_only_in_modal_eval():
    assert "smoke" in me.TIERS
    assert "smoke" not in em.TIERS
    tier = me.TIERS["smoke"]
    assert tier.dims["1:1"] == (128, 128)
    assert tier.steps == 20 and tier.save_every == 10
    assert tuple(tier.dims) == em.ASPECTS


# ----------------------------------------------------------------------
# torch-backend gate
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["mlx", "mlx_full"])
def test_non_torch_backend_dies_naming_constraint(tmp_path, backend):
    with pytest.raises(SystemExit, match="Metal-only"):
        _campaign(
            "--legs", f"bad:perceptor_backend={backend}", "--subset", "1",
            root=tmp_path,
        )


def test_explicit_torch_backend_and_default_pass(tmp_path):
    campaign = _campaign(
        "--legs", "ok:perceptor_backend=torch", "--legs", "default",
        "--subset", "1", root=tmp_path,
    )
    assert [leg.name for leg in campaign.leg_set.legs] == ["ok", "default"]


def test_non_cuda_device_override_dies(tmp_path):
    with pytest.raises(SystemExit, match="device"):
        _campaign("--legs", "bad:device=mps", "--subset", "1", root=tmp_path)


# ----------------------------------------------------------------------
# waves
# ----------------------------------------------------------------------


def _wave_leg_names(plan) -> list[set]:
    return [{c.leg.name for c in wave} for wave in me.waves(plan)]


def test_waves_flat_legs_are_one_wave(tmp_path):
    campaign = _campaign(
        "--legs", "a", "--legs", "b:cutouts=8", "--subset", "2", root=tmp_path
    )
    assert _wave_leg_names(campaign.plan) == [{"a", "b"}]


def test_waves_diamond(tmp_path):
    # a -> (b, c) -> d : b and c share wave 2, d alone in wave 3
    campaign = _campaign(
        "--legs", "a",
        "--legs", "b:init_from=a",
        "--legs", "c:init_from=a,cutouts=8",
        "--legs", "d:init_from=b",
        "--subset", "1", root=tmp_path,
    )
    assert _wave_leg_names(campaign.plan) == [{"a"}, {"b", "c"}, {"d"}]


def test_waves_partition_every_cell_exactly_once(tmp_path):
    campaign = _campaign(
        "--legs", "a", "--legs", "b:init_from=a", "--legs", "c:init_from=b",
        "--subset", "2", root=tmp_path,
    )
    plan = campaign.plan
    seen = [em.cell_key(c) for wave in me.waves(plan) for c in wave]
    assert sorted(seen) == sorted(em.cell_key(c) for c in plan.cells)
    assert len(seen) == len(set(seen))


# ----------------------------------------------------------------------
# conf init_image rewrite (byte-carried chained inits)
# ----------------------------------------------------------------------


def _chained_cells(plan):
    dep = [c for c in plan.cells if c.init_image is not None]
    assert dep
    return dep


def test_rewrite_conf_init_image_preserves_everything_else(tmp_path):
    campaign = _campaign(
        "--legs", "src", "--legs", "dep:init_from=src,direct_init_weight=1",
        "--subset", "1", root=tmp_path,
    )
    cell = _chained_cells(campaign.plan)[0]
    original = em.cell_conf_text(cell, campaign.plan.tier)
    new_path = me.container_init_path(cell)
    rewritten = me.rewrite_conf_init_image(original, new_path)

    def split(text):
        lines = text.splitlines(keepends=True)
        n = 0
        while n < len(lines) and lines[n].startswith("#"):
            n += 1
        return "".join(lines[:n]), yaml.safe_load("".join(lines[n:]))

    orig_header, orig_data = split(original)
    new_header, new_data = split(rewritten)
    assert new_header == orig_header  # hydra's @package header survives
    assert "# @package _global_" in new_header
    assert new_data.pop("init_image") == new_path
    orig_data.pop("init_image")
    assert new_data == orig_data
    assert list(new_data) == list(orig_data)  # key order stable


def test_rewrite_conf_without_init_image_dies(tmp_path):
    campaign = _campaign("--legs", "solo", "--subset", "1", root=tmp_path)
    cell = campaign.plan.cells[0]
    conf = em.cell_conf_text(cell, campaign.plan.tier)
    with pytest.raises(SystemExit, match="init_image"):
        me.rewrite_conf_init_image(conf, "/work/x/init.png")


def test_container_paths_are_unique_across_legs(tmp_path):
    campaign = _campaign(
        "--legs", "src", "--legs", "dep:init_from=src", "--subset", "1",
        root=tmp_path,
    )
    slugs = [me.cell_slug(c) for c in campaign.plan.cells]
    assert len(slugs) == len(set(slugs))  # same cell name, different legs


def test_final_rel_matches_eval_matrix_layout(tmp_path):
    campaign = _campaign("--legs", "solo", "--subset", "1", root=tmp_path)
    plan = campaign.plan
    cell = plan.cells[0]
    workdir = em.cell_workdir(plan, cell)
    assert workdir / me.final_rel(cell) == em.final_frame_path(plan, cell)


# ----------------------------------------------------------------------
# resumability
# ----------------------------------------------------------------------


def _mark_done(plan, cell) -> None:
    em.prepare_cell_workspace(plan, cell)
    final = em.final_frame_path(plan, cell)
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(_png_bytes())


def test_partition_pending_skips_done_cells(tmp_path):
    campaign = _campaign("--legs", "solo", "--subset", "1", root=tmp_path)
    plan = campaign.plan
    _mark_done(plan, plan.cells[0])
    done, pending = me.partition_pending(plan)
    assert [em.cell_key(c) for c in done] == [em.cell_key(plan.cells[0])]
    assert [em.cell_key(c) for c in pending] == [
        em.cell_key(c) for c in plan.cells[1:]
    ]


def test_partition_pending_dies_on_conf_mismatch(tmp_path):
    campaign = _campaign("--legs", "solo", "--subset", "1", root=tmp_path)
    plan = campaign.plan
    cell = plan.cells[0]
    _mark_done(plan, cell)
    em.cell_conf_path(plan, cell).write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match this plan"):
        me.partition_pending(plan)


# ----------------------------------------------------------------------
# cost estimator
# ----------------------------------------------------------------------


def test_estimate_dollars_arithmetic():
    # 20 cells x 6 min = 2 GPU-hours; L4 = $0.80/h
    assert me.estimate_dollars(20, 6.0, "L4") == pytest.approx(1.60)
    assert me.estimate_dollars(0, 6.0, "L4") == 0.0
    assert me.estimate_dollars(80, 5.0, "A10G") == pytest.approx(
        80 * 5.0 / 60.0 * 1.10
    )


def test_unknown_gpu_dies():
    with pytest.raises(SystemExit, match="no price on file"):
        me.gpu_price("B300")


def test_every_cli_gpu_choice_is_priced():
    for gpu in me.GPU_DOLLARS_PER_HOUR:
        assert me.gpu_price(gpu) > 0


# ----------------------------------------------------------------------
# wave driver end-to-end, Modal mocked
# ----------------------------------------------------------------------


class FakeRemote:
    """Stands in for the Modal .starmap: records job batches, returns
    per-cell results, and remembers the bytes it produced per cell."""

    def __init__(self):
        self.batches: list[list[tuple]] = []
        self.finals: dict[str, bytes] = {}
        self.fail_slug: str | None = None

    def __call__(self, jobs: list[tuple]) -> list[dict]:
        self.batches.append(jobs)
        results = []
        for slug, _conf_name, _conf_text, _final_rel, _init_png in jobs:
            if slug == self.fail_slug:
                results.append(
                    {
                        "ok": False,
                        "reason": "workhorse exit code 1",
                        "log_tail": "RuntimeError: synthetic boom",
                        "wall_s": 1.0,
                        "gpu_name": "FAKE",
                    }
                )
                continue
            png = _png_bytes()
            self.finals[slug] = png
            results.append(
                {
                    "ok": True,
                    "final_png": png,
                    "log_tail": "20/20 [00:01]",
                    "wall_s": 1.5,
                    "gpu_name": "FAKE",
                }
            )
        return results


def test_run_waves_chained_end_to_end(tmp_path):
    campaign = _campaign(
        "--legs", "src", "--legs", "dep:init_from=src,direct_init_weight=1",
        "--subset", "1", root=tmp_path,
    )
    plan = campaign.plan
    remote = FakeRemote()
    _, pending = me.partition_pending(plan)
    metas = me.run_waves(plan, pending, submit=remote, echo=lambda _: None)

    # two waves: src cells first, then dep cells
    assert len(remote.batches) == 2
    wave1, wave2 = remote.batches
    assert all(init is None for *_, init in wave1)
    src_cells = [c for c in plan.cells if c.leg.name == "src"]
    dep_cells = [c for c in plan.cells if c.leg.name == "dep"]
    assert len(wave1) == len(src_cells) and len(wave2) == len(dep_cells)

    # every final landed at eval_matrix's exact local path, with meta + log
    for cell in plan.cells:
        final = em.final_frame_path(plan, cell)
        assert final.is_file()
        assert final.read_bytes() == remote.finals[me.cell_slug(cell)]
        workdir = em.cell_workdir(plan, cell)
        assert (workdir / "render.log").is_file()
        meta = json.loads((workdir / "modal_meta.json").read_text())
        assert meta["wall_s"] == 1.5 and meta["gpu_name"] == "FAKE"
        # the LOCAL conf stays canonical (local init path, not container's)
        conf_on_disk = em.cell_conf_path(plan, cell).read_text(encoding="utf-8")
        assert conf_on_disk == em.cell_conf_text(cell, plan.tier)
    assert set(metas) == {em.cell_key(c) for c in plan.cells}

    # chained jobs carried the SOURCE's final bytes + the container init path
    for dep, job in zip(dep_cells, wave2, strict=True):
        slug, _name, conf_text, _rel, init_png = job
        assert slug == me.cell_slug(dep)
        source = next(
            c for c in src_cells
            if c.prompt.id == dep.prompt.id and c.seed == dep.seed
        )
        assert init_png == remote.finals[me.cell_slug(source)]
        assert yaml.safe_load(
            "".join(
                line for line in conf_text.splitlines(keepends=True)
                if not line.startswith("#")
            )
        )["init_image"] == me.container_init_path(dep)


def test_run_waves_skips_done_source_but_chains_from_its_final(tmp_path):
    campaign = _campaign(
        "--legs", "src", "--legs", "dep:init_from=src",
        "--subset", "1", root=tmp_path,
    )
    plan = campaign.plan
    src_cells = [c for c in plan.cells if c.leg.name == "src"]
    for cell in src_cells:
        _mark_done(plan, cell)
    done, pending = me.partition_pending(plan)
    assert {c.leg.name for c in done} == {"src"}
    remote = FakeRemote()
    me.run_waves(plan, pending, submit=remote, echo=lambda _: None)
    assert len(remote.batches) == 1  # only the dep wave went to Modal
    for job in remote.batches[0]:
        *_, init_png = job
        assert init_png == _png_bytes()  # the resumed source's local final


def test_run_waves_render_failure_dies_with_log_tail(tmp_path):
    campaign = _campaign("--legs", "solo", "--subset", "1", root=tmp_path)
    plan = campaign.plan
    remote = FakeRemote()
    remote.fail_slug = me.cell_slug(plan.cells[0])
    _, pending = me.partition_pending(plan)
    with pytest.raises(SystemExit, match="synthetic boom"):
        me.run_waves(plan, pending, submit=remote, echo=lambda _: None)


def test_mid_wave_failure_persists_paid_successes(tmp_path):
    """A failed cell mid-wave must not discard its wave-mates' PAID renders:
    successes persist and resume as done; only the failure re-renders."""
    campaign = _campaign("--legs", "solo", "--subset", "2", root=tmp_path)
    plan = campaign.plan
    remote = FakeRemote()
    failed = plan.cells[0]
    remote.fail_slug = me.cell_slug(failed)
    _, pending = me.partition_pending(plan)
    with pytest.raises(SystemExit, match="1/4 renders failed"):
        me.run_waves(plan, pending, submit=remote, echo=lambda _: None)
    for cell in plan.cells[1:]:
        assert em.final_frame_path(plan, cell).is_file()
    done, still_pending = me.partition_pending(plan)
    assert [em.cell_key(c) for c in done] == [em.cell_key(c) for c in plan.cells[1:]]
    assert [em.cell_key(c) for c in still_pending] == [em.cell_key(failed)]


def test_infra_exception_mid_stream_keeps_earlier_finals(tmp_path):
    """submit is consumed lazily: results already yielded before an infra
    exception (container loss after retries) are persisted, not lost."""
    campaign = _campaign("--legs", "solo", "--subset", "2", root=tmp_path)
    plan = campaign.plan

    def submit(jobs):
        yield {
            "ok": True,
            "final_png": _png_bytes(),
            "log_tail": "20/20",
            "wall_s": 1.0,
            "gpu_name": "FAKE",
        }
        raise RuntimeError("container preempted after retries")

    _, pending = me.partition_pending(plan)
    with pytest.raises(RuntimeError, match="preempted"):
        me.run_waves(plan, pending, submit=submit, echo=lambda _: None)
    done, _ = me.partition_pending(plan)
    assert [em.cell_key(c) for c in done] == [em.cell_key(plan.cells[0])]


def test_uniform_final_dies_as_dead_render_and_is_not_left_on_disk(tmp_path):
    from PIL import Image

    campaign = _campaign("--legs", "solo", "--subset", "1", root=tmp_path)
    plan = campaign.plan

    black = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 0, 0)).save(black, format="PNG")

    class BlackRemote(FakeRemote):
        def __call__(self, jobs):
            results = super().__call__(jobs)
            for r in results:
                r["final_png"] = black.getvalue()
            return results

    _, pending = me.partition_pending(plan)
    with pytest.raises(SystemExit, match="uniform"):
        me.run_waves(plan, pending, submit=BlackRemote(), echo=lambda _: None)
    # the dead frame must NOT satisfy resumability — a rerun re-renders it
    for cell in plan.cells:
        assert not em.final_frame_path(plan, cell).is_file()
    done, still_pending = me.partition_pending(plan)
    assert done == []
    assert len(still_pending) == len(plan.cells)


# ----------------------------------------------------------------------
# spend discipline: paid submission requires the explicit --launch flag
# ----------------------------------------------------------------------


def test_launch_gate_refuses_without_flag(tmp_path):
    campaign = _campaign("--legs", "solo", "--subset", "1", root=tmp_path)
    _, pending = me.partition_pending(campaign.plan)
    assert pending
    with pytest.raises(SystemExit, match="--launch"):
        me.launch_gate(pending, launch=False)


def test_launch_gate_passes_with_flag_or_nothing_pending(tmp_path):
    campaign = _campaign("--legs", "solo", "--subset", "1", root=tmp_path)
    _, pending = me.partition_pending(campaign.plan)
    me.launch_gate(pending, launch=True)  # explicit spend: no exception
    me.launch_gate([], launch=False)  # judge-only resume: free, no flag


def test_parse_args_launch_defaults_off():
    assert _args("--legs", "solo").launch is False
    assert _args("--legs", "solo", "--launch").launch is True


# ----------------------------------------------------------------------
# smoke battery fixture + dry-run summary
# ----------------------------------------------------------------------


def test_smoke_battery_two_cell_campaign(tmp_path):
    campaign = _campaign(
        "--tier", "smoke",
        "--battery", str(REPO_ROOT / "tests" / "fixtures" / "eval" / "battery_smoke.yaml"),
        "--legs", "plain:image_model=Limited Palette,ViTB32=false,FARE4ViTB32=true,cutouts=8",
        "--legs", "finish:init_from=plain,image_model=Limited Palette,"
        "ViTB32=false,FARE4ViTB32=true,cutouts=8,direct_init_weight=1",
        root=tmp_path,
    )
    plan = campaign.plan
    assert len(plan.cells) == 2
    assert _wave_leg_names(plan) == [{"plain"}, {"finish"}]
    for cell in plan.cells:
        assert (cell.width, cell.height) == (128, 128)
        assert cell.steps == 20 and cell.final_index == 2
    conf = em.cell_conf_text(plan.cells[0], plan.tier)
    data = yaml.safe_load(
        "".join(
            line for line in conf.splitlines(keepends=True)
            if not line.startswith("#")
        )
    )
    assert data["image_model"] == "Limited Palette"
    assert data["FARE4ViTB32"] is True and data["ViTB32"] is False


def test_format_modal_summary_reports_waves_and_cost(tmp_path):
    campaign = _campaign(
        "--legs", "src", "--legs", "dep:init_from=src", "--subset", "1",
        root=tmp_path,
    )
    plan = campaign.plan
    done, pending = me.partition_pending(plan)
    text = me.format_modal_summary(
        plan, done, pending, "L4", 2.0, max_parallel=10, timeout_min=20
    )
    assert me.PINNED_COMMIT[:7] in text
    assert "wave 1" in text and "wave 2" in text
    n = len(pending)
    assert f"{n} cells x 2 min x $0.80/h = ${n * 2 / 60 * 0.80:.2f}" in text
