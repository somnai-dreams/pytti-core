"""
Golden generator for the coherence_weighting OFF-path gate
(tests/test_coherence_weighting.py::TestMlxOffPathGolden).

MUST be generated from a tree WITHOUT the coherence feature active (it was
first generated from commit b839cfe, the last commit before the feature
landed) — the fixture pins "coherence_weighting off == the old engine
behavior" for the mlx_full backend: stub-tower engine, fixed seed, 3 train
steps, every loss record + the final params tree.

Regenerate (only if an UNRELATED upstream change legitimately moves the
engine's draws/maths — never to paper over a coherence regression):

    .venv/bin/python tests/fixtures/gen_coherence_off_golden.py

Determinism note: MLX's gather-bilinear backward is a GPU scatter-add with
unfixed accumulation order (see test_mlx_engine_step.py::
test_deterministic_given_seed) — identical runs spread <= ~1e-7 rel from
step 1's BACKWARD onward. Step-1 forward records are bit-reproducible; the
test gates step 1 tightly and later steps at the measured floor.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_PATH = Path(__file__).parent / "coherence_off_golden_mlx.npz"
STEPS = 3


def run_engine() -> dict[str, np.ndarray]:
    from tests.test_mlx_engine_step import _make_engine, _prompt, _tv_loss

    engine = _make_engine()
    prompts = [
        _prompt("a test prompt:1"),
        _prompt("left side:1_r_0.4", seed=2),
    ]
    augs = [_tv_loss(0.02)]
    arrays: dict[str, np.ndarray] = {}
    for i in range(STEPS):
        record = engine.train_step(i, prompts, [], augs)
        for name, value in record.items():
            arrays[f"record/{i}/{name}"] = np.array(value)
    for key, value in engine._params.items():
        arrays[f"params/{key}"] = np.array(value)
    return arrays


def main() -> None:
    arrays = run_engine()
    # measure the same-code determinism floor while we're here (two engine
    # builds, same seed) so the test's gates are grounded in a measurement
    repeat = run_engine()
    floor = 0.0
    for key, value in arrays.items():
        ref = np.abs(value.astype(np.float64)).max()
        diff = np.abs(value.astype(np.float64) - repeat[key].astype(np.float64)).max()
        floor = max(floor, diff / max(ref, 1e-12))
    np.savez(OUT_PATH, **arrays)
    print(f"wrote {OUT_PATH} ({len(arrays)} arrays, {STEPS} steps)")
    print(f"same-code rebuild determinism floor: max rel {floor:.3e}")
    step1_exact = all(
        np.array_equal(arrays[k], repeat[k])
        for k in arrays
        if k.startswith("record/0/")
    )
    print(f"step-1 records bit-reproducible across rebuilds: {step1_exact}")


if __name__ == "__main__":
    main()
