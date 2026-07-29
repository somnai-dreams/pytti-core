# pytti v2 Engine Modernization Plan

**Date:** 2026-07-29 · **Branch:** v2 · **Primary target:** Apple Silicon (MPS, torch 2.13, fp32)

Synthesized from five subsystem research reports (perceptors, depth, optical flow,
optimizer/perf, model distribution), grounded against the current source tree. All file
references below were verified against the code as of this writing.

The prime directive: **the dreamlike evolving-animation aesthetic of the VQGAN+CLIP era is
THE PRODUCT.** We modernize the machinery — loaders, hosts, dead forks, slow paths — while
preserving the medium by construction wherever possible, and A/B-gating it everywhere else.

---

## 1. Vision

Post-modernization, pytti v2 installs with one `pip install` from PyPI — zero git-fork
dependencies, zero 2021-era download hosts, every checkpoint fetched sha256-verified and
revision-pinned from Hugging Face or the torchvision CDN — and runs first-class on a Mac:
the cutout pipeline is sync-free batched (measured 1.7x on the sampler stage), the optimizer
is schedule-free with a principled averaged iterate that reduces inter-frame shimmer, and
every known MPS silent-wrongness pathology is behind a fail-loud fence with a CI parity test
pinning it. "Better" means the same signature looks (identical OpenAI CLIP weights, identical
six taming checkpoints, identical spherical loss and Crowson cutouts) plus an *additive*
palette: adversarially-robust CLIP members with perceptually-aligned gradients, SigLIP2
members that actually resolve compositional prompts, Depth Anything V2 giving crisp parallax
instead of AdaBins blob-depth, and a LlamaGen VQ image model that keeps the codebook-snapping
evolution but decodes at rFID ~1.03 instead of ~4–5. "Smarter" means learning-rate and EMA
questions answered by the optimizer instead of by config folklore. Nothing about the prompt
DSL, config schema, or session semantics breaks; new capability is new config keys, never
changed ones.

---

## 2. Phased slices

Ordered by value/risk: no-brainers and time-critical rescues first, aesthetic-affecting
swaps gated behind the A/B harness, palette expansions last. Each slice ends in a commit
(or small commit series) with its verification evidence in the message.

### Slice 1 — Optical flow: GMA fork → torchvision RAFT

**No-brainer. Empirically pre-verified on this machine** (torch 2.13.0 / torchvision 0.28.0 /
MPS, fp32): raft_large 512² pair 113 ms @ 12 iters / 39 ms @ 3 iters; MPS-vs-CPU max flow
diff 7.2e-5 px; backprop-through-model produces finite nonzero input grads (required — 
`TargetFlowLoss.get_loss` differentiates *through* the flow net).

- **What changes:** `pyttitools-gma` git dep deleted; flow model becomes
  `torchvision.models.optical_flow.raft_large` (weights `Raft_Large_Weights.C_T_SKHT_V2`),
  with `raft_small` (`C_T_V2`) as a fast-preview knob. Weights land in the standard
  torch.hub cache (~20 MB / ~4 MB).
- **Files/seams** (all under `/Users/max/Documents/Dev/pytti-core`):
  - `pyproject.toml` — delete both `pyttitools-gma @ git+...` lines (40, 44); `video` extra
    becomes empty (keep the extra name as a no-op shim); update the `extras` pytest marker
    text (line 98) to adabins-only (removed entirely by Slice 4).
  - `src/pytti/LossAug/OpticalFlowLossClass.py` — the only file importing `gma`. Replace
    `init_GMA`/`get_gma_checkpoint_path` (lines 23–60) with a lazy singleton under
    `vram_usage_mode("RAFT")`. Input-contract changes: torchvision RAFT wants `[-1,1]`
    (current code feeds `[0,1]` — `img.mul(2).sub(1)`) and H,W divisible by 8 — replace the
    two `from gma.core.utils.utils import InputPadder` imports (lines 149, 265) with a
    ~10-line local `_pad8`/crop-back helper. Call shape: torchvision returns a *list* of
    flow iterates — `flow = _FLOW(im1, im2, num_flow_updates=12)[-1]` replaces
    `_, flow_up = GMA(im1, im2, iters=12, test_mode=True)` in `get_flow` (line 275), and
    `num_flow_updates=3` replaces `iters=3` in `TargetFlowLoss.get_loss` (line 155), which
    **must stay outside `no_grad`**. `motion_edge_map`, `set_flow`, the bg-mask noise trick,
    and the rotoscoper path are model-agnostic — untouched.
  - `LossOrchestratorClass.py` / `Transforms.py` — no changes; they call through
    `set_flow`/`get_flow`.
- **New deps:** none (torchvision already core).
- **Config surface:** additive `flow_model: raft_large | raft_small` (default `raft_large`).
- **MPS verification:** re-run the existing bench (`/tmp/raft_mps_bench.py` methodology) in
  a checked-in test: synthetic (8,4)-px shift recovered < 0.05 px; MPS-vs-CPU flow diff
  < 1e-3 px; grad-through-model finite. Smoke a 10-frame Video Source run; eyeball the
  motion-edge mask output vs a GMA reference frame pair (one-time, before deleting the env
  that still has GMA).
- **Honest quality note:** GMA modestly outscores RAFT on Sintel EPE (2.47 vs 3.07 final);
  irrelevant here — flow feeds consistency masks and a 3-iter matching loss, both dominated
  by the mask gating. SEA-RAFT vendoring is the documented accuracy escape hatch if output
  ever visibly regresses.
- **Effort:** hours.

### Slice 2 — VQGAN checkpoint rescue: self-owned HF mirror (TIME-CRITICAL)

**The wikiart host (`eaidata.bmk.sh`, `vqgan.py` lines 33/50) is already dead** — wikiart is
broken *today* for fresh installs — and two hosts are plain HTTP. Seed the mirror
immediately, while heibox/nmkd/koofr/pixray are still live.

- **What changes:** all six taming ckpt+yaml pairs move to one HF model repo we control
  (e.g. `pytti-tools/vqgan-taming`), fetched via `huggingface_hub.hf_hub_download` with a
  pinned `revision` sha — sha256-verified, resumable, cached. Checkpoints stay
  **byte-identical**; this slice has zero aesthetic surface.
- **Files/seams:**
  - One-time `scripts/mirror_vqgan.py` — download the six pairs from verified-live sources
    (imagenet/sflickr/openimages: heibox links already in `vqgan.py`; coco: `dl.nmkd.de`;
    faceshq: koofr — use GET not HEAD, it 405s on HEAD; **wikiart: pixray GitHub release
    v1.7.1 assets**, the rescue for the dead host), record sha256s, upload.
  - `src/pytti/image_models/vqgan.py` — delete `VQGAN_CONFIG_URLS`/`VQGAN_CHECKPOINT_URLS`
    (lines 28–61) and `_download` (line 64); `init_vqgan` (line 277) becomes
    `hf_hub_download(repo_id=..., filename=f"{model_name}.{yaml|ckpt}", revision=PINNED_SHA,
    local_dir=model_artifacts_path)`. Keeping `model_artifacts_path` as `local_dir` means
    already-downloaded checkpoints on users' disks are reused as-is.
  - `src/pytti/config/model_names.py` — unchanged (`sflckr` alias preserved).
- **New deps:** `huggingface_hub>=0.20` (pure Python, PyPI).
- **Config surface:** none. Zero schema change.
- **MPS verification:** n/a (download plumbing). Gate: sha256 of every mirrored file matches
  the original download; `tests/test_models_load.py` `@pytest.mark.download` round-trip on
  one model.
- **Effort:** ~1 day (mirror-seeding script can run today).

### Slice 3 — A/B measurement harness + MPS parity baseline

Prerequisite infrastructure for every aesthetic-affecting slice below (details in §4).

- **What changes:** a checked-in golden-scene suite + `scripts/ab_render.py` that renders
  fixed-seed runs under two installed variants and emits contact sheets, side-by-side mp4s,
  loss trajectories, s/step, and peak memory; `vram_tools.py` (currently CUDA-only, see its
  module docstring) grows an MPS backend via `torch.mps.current_allocated_memory()` so the
  `vram_usage_mode` calibration buckets work on Mac.
- **Files/seams:** `src/pytti/vram_tools.py` (`_allocated`, `vram_profiling` gate);
  `tests/fixtures/` golden configs (Limited Palette 2D, VQGAN-wikiart 3D, Video Source);
  new `scripts/ab_render.py`; a `debug_mps_parity` startup check in `src/pytti/device.py`
  (20-step Limited Palette run MPS vs CPU, assert loss-trajectory tolerance) — `device.py`
  is already the single source of truth for device decisions and hosts the channels_last
  fence.
- **New deps:** `lpips` (or `torchmetrics[image]`) as a **dev-group** dependency only, for
  flicker/drift metrics.
- **Config surface:** none user-facing.
- **Effort:** ~1 day.

### Slice 4 — Depth: AdaBins fork → Depth Anything V2 Small

Kills the broken checkpoint download and the `pyttitools-adabins` + `gdown` fork deps.
DA V2-S: 24.8M params, Apache-2.0, plain transformers install, real-time-class, and its
forward pass is differentiable — preserving the depth-consistency loss that backprops
*through* the depth net.

- **Files/seams:**
  - `src/pytti/LossAug/DepthLossClass.py` — replace `init_AdaBins`/`infer_helper`
    (lines 16–27) with a lazy `AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf")`. Rewrite `_model_depth` (lines 30–42):
    tensor-native ImageNet-normalize + resize to multiples of 14 (cap long side ~518,
    replacing the ad-hoc 500k-pixel area cap), forward, `outputs.predicted_depth` (relative
    **inverse** depth, bigger = closer), unsqueeze channel dim. **No `torch.no_grad` inside**
    — `get_loss` (line 53) must keep backpropping through the net. `set_comp`/`make_comp`
    keep their `@torch.no_grad` and the same-pipeline invariant already documented at
    lines 64–67. Skip `AutoImageProcessor` (PIL round-trip would break the differentiable
    path). Add a small `DepthScaler` class holding EMA-smoothed 2nd/98th-percentile lo/hi
    (alpha ~0.1, reset per run) — the anti-flicker normalizer.
  - `src/pytti/Transforms.py` `zoom_3d` — replace the AdaBins metric remap
    `np.interp(depth_map, (1e-3, 10), (near*px, far*px))` (line 287) with: normalize
    disparity to [0,1] via the `DepthScaler`, then `depth = (near + (1-d_norm)*(far-near))*px`.
    Downstream (`view_pos = cat([xy, -depth, ones])`, line 221) unchanged. The `r/R/mu`
    parametric-eval values (lines 289–296) keep their meaning (pixel-space near/far range).
  - `pyproject.toml` — `threed = ["transformers>=4.45"]`; drop `pyttitools-adabins` and
    `gdown` from `threed` and `animation`.
- **New deps:** `transformers>=4.45` (optional extra), weights from HF hub.
- **Config surface:** additive `depth_model_size: small` (default; `base`/`large` opt-in
  with a CC-BY-NC-4.0 license warning logged — only Small is Apache-2.0).
- **MPS verification:** (a) benchmark grad-through-ViT-S fwd+bwd explicitly (the 100x
  pathology was a *gradient* pathology — measure, don't assume; AdaBins was ~78M params with
  the same backprop-through-model structure, so this should be a strict improvement);
  (b) CPU-vs-MPS golden-depth-map tolerance test (covers the `align_corners=True`
  interpolate history); (c) smoke test asserting grad reaches the image through
  `DepthLoss.get_loss`; (d) A/B per §4: same-seed 3D scene, old vs new — judged by eye for
  parallax articulation and by flicker metric across 30 frames.
- **Effort:** ~1 day.

### Slice 5 — Perceptor loader: openai/CLIP pip fork → open_clip (same weights)

Default tier is **bit-comparable by construction**: the identical OpenAI ViT-B/32, B/16,
RN50… weights, loaded via `open_clip`'s `'openai'` pretrained tags from HF hub instead of
the `clip @ git+github.com/openai/CLIP` dep (pyproject line 30). The signature aesthetic is
preserved because the weights, the spherical loss, and the cutout ensemble are all unchanged.

- **Files/seams:**
  - `src/pytti/Perceptor/__init__.py` — replace `from clip import clip` + `clip.load(model,
    jit=False)` in `init_clip` (lines 62–76) with a curated `PERCEPTOR_REGISTRY`:
    config key → `(open_clip model_name, pretrained tag | hf-hub id, tokenizer)`. Keep
    `_sanitize_for_config` so existing keys (`ViTB32`, `RN50x64`, …, schema lines 169–177 in
    `structured_config.py`) map to the same OpenAI weights; build the registry from a
    curated list instead of `clip.available_models()` (lines 20–23). Return
    (model, tokenizer, preprocess-stats) triples via
    `open_clip.create_model_and_transforms` + `open_clip.get_tokenizer`.
  - `_install_grad_fences` (lines 41–58) — open_clip native ViTs share OpenAI's
    `visual.conv1`/`visual.transformer` layout, so the existing MPS contiguous-grad fence
    (the single most important lines in the codebase for MPS; independently reproduced at
    130x) applies verbatim. **Add a fail-loud `else`** for unknown tower layouts — this is
    the guard that protects Slice 8's timm towers.
  - `src/pytti/Perceptor/Embedder.py` — `p.visual.input_resolution` (line 51) → normalize
    over open_clip's `visual.image_size` (int or tuple); replace the single global
    `normalize` (OpenAI mean/std, imported from `pytti/__init__.py` and applied at line 132)
    with **per-perceptor normalization** from each model's preprocess cfg — mandatory
    groundwork for Slice 8 (SigLIP uses mean=std=0.5; CLIP constants would silently skew
    gradients).
  - `src/pytti/Perceptor/Prompt.py` — `clip.tokenize(text)` at lines 175 and 241 →
    per-perceptor `tokenizer(text)` carried alongside each model. `spherical_dist_loss`
    (line 39) and the weight/stop machinery (line 311) untouched.
  - `pyproject.toml` — swap `clip @ git+...` for `open-clip-torch>=2.32` + `timm>=1.0.15`.
  - No changes to `cutouts/samplers.py` or `cutouts/augs.py`.
- **Config surface:** none changed; existing per-model boolean flags keep meaning the same
  weights.
- **MPS verification:** bit-comparability gate — fixed image/text battery embedded through
  old `clip` package vs new loader, assert allclose (atol 1e-6) *before* deleting the old
  dep; then the standard §4 same-seed A/B must be visually indistinguishable; fwd+bwd timing
  within 5% of baseline (fence confirmed active).
- **Effort:** ~2 days.

### Slice 6 — Batched sync-free cutout sampler

Measured on this machine: 14.2 ms → 8.4 ms fwd+bwd for cutn=40 @ 768×432 (1.7x on the
sampler stage), value diff vs slice+interpolate < 2/255, gradient parity 2.9e-11 — the
aesthetic is provably untouched (same bilinear resampling math).

- **Files/seams:**
  - `src/pytti/Perceptor/cutouts/samplers.py` — add `pytti_batched` beside `pytti_classic`
    (lines 16–103): sizes sampled on-device
    (`torch.empty(cutn, device=dev).normal_(0.8, 0.3).clamp(cut_size/max_size, 1.0) **
    cut_pow) * max_size`), offsets via `torch.rand(cutn, device=dev)`, theta `[cutn,2,3]`
    built entirely on-device, one `F.affine_grid` + one `F.grid_sample` on
    `input.expand(cutn, -1, -1, -1)` (expand, not repeat). **Absolutely no `int()`/
    `.item()` in the path** — the critical negative finding: each is a GPU sync, and a naive
    version measured 8x *slower* (27 ms). Return offsets/sizes as single `[cutn,2]` tensors
    (kills 80 tiny `.to(device)` transfers per call, samplers.py lines 93–96).
  - `src/pytti/Perceptor/Embedder.py` `make_cutouts` (lines 62–93) — dispatch on a new
    config field; augs and noise_fac already operate on the concatenated batch (lines
    97–102 of samplers.py), no change.
  - `src/pytti/config/structured_config.py` — new field.
  - `tests/` — parity test: grid_sample-vs-slice+interpolate max-abs-diff < 2/255, MPS-vs-CPU
    grad parity < 1e-6. **Pin in CI** so a torch upgrade regressing MPS grid_sampler backward
    fails loudly.
- **Config surface:** additive `cutout_sampler: batched | classic` (default `classic` until
  the A/B soak passes, then flip default; `classic` stays available).
- **Effort:** ~1 day.

### Slice 7 — Optimizer: schedule-free AdamW (opt-in → default after A/B)

NeurIPS 2024 oral, AlgoPerf 2024 self-tuning winner, pure-Python PyPI package, verified
converging on MPS locally. Fits pytti's fresh-optimizer-per-frame regime
(`reset_lr_each_frame` default true): no LR schedule question, and the eval-mode
Polyak-averaged iterate gives a principled smoothed frame — directly targeting inter-frame
shimmer, and extending EMA benefits to Limited Palette and RGB models which currently have
none (`EMAImage` decay=0.99 only wraps VQGAN).

- **Files/seams:**
  - `pyproject.toml` — add `schedulefree>=1.4.1`.
  - `src/pytti/ImageGuide.py` — replace the two bare `optim.Adam` constructions (lines 72
    and 137, `set_optim`) with a factory keyed by config; for `AdamWScheduleFree` pass
    `warmup_steps≈10` (frames are ~50-step bursts) and call `opt.train()` immediately after
    construction.
  - `src/pytti/workhorse.py` — wrap the frame-save/display decode in `opt.eval()`/`opt.train()`
    **in a single save path, not at call sites** — a missed `opt.eval()` silently emits the
    un-averaged image; this is optimizer state, treat it as such.
  - `src/pytti/image_models/ema.py` — unchanged for now; evaluate retiring `EMAImage` for
    VQGAN after the A/B (schedule-free averaging may make it redundant).
- **Config surface:** additive `optimizer: adam | adamw_sf` (default `adam` until a full
  animation A/B passes).
- **MPS verification:** convergence smoke (already verified: 60-step quadratic 0.33→3.4e-4);
  §4 A/B on all three golden scenes with the flicker metric — the explicit hypothesis is
  *lower* inter-frame shimmer at equal prompt adherence.
- **Effort:** ~1–2 days including the A/B soak.

### Slice 8 — Perceptor palette: FARE (robust gradients) + SigLIP2 (semantics)

Additive ensemble members via the Slice-5 registry. FARE adversarially-robust CLIP
(`chs20/fare2-clip` / `chs20/FARE4-ViT-B-32-laion2B-s34B-b79K`, OpenAI-init) is the one
2022–2026 research line targeting exactly what pytti needs — perceptually-aligned gradients
through the perceptor (CLIPAG WACV 2024; arXiv 2502.11725; arXiv 2505.23161). SigLIP2
(`timm/ViT-B-16-SigLIP2`, opt-in `ViT-SO400M-16-SigLIP2-256`) buys ~78–85% IN-1k semantics
vs ~63–68% for OpenAI B-scale; embeddings are still L2-normalized hypersphere vectors, so
`spherical_dist_loss` works unchanged.

- **Files/seams:** registry entries in `Perceptor/__init__.py`; a **new grad-fence branch**
  for timm-backed towers (`visual.trunk.patch_embed`) — the fail-loud else from Slice 5
  forces this to be conscious; per-perceptor normalize/tokenizer from Slice 5 make the rest
  free. New boolean flags in `structured_config.py` (`FARE2ViTB32`, `SigLIP2B16`,
  `SigLIP2SO400M`, `PECoreB16` opt-in).
- **New deps:** none beyond Slice 5 (weights from HF hub; pin open_clip version and
  **assert the resolved preprocess cfg at load** — open_clip issue #1068, cached SigLIP2
  loads can silently pick the wrong preprocessor).
- **MPS verification:** benchmark fwd+bwd per new architecture *before* enabling (the 100x
  pathology was layout-specific; timm towers take a different code path — may be absent,
  present elsewhere, or new). Memory calibration per §4: B-scale ~2–3 GB activations at
  cutn 40 fp32; SO400M ~8–12 GB → gate large towers behind cutn auto-scaling on <32 GB Macs.
  A/B: default tier alone vs default+FARE vs default+SigLIP2 — judged for structure
  cleanliness / high-frequency artifacting (FARE hypothesis) and compositional prompt
  landing (SigLIP2 hypothesis).
- **Effort:** ~2–3 days.

### Slice 9 — LlamaGen VQ tokenizer as a new image model

A modernized taming VQGAN (same conv encoder/decoder family, 8-dim L2-normalized codebook,
MIT, single-file weights on HF) — keeps the codebook-snapping evolution that defines the
medium, decodes at rFID ~1.03 @512 vs ~4–5 for taming f16. Legacy six checkpoints stay
byte-identical; this grows the look palette.

- **Files/seams:**
  - `src/pytti/vendor/llamagen/vq_model.py` — vendor ~350 lines (Encoder/Decoder/
    VectorQuantizer/ModelArgs), mirroring the existing `src/pytti/vendor/taming/` pattern
    (ruff-excluded, verbatim-diffable per pyproject line 70).
  - `src/pytti/image_models/llamagen.py` — `LlamaGenImage(EMAImage)` cloned from
    `VQGANImage`; z shape `(1, toksY, toksX, 8)`; extend module-level `vector_quantize` in
    `vqgan.py` with an `l2_norm` flag (F.normalize rows + codebook before the distance
    matrix; `replace_grad` STE unchanged); f=16 so toksX/toksY math is identical; weights
    via `hf_hub_download("FoundationVision/LlamaGen", "vq_ds16_c2i.pt", revision=pinned)`;
    ds8 variant as a finer-texture second look.
  - Wire-up: `config/model_names.py`, `image_model` choice in `structured_config.py`
    (line 59 validator gains `"LlamaGen"`), new `elif` branch in `workhorse.py` dispatch
    (lines 249–288 — the error message at 287 lists supported models, update it).
    `LatentLoss` is already generic via `get_preferred_loss()`.
- **Config surface:** additive `image_model: "LlamaGen"` (+ variant key). Existing values
  untouched.
- **MPS verification:** op set is exactly what the vendored taming code already runs fp32 on
  MPS (Conv2d, GroupNorm, softmax attention, interpolate); codebook distance matrix is
  *cheaper* (e_dim 8 vs 256). Still: benchmark backward-through-decode vs taming at the same
  canvas before merging, include F.normalize backward in the grad test,
  `@pytest.mark.download` encode→decode round-trip on CPU+MPS asserting grad reaches z.
- **Effort:** ~2–3 days.

### Slice 10 — MPS hardening: pathology registry, contiguity asserts, parity CI

Codifies what the 100x incident taught us; woven partly through earlier slices, finished here.

- **Files/seams:** `src/pytti/device.py` grows a documented MPS-pathology registry
  (documented, dated, linked to upstream issues): addcmul_/addcdiv_ silent no-op on
  non-contiguous outputs (exactly Adam's step ops — assert `image_rep` parameters
  `.is_contiguous()` before `optimizer.step()` in `ImageGuide.train`); random in-place ops
  silently no-op on non-contiguous tensors on macOS <15 (pytorch#165257 — samplers.py's
  `new_empty().uniform_()` noise path is contiguous, keep it that way); isfinite
  non-contiguous corruption (#183419); clamp (#167767) / where (#122916) wrongness;
  grid_sampler_3d half-precision divergence (#160541 — one more reason to stay fp32 around
  grid_sample); channels_last ResNet-CLIP backward break (already fenced in
  `memory_format_for`). The `debug_mps_parity` startup check from Slice 3 becomes a CI job.
- **Effort:** ~1 day.

**Total: roughly 2–3 weeks of focused slices.**

---

## 3. Compat contract

**Keeps working, byte/bit-identical:**
- The prompt DSL, scene syntax, and session semantics — no slice touches them.
- All existing config files: every schema change is additive with defaults matching current
  behavior; validators gain values, never lose them.
- All six VQGAN checkpoint names incl. the `sflckr` alias; mirrored files are sha256-matched
  to the originals; already-downloaded checkpoints in `model_artifacts_path` are reused.
- CLIP perceptor flags (`ViTB32`, …) select the identical OpenAI weights through the new
  loader, gated by an allclose test before the old dep is deleted.
- The Crowson cutout ensemble, kornia augs, spherical distance loss, EMAImage semantics —
  unchanged (batched sampler is opt-in until parity-proven, `classic` remains).

**Deprecation shims:**
- pip extras `threed` / `video` / `animation` keep existing as names: `video` becomes empty
  (RAFT is core torchvision), `threed`/`animation` resolve to `transformers` instead of the
  AdaBins/GMA forks. Old install commands keep succeeding.
- `save_every` / restore-frame math (`workhorse.py` line 407) unchanged.

**Honestly breaks (accepted, pre-user product — per repo policy, no techdebt kept for
unreleased code):**
- `import clip`, `import gma`, `import adabins` stop existing in the environment. Nothing
  in-tree uses them after Slices 1/4/5; out-of-tree notebooks that did will break.
- **Cross-version animation resume is not reproducible:** a session started on v2.0-old and
  resumed post-upgrade won't crash, but flow numerics (RAFT vs GMA), depth semantics
  (relative inverse vs AdaBins metric), and therefore 3D camera motion for the same
  `translate`/`rotate` expressions will differ. The `r/R/mu` parametric-eval values in
  `zoom_3d` change distribution. Finish in-flight animations before upgrading.
- AdaBins' metric-depth assumption is gone: any workflow that relied on absolute
  metric-meters depth values (none in-tree) has no equivalent.
- fp16 remains unsupported on MPS (unchanged; MPSGraph hard-crash).

---

## 4. Measurement plan

**Principle:** "better" is judged by eye AND numbers, never by benchmark tables alone. The
aesthetic is the product; every aesthetic-affecting slice ships with its A/B evidence.

**Golden-scene suite** (`tests/fixtures/`, built in Slice 3): three checked-in configs with
pinned seeds — (a) Limited Palette 2D animation, (b) VQGAN-wikiart 3D with a translate/rotate
camera move, (c) Video Source restyling — 10–30 frames each, small canvas for CI, full
canvas for release A/Bs.

**Per-slice A/B protocol** (`scripts/ab_render.py`): render each golden scene under old and
new code, same machine, same seed. Outputs:
- **Eye:** contact sheet (every Nth frame, old row vs new row) + side-by-side mp4,
  reviewed *blind* (variants unlabeled, order shuffled) before the reviewer sees numbers.
- **Numbers:** per-step loss trajectories overlaid; s/step and peak memory per
  `vram_usage_mode` bucket; **flicker score** = mean LPIPS between consecutive frames
  (the shimmer metric — Slices 4 and 7 must not raise it; Slice 7's hypothesis is that it
  drops); **drift score** = LPIPS between old-frame-k and new-frame-k (bit-comparable slices
  1, 2, 5, 6 must stay ~0; aesthetic slices report it, eye judges it); **prompt adherence** =
  CLIP-score against a *held-out judge encoder never used in the optimization ensemble*
  (SigLIP2 judges OpenAI-guided runs and vice versa — never grade with the model being
  optimized, it's trivially gamed).
- **Hard gates for identity-preserving slices:** open_clip-vs-clip embed allclose (atol 1e-6);
  mirrored checkpoint sha256 equality; batched-cutout value diff < 2/255 and MPS-vs-CPU grad
  parity < 1e-6; RAFT MPS-vs-CPU flow diff < 1e-3 px. These are CI-pinned so torch upgrades
  that regress MPS numerics fail loudly.

**Calibration buckets:** `vram_usage_mode` (vram_tools.py) is the existing per-subsystem
memory-bucket mechanism ("CLIP", "AdaBins", "GMA", "Depth Loss", …). It extends to new
models by (a) growing an MPS backend (`torch.mps.current_allocated_memory()`, Slice 3 — it
is CUDA-only today) and (b) new bucket keys per model: "RAFT", "DepthAnythingV2",
per-perceptor keys ("CLIP ViT-B/32", "FARE2 ViT-B/32", "SigLIP2 B/16", "SigLIP2 SO400M"),
"LlamaGen". Measured bucket sizes at cutn=40 fp32 feed a **cutn auto-scaling table** — large
towers (SO400M, any L/H-class) auto-reduce cutn on <32 GB Macs with a logged warning, never
an OOM crash. Every new perceptor/image model must land with its bucket measured on MPS
before its config flag ships enabled-capable.

---

## 5. Rejected ideas

From the research, with reasons:

- **bf16 autocast on MPS** — measured 1.04x for real accuracy loss (grad cosine 0.99903);
  Apple GPUs have no half-precision tensor-core advantage. (CUDA-only autocast flag may
  return later.)
- **fp16 anywhere** — hard-crashes MPSGraph (torch 2.13); standing project rule.
- **torch.compile (Inductor Metal)** — experimental, known multistage-reduction codegen bugs
  (pytorch#152155); revisit at torch 2.14+, after the batched sampler gives static shapes.
- **FlexAttention/SDPA work** — Metal FlexAttention wins are at 32k-token sparse regimes;
  CLIP's dense 50–257-token attention already favors existing SDPA.
- **Diffusion-prior / SDS guidance** — replaces the medium with one-shot diffusion
  aesthetics; contrary to the product. Categorical rejection.
- **LAION ViT-H/14 as default** — 4–6x ViT-B fp32 backprop cost at cutn 40; opt-in with
  auto-reduced cutn at most.
- **EVA-CLIP 8B/18B** — absurdly oversized for cutout backprop.
- **DFN2B L/14, MetaCLIP-2** — license-blocked for a distributable art tool (apple-amlr /
  CC-BY-NC-4.0).
- **Apple DepthPro** — research-only license (their issue #66), 1.0B params at fixed
  1536², and no actual Apple Silicon advantage in the torch pipeline.
- **Marigold depth** — slowest option and the flickeriest (per-frame affine-invariant
  normalization); wrong tool for a per-frame loop.
- **MoGe-2 / Metric3D v2** — good models, but git-only installs violate the
  no-fork-repo constraint.
- **MiDaS 3.1** — fine license, 2022 quality thoroughly superseded by DA V2.
- **Depth Anything 3** — third-party PyPI package of unknown provenance, xformers
  (CUDA-centric) dep; revisit when it lands in `transformers`.
- **ptlflow** — drags in Lightning, self-trained weights are CC BY-NC-SA, MPS unverified.
- **NeuFlow v2 / DPFlow / Optical-Flow-Matching** — no PyPI / research repos; benchmark EPE
  buys nothing for consistency-mask flow.
- **Lion optimizer** — needs 3–10x LR/decay retuning, batch-size sensitive; experiment-only.
- **Prodigy** — d-estimation warmup restarts from scratch every frame under
  `reset_lr_each_frame`; opt-in experiment at most.
- **Karras power-function EMA** — likely redundant once schedule-free averaging lands;
  re-evaluate only if frame smoothing is still a complaint after Slice 7.
- **ViT-VQGAN** — no official weights exist; unofficial repros of unknown quality.
- **Open-MAGVIT2** — best-in-class VQ rFID but pytorch-lightning fork install; only viable
  with heavy vendoring, not worth it now.
- **NVIDIA Cosmos tokenizer** — non-MIT NVIDIA license, causal convs unverified on MPS.
- **MoVQ (Kandinsky decoder)** — pulls diffusers, drifts the look photoreal rather than
  dreamlike.
- **DCT/JPEG/SIREN parameterizations** — 2020-era pixray looks, low ROI.

**Deferred, not rejected:**
- **TAESD latent image model** (~10 MB, MPS-proven) — a genuinely different smooth-painterly
  medium-feel; best *second* image-model addition after LlamaGen proves the seam.
- **SEA-RAFT vendoring** — the flow-accuracy escape hatch if Slice 1 ever visibly regresses.
- **Video Depth Anything Small** — the escalation path if EMA-normalized DA V2 still
  flickers in long animations.
- **PE-Core perceptors** — free to add via the Slice-5 registry; benchmark gradients on MPS
  first.
