# M2 Seam Map — full-MLX still-image step (pytti-core @ core-v2 tip)

Scope verified against the code as of today. All paths relative to `/Users/max/Documents/Dev/pytti-core/src/pytti/` unless noted. MLX API claims verified against the project venv (`.venv`, mlx 0.32.0).

---

## 1+2. Full step trace with per-op MLX equivalents

The still-image step (`animation_mode=off`, `image_model` ∈ {Limited Palette, Unlimited Palette}, `perceptor_backend=mlx` today) is `DirectImageGuide.train` (ImageGuide.py:219-325), called from `run_steps` (149-182). Today's mlx path runs: torch decode → torch cutouts/augs/noise/normalize → **MLX towers+reduction via bridge** → torch autograd backward → torch Adam → torch clamps. M2 moves everything between "decode" and "clamp" inside one `mx.compile`d function.

### Stage A — harness (ImageGuide.py)

| Op | Loc | MLX |
|---|---|---|
| `optimizer.zero_grad` | :233, :319 | not needed (functional `mx.value_and_grad`) |
| interp ramp `t = i/interp_steps` | :243 | host scalar → compiled-fn input (already folded as `scales` in the M1 bridge) |
| microbatch loop (`gradient_accumulation_steps`) | :249-315 | loop **inside** the compiled step fn, grads summed before Adam; `/gas` scaling at :278, :304-306 |
| `mb_total.backward()` | :309 | `mx.value_and_grad` over the whole step loss |
| `optimizer.step()` | :317 | `mlx.optimizers.Adam.update` — see Adam note below |
| `image_rep.update()` (clamps) | :318 | functional clamp of params in the state tree after the optimizer update (order matters: step → clamp, matching :317-318) |
| loss records (0-dim tensors) | :239, :260, :264, :302, :321 | mx scalar outputs of the compiled fn; `float()` on mx arrays works at report time (`_report_losses` :335, early-stop :180) |

**Adam semantics** (`make_optimizer` :31-54; workhorse passes no extra kwargs → torch defaults betas=(0.9,0.999), eps=1e-8, wd=0, bias-corrected): `mlx.optimizers.Adam` **defaults `bias_correction=False` — must pass `True`**. Verified against mlx 0.32 source: with `bias_correction=True` the update is `param − (lr/(1−β₁ᵗ))·m / (√v·rsqrt(1−β₂ᵗ) + eps)` ≡ torch's `m̂/(√v̂+eps)` exactly (same eps placement). lr resolution: `params.learning_rate` default None → `image_rep.lr` (ImageGuide.py:96-98) = **0.02** for both PixelImage and RGBImage (differentiable_image.py:27; neither overrides — only VQGAN does, vqgan.py:187). Single param group everywhere. The palette-fit bare guide (pixel.py:454-460) builds `optim.Adam([palette, tensor], lr=0.1)` — `value` excluded — with `params=None` → `optimizer_name="adam"` (ImageGuide.py:102-104) and `update()` early-returns (:389-391). That fit is init-time-only (201 steps, no embedder/cutouts) — **recommend it stays torch**; its output params cross to MLX once.

`optimizer` config = `adamw_sf` (schedulefree, :43-51, eval-swap :197-214): no MLX equivalent. **Decision needed**: port schedule-free AdamW to MLX (~50-80 lines, z/x iterates + eval swap at every `_save_frame`) or fail loud (`mlx` whole-step requires `optimizer=adam`). Recommend fail-loud for the first M2 cut.

### Stage B — decode (image models)

**No EMA in the still path.** `EMAImage` (ema.py) is subclassed only by `VQGANImage` (vqgan.py:137); PixelImage/RGBImage inherit `decode_training_tensor == decode_tensor` (differentiable_image.py:30-34). EMA accum/biased/average buffers are out of M2 scope.

`PixelImage.decode_tensor` (pixel.py:310-352):

| Op | Loc | MLX |
|---|---|---|
| `sort_palette`: div/clamp/square/mul/sum | :287-290 | `mx.clip`, arithmetic |
| `argsort(dim=0).T` | :291 | `mx.argsort` |
| per-column gather `stack([palette[i][:, j] ...])` | :292-294 | `mx.take_along_axis` (verified present) |
| `palette_target` passthrough | :285-286 | state array |
| `break_tensor` floor/ceil/round/frac + `.long()` | :17-35, :322 | `mx.floor/ceil/round`, `.astype(mx.int32)` |
| `clamp(0,1) * (palette_size-1)` | :321 | `mx.clip` |
| `movedim(0,2)` | :325 | `mx.transpose` |
| `F.one_hot(argmax)` | :326 | `mx.argmax` + `(mx.arange(n) == idx[..., None])` |
| `softmax(dim=2)` | :328 | `mx.softmax` |
| int-array indexing `palette[value_rounds]` | :331, :342 | mx advanced integer indexing (gather) — differentiable w.r.t. `palette` |
| `F.interpolate(..., mode="nearest")` int scale | :333-339, :345-351 | `mlx.nn.Upsample(mode="nearest")` (present; plan probed grad OK) or reshape-broadcast repeat |
| `replace_grad(disc, 0.5·cont+0.5·disc)` | :352 | straight-through: `bwd + mx.stop_gradient(fwd − bwd)` (plan-confirmed recipe; note grad flows into **both** disc and cont via the 0.5 mix) |
| `update()` in-place clamps | :408-417 | functional `mx.clip` writes into param state (palette→[0,2], value→[0,1], tensor→[0,∞)) |

`RGBImage.decode_tensor` (rgb_image.py:30-33):

| Op | Loc | MLX |
|---|---|---|
| `F.interpolate` nearest (scale usually 1) | :32 | as above / identity |
| `clamp_with_grad` | :33; tensor_tools.py:86-108 | **recipe needed**: custom VJP `grad·(grad·(x−clamp(x)) ≥ 0)` via `mx.custom_function` (present in 0.32, compile-compatible). It is *not* plain clamp-grad nor plain pass-through — don't substitute `clamp_grad` (tensor_tools.py:111-112), it changes semantics |

### Stage C — losses

`Loss.forward` wrapper (LossAug/BaseLossClass.py:30-41): `parametric_eval` (host), `weight.sign()/abs()`, `torch.maximum(loss, stop)` + `replace_grad` → all mx arithmetic + the stop-gradient recipe (this is exactly `mlx_semantic_reduction`'s gating, bridge.py:176-181 — reuse the pattern). Weight/stop scalars are **per-step host inputs** (t-parametric, eval_tools.py:63-89).

| Loss | Ops | MLX |
|---|---|---|
| TVLoss (LossAug/TVLossClass.py:8-13) — active by default, `smoothing_weight: 0.02` (structured_config.py:134), attached at LossOrchestratorClass.py:145-146 | `F.pad(...,"replicate")`, diffs, `.mean([1,2,3])` | `mx.pad(mode="edge")` (**native — verified**), arithmetic, `mx.mean(axis=…)` |
| MSELoss (LossAug/MSELossClass.py:131-141) — init weight / direct image prompts / direct stabilization | `F.mse_loss`, mask multiply; lazy mask `TF.resize` :134-138 | `mx.mean(mx.square(a−b))`; mask resize is no-grad, once-per-shape → do torch-side at setup, cross as constant |
| HSVLoss (LossAug/HSVLossClass.py:7-12) — `get_preferred_loss` for both still models | kornia `rgb_to_hsv`, `cat([rgb, hsv[:,1:]])` | **recipe needed but small**: only S and V survive the `[:,1:]` slice — `v = max(r,g,b)`, `s = (v−min)/(v+eps)` (kornia eps=1e-8). The hue/argmax cascade is *not needed* |
| EdgeLoss (LossAug/EdgeLossClass.py:10-33) — only if `edge_stabilization_weight` set + init image | grayscale, two 3×3 Sobel `conv2d` pad=same | `mx.conv2d` (NHWC) + constant weights; grayscale = luma dot |
| DepthLoss — MiDaS/AdaBins torch model | — | **stays torch, fail loud** under mlx whole-step |
| PaletteLoss (pixel.py:49-76) | movedim/view/softmax, `mean/std(dim=0)`, `Xᵀ@X`, `diag(diagonal())`, `pow(-1)` | `mx.transpose/reshape/softmax`, `mx.mean`, `mx.std` (present), `mx.matmul`, `mx.diag`+`mx.diagonal` (present) |
| HdrLoss (pixel.py:127-143) | `sort_palette`, `linalg.vector_norm`, `mse_loss`; `comp` buffer init :117-124 | `mx.linalg.norm` (present) or sqrt-sum-square; `comp` is a constant, cross once |

### Stage D — cutouts (the noise_fac path lives here)

`Embedder.cutout_batches` (Perceptor/Embedder.py:108-150): pre-pad for non-clamp border modes (:137-141, `PADDING_MODES` :19-24) then one sampler call per unique `cut_size` (cache :142-149; B/32+B/16 share 224 → one batch). `noise_fac` (default 0.1, :54) is applied **inside the samplers**, not the Embedder: samplers.py:255-257 (batched), :100-102 (classic).

| Op | Loc | MLX |
|---|---|---|
| `F.pad` mirror→reflect, wrap→circular | Embedder.py:137-141 | **recipe**: composed flips / slice-concat (plan-confirmed); smear→replicate = `mx.pad(mode="edge")` native; black→constant native |
| size draw `normal_(0.8,0.3).clamp.pow.mul.floor` | samplers.py:218-225 | `mx.random.normal`, `mx.clip`, `**`, `mx.floor` |
| offset draws + clamp arithmetic | :226-241 | `mx.random.uniform`, `mx.minimum/clip/floor` |
| `_affine_crop_grid` (theta build + `F.affine_grid` + grid clamp) | :106-157 | pure arithmetic — port directly (`mx.stack`, broadcast); `affine_grid` is already explicit math here |
| `F.grid_sample` bilinear/border/align_corners=False | :244-250 | **recipe needed** (absent from MLX): pure-ops gather bilinear — floor/ceil index gathers (`mx.take_along_axis` / advanced indexing) + lerp; differentiable (plan-probed); official custom-Metal-kernel example is the profiling escape hatch. The sampler's grid is axis-aligned per-cutout scale+translate → separable (cheaper) |
| `BatchedAugs._warp_matrices` | augs.py:57-109 | `mx.eye`, `mx.random.uniform`, `mx.where`; `diag_embed` of `[n,3]` → `flip_vec[...,None] * mx.eye(3)`; cos/sin/stack/`@`→`mx.matmul` |
| base grid + projective divide + `grid_sample` | :149-158 | `mx.meshgrid` (present), arithmetic, same gather-bilinear recipe (projective → non-separable full 2-D gather) |
| color matrices + `einsum("nij,njhw->nihw")` | :111-140, :161-162 | `mx.einsum` (present) |
| erase mask (arange comparisons, `~erase`) | :164-189 | `mx.arange`, comparisons, `mx.logical_not`, multiply |
| noise: `uniform_(0,noise_fac)` + `randn_like` | samplers.py:255-257 | `mx.random.uniform(0, noise_fac, [cutn,1,1,1])` + `mx.random.normal(cutouts.shape)` |
| `perceptor.normalize` (torchvision, per-perceptor stats) | Embedder.py:171 / bridge.py:387-389 | `(x − mean)/std` constants (stats from `LoadedPerceptor.normalize`, Perceptor/__init__.py:193) |
| `named_rearrange`/`format_input` | tensor_tools.py:10-45 | identity for the still path (`("n","s","y","x")` end to end); port as transpose if kept general |

`cutout_sampler=classic` (samplers.py:16-103, kornia augs augs.py:8-21): host-synced, CPU RNG, kornia stack — **out of M2 scope, fail loud** (batched is the default).

### Stage E — towers + prompt reduction

Done in M1: `VisionTower.encode` (mlx_backend/vit.py:150-171, fp16, `mx.fast` sdpa/layer_norm) and `mlx_semantic_reduction` (mlx_backend/bridge.py:151-184 — spherical distance, sign/stop straight-through, exactly `Prompt.forward` Prompt.py:302-324). In M2 these become the middle of the whole-step fn instead of `_make_step`'s bridge-wrapped island; the fp32 cast at tower entry (vit.py:158) already gives fp32 input-grads.

**Masks**: geometric masks (Prompt.py:72-107) are pure arithmetic on offsets/sizes — but in M2 offsets/sizes are MLX arrays produced *inside* the step, so mask weights can no longer be torch-side constants (bridge.py:94-143). Geometric masks + `mask_all` must be reimplemented as ~5-line mx functions in-step. `mask_image` (Prompt.py:112-173) is a sequential, data-dependent host loop **with cross-step mutable state** (`err_tensor`, :125/:163-165) — not compilable; carry the M1 fail-loud (or accept a per-step host round-trip, breaking single-compile). `mask_semantic` and `LocationAwareMCIP` stay fail-loud (bridge.py:108-125 already does).

---

## 3. RNG inventory

Seeding: `torch.manual_seed(params.seed)` workhorse.py:188 (seed itself from `random.randint` :186). **M2 must add `mx.random.seed(params.seed)`.**

| Draw | Loc | Generator today | Distribution | M2 home |
|---|---|---|---|---|
| PixelImage init: `value/tensor.uniform_()`, `palette.uniform_(to=2)` (if random_initial_palette) | pixel.py:472-475 | torch device (MPS) | U[0,1] / U[0,2] | **stays torch** (init-time, params cross once) |
| RGBImage init `tensor.uniform_()` | rgb_image.py:64 | torch device | U[0,1] | stays torch |
| cutout sizes | samplers.py:218-225 | torch device | N(0.8,0.3) clipped, ^cut_pow | `mx.random.normal` — in compiled state |
| cutout offsets ×2 | :228-229 | torch device | U[0,1] | `mx.random.uniform` |
| BatchedAugs: 5 gate draws + flip/θ/tx/ty/px/py/ang/sat/area/log_r/y0/x0 (~17 draws of [n]-shapes) | augs.py:64, 72-76, 88, 94-95, 104, 118, 123, 139, 167-177, 187 | torch device | U[0,1] (+log-uniform ratio) | `mx.random.uniform` |
| noise_fac: facs + gaussian field | samplers.py:256-257 | torch device | U[0,noise_fac], N(0,1) | `mx.random.uniform` + `mx.random.normal` |
| palette fit (encode_image smart_encode) | pixel.py:445-462 | **none** (no embedder → no cutouts/noise; HSV/Hdr/Palette losses deterministic) | — | — |

All per-step draws repeat per microbatch and per unique `cut_size` group. Everything must move to `mx.random.*` with **`mx.random.state` in the compiled function's `inputs=`/`outputs=` state** (plan requirement), otherwise draws freeze into the trace.

**Divergence contract**: MLX uses a Threefry counter RNG — same seed will NOT reproduce torch-backend renders. Precedent already exists: `pytti_classic` draws on the CPU stream (samplers.py:45-75) while `pytti_batched` draws on the device stream, so `cutout_sampler` choice already changes renders per seed. **Confirmed nothing else consumes the torch RNG stream mid-step** in the still path: losses, decode, Adam, parametric eval, and mask evaluation are deterministic; perceptor models are `.eval()` with no dropout (Perceptor/__init__.py:163). The torch stream is consumed only at init, before the loop — moving step draws to MLX cannot skew any remaining torch consumer.

---

## 4. State / lifecycle map

**Crosses steps (must live in MLX compiled state):**
- Image params: `PixelImage.{value,tensor,palette}` (pixel.py:198-205), `RGBImage.tensor` (rgb_image.py:22-26) — fp32.
- Adam `m`/`v` + step count per param.
- `mx.random.state`.
- Per-step host→device inputs (not retrace triggers): parametric weight/stop scalars (t advances every step via `set_t`, ImageGuide.py:402-407), interp ramp. Retrace only on shape change (prompt count / cutn / resolution) — same contract as `_make_step` (bridge.py:187-192).
- `mask_image` `err_tensor` (Prompt.py:125) — the one non-portable cross-step state; fail loud.
- `loss_history` — becomes lazy mx outputs; `float()` only at `_report_losses`/early-stop (preserving the no-mid-step-sync rule, ImageGuide.py:234-239).

**Crosses the frame boundary (still mode — yes, still mode has frame boundaries):**
- `set_optim(None)` fires **every `steps_per_frame` even with `animation_mode=off`** — the gate at ImageGuide.py:408-411 checks only `pre_animation_steps`/`steps_per_frame`, then :424-425 rebuilds the optimizer when `reset_lr_each_frame` (default **true**). M2 equivalent: zero `m`/`v`, reset step counter in-state. Must be reproduced or Adam trajectories diverge from torch at step `pre_animation_steps`.
- `_save_frame` (ImageGuide.py:340-381): `optimizer_eval` (adamw_sf only), `decode_image` → PIL PNG, breath blend (:353-364, PIL-side), `torch.save(img.state_dict(), *.bak)` (:373-375). **Round-trip requirement**: MLX params → torch tensors → the module's `state_dict()` so `.bak` stays torch-serialized and restore-compatible (workhorse.py:391-393 `torch.load` + `load_state_dict`). PixelImage `.bak` contents: `value`, `tensor`, `palette`, `palette_target` buffer, plus submodule buffers `hdr_loss.comp`, `hdr_loss.weight`, `loss.weight`; RGBImage: `tensor`. `use_palette_target` is a plain bool, not serialized — reconstructed from config at workhorse.py:263-268 (existing behavior, no new work). Restore direction: torch load at setup → copy into MLX state once. Adam moments are never serialized today (restore resets them) — MLX matches for free.
- Animation warps (`zoom_2d/3d`, video, stabilization enable/disable, ImageGuide.py:427-499) — torch, out of M2.

**Proposed M2 torch boundary (concurring with the task's proposal, with precise edges):**
MLX owns `train()` end-to-end when ALL hold: `animation_mode=off`, image model ∈ {PixelImage, RGBImage}, `cutout_sampler=batched`, `optimizer=adam`, plain-text prompts with geometric/all masks, no depth/semantic-mask/image-prompt features. Torch owns: model download/conversion, text embedding (`LoadedPerceptor.embed_text`), prompt/mask parsing, init (`encode_image` incl. the 201-step palette fit, `encode_random`), save/restore/PNG/breath, and everything animated/VQGAN (which keeps the M1 bridge — see below). Steady-state boundary crossings per step: **zero** (host scalars in, lazy loss scalars out); params exit MLX only at `save_every`.

---

## 5. Work breakdown — ordered M2 slices

| # | Slice | Content | Parity test vs torch | Size |
|---|---|---|---|---|
| S1 | MLX image models | Pixel decode (sort_palette, one-hot, gathers, nearest-upsample, straight-through), RGB decode + `clamp_with_grad` custom VJP, `update()` clamps, PaletteLoss + HdrLoss | Copy identical param values torch→mx; decode values ≤1e-6 fp32; grads (`mx.grad` vs torch autograd) ≤1e-6; argsort index equality | 1–1.5d |
| S2 | Direct losses | TV, MSE(+mask), HSV (S/V-only recipe), Edge (optional), `Loss.forward` sign/stop wrapper | Per-loss value + input-grad parity on fixed inputs; weight/stop/negative-weight cases (mirror tests/test_mlx_bridge.py:59-142 style) | 0.5–1d |
| S3 | MLX cutout sampler | geometry math port, gather-bilinear grid_sample (border, ac=False), pad-mode composition (reflect/circular) | Inject fixed integer sizes/offsets (exact pattern of tests/test_cutout_sampler.py:59-95 `_reference_cutouts`) → value ≤2/255 vs torch, grad rel ≤1e-5; distribution sanity for the RNG half. Riskiest slice; Metal-kernel escape hatch if the gather profile disappoints | 1.5–2d |
| S4 | MLX BatchedAugs | warp/color/erase with injected parameters | Inject fixed matrices/params → parity vs torch `BatchedAugs`; port invariants from tests/test_batched_augs.py (identity, exact flip, gray-preservation, erase rectangle) | 1d |
| S5 | Optimizer + whole-step assembly | `Adam(bias_correction=True)`, whole-step `mx.compile` with {params, m/v, `mx.random.state`} state, per-step constant inputs, grad-accum in-step loop, frame-boundary moment reset | (a) Adam lockstep vs `torch.optim.Adam` with injected identical grads, N steps, exact-to-fp32; (b) full-step parity with ALL RNG factored out (inject geometry+aug params+noise): loss & param-delta fp32 ≤1e-5, fp16 towers rel ≤1e-2 (M1 gate class) | 1–1.5d |
| S6 | Engine integration | dispatch in `DirectImageGuide` (whole-step path replaces `mlx_semantic_loss` hook), `set_optim` semantics, MLX↔torch state_dict round-trip, loss records / early-stop / display, fail-loud gates (classic sampler, adamw_sf, image/semantic masks, depth, VQGAN, animation) | `.bak` round-trip test: save under mlx → `torch.load` + restore under torch → identical decode; restore-mid-run resume test | 1–1.5d |
| S7 | Gates | `scripts/ab_render.py --conf golden-lp-still` (+ an RGB golden) torch vs mlx, null-pair judged; it/s probe vs plan's ≥1.9× projection; quiet-machine re-baseline (plan note) | A/B + perf | 0.5d |

Total ≈ 7–9 days.

**M1 bridge — retired vs reused:**
- **Reused verbatim**: `convert.py` (all), `vit.py` (all), `mlx_semantic_reduction` (bridge.py:151-184), tower grouping + normalization-stats validation (bridge.py:293-312), `load_clip` gating (Perceptor/__init__.py:226-245), config plumbing (`perceptor_backend`, structured_config.py:232-234).
- **Evolved**: `_make_step` (bridge.py:187-209) → the semantic middle of the whole-step fn; `semantic_prompt_constants` (bridge.py:94-143) splits — parametric weight/stop eval stays host-side, geometric-mask weight computation moves in-step to MLX (offsets/sizes are MLX-side now), `PromptConstants` shrinks to embeds + scalars + mask kind.
- **Retired from the still path**: `_MLXBridgeFunction` (bridge.py:217-262), the DLPack crossing + two-sync discipline (bridge.py:63-69, 235-254), `MLXSemanticLoss.__call__`, the `mlx_semantic_loss` hook in train (ImageGuide.py:119-128, 266-278).
- **Not deleted**: the M1 bridge remains the mlx execution path for animation modes and any config the M2 gates reject (or those fall back to torch — decision below).

**Decisions to surface before starting:**
1. `adamw_sf` under mlx: fail loud (recommended first) vs port schedule-free.
2. Configs M2's whole-step can't take (image masks, semantic masks/image prompts, classic sampler, depth stabilization, VQGAN, animation): fail loud vs silent fallback to the M1 bridge. Recommend: animation/VQGAN → M1 bridge stays; everything else fail loud (M1 already does).
3. `gradient_accumulation_steps` >1: in-step loop (recommended) vs unsupported.
4. Precision layout: towers fp16, image params + Adam state + image/direct losses + samplers/augs fp32, cast at tower entry (vit.py:158 already does) — matches plan's "fp16 end-to-end with fp32 image params" with the cast point being the tower input.