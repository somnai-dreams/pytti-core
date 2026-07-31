# MLX (Metal) port plan

Goal: run the render step's hot loop on MLX, where fp16 actually pays.
Spike measured 2026-07-30 (M3 Max, mlx 0.32.0, torch 2.13.0, batch 40 ×
[3,224,224], fwd+bwd w.r.t. input, same-session best-of-N):

| tower | torch-MPS fp32 | MLX fp16+compile | speedup |
|---|---|---|---|
| ViT-B/32 | 207.5 ms | 92.5 ms | 2.24× |
| ViT-B/16 | 506.1 ms | 318.4 ms | 1.59× |
| tower total | 713.6 ms | 410.9 ms | **1.74×** |

- fp32 MLX misses the 1.4× bar (1.23–1.35×): **the port is only worth it in
  fp16.** fp16 grads were finite with norms matching fp32 to 4 s.f. (random
  weights; the real-weights gate is below).
- Interop: DLPack works both directions with torch 2.13 MPS. torch→MLX is a
  copy (safe, 0.86 ms); MLX→torch is a **zero-copy view** (0.31 ms — treat
  read-only or clone). ~2.3 ms per step total. Both runtimes must be
  synchronized at boundary crossings (DLPack does not sync pending work).
- Op audit (mlx 0.32): sdpa/layer_norm grads probed OK; upsample-nearest
  grad OK; straight-through = `bwd + stop_gradient(fwd - bwd)`; `grid_sample`
  absent — pure-ops gather recipe is differentiable (probed), official
  custom-Metal-kernel example exists if profiling demands it; pad modes
  reflect/circular must be composed; `precise=True` softmax accumulates fp32.
- MLX has no loss scaler; the historical norm for this exact workload
  (fp16 CLIP + fp32 image params + Adam) never needed one. bf16 native on M3.
- Spike scripts: /tmp/mlx-spike/ (throwaway).

## Phase M1 — hybrid: MLX towers inside the torch engine

Everything stays torch except tower forward+backward and the prompt-loss
reduction. Projected step at 512²/B32+B16/cutn40: 790 → ~575 ms.

New package `src/pytti/Perceptor/mlx_backend/`:

- `convert.py` — HF CLIP safetensors → MLX weights (mlx-examples pattern:
  drop position_ids, conv OIHW→OHWI transpose, **QuickGELU for the OpenAI
  tier**), cached at `~/.cache/pytti/mlx/<model>/`, fp16 params.
- `vit.py` — CLIP visual tower in mlx.nn: patch conv (no bias), cls+pos
  embed, pre-LN blocks with `mx.fast.scaled_dot_product_attention` +
  `mx.fast.layer_norm`, QuickGELU MLP, final LN + proj. `mx.compile`d.
- `bridge.py` — the boundary. One `torch.autograd.Function` whose forward
  takes (cutout batch [n,3,s,s] torch-MPS fp32, per-prompt text embeds,
  per-prompt × per-cutout weight vectors and stop values as no-grad
  constants) and returns per-prompt scalar losses. Internally: NHWC permute
  → DLPack → MLX `value_and_grad` over towers + spherical distance +
  weighted reduction (ONE fwd+bwd at forward time) → stash input-grads →
  losses back to torch. `backward` = stored grad × incoming cotangent
  scalars (exact: grads are linear in the cotangent of scalar outputs).
  NO tower recompute at backward — this is why the loss reduction must live
  inside MLX. Mask/weight vectors depend only on crop geometry (offsets/
  sizes/mask images), not on image params → computable in torch pre-tower
  with no_grad and passed as constants.
- Config: `perceptor_backend: "torch" | "mlx"` (schema + pytti-able
  annotation). MLX applies to the classic ViT tier (ViTB32/B16/L14/L14-336);
  RN/SigLIP2/FARE towers stay torch in M1 — a mixed ensemble is fine, each
  LoadedPerceptor is independent. Selecting mlx logs which towers actually
  moved. `mlx` is a darwin-only dependency (`sys_platform == 'darwin'`
  marker), imported lazily so CI/linux never touches it.

### Gates (in order, all against torch fp32 as reference, real weights)

1. **Embedding parity**: MLX fp16 tower embeds vs torch, cosine ≥ 0.999
   per-image on a 256-image batch.
2. **Input-grad parity**: cosine ≥ 0.995 through the full bridge loss (the
   torch fp16-autocast probe measured 0.998 — that is the expected class).
   If fp16 fails: `precise=True` softmax everywhere, then fp32 LN
   accumulation, then bf16, then fp32 — in that order, re-measuring speed.
3. **Prompt-loss parity**: bridge loss values vs torch `Prompt.forward` on
   fixed inputs (weights/stops/masks exercised), rel ≤ 1e-2 fp16 / 1e-5 fp32
   debug mode.
4. **A/B render**: golden-lp-still, torch vs mlx backend, judged against a
   same-step null pair + eyeball. it/s probe must show ≥1.3× end-to-end.

## Phase M2 — full MLX still engine (after M1 ships)

Port the rest of the still-image loop: PixelImage/RGB image reps, batched
cutouts (pure-ops grid_sample first, Metal kernel only if profiling says),
BatchedAugs, TV/HDR losses, Adam — whole-step `mx.compile` with
`mx.random.state` in the compiled state, no boundary crossings, fp16
end-to-end with fp32 image params. torch remains for model download/
conversion and the non-Mac/animation paths (VQGAN, depth, flow). Projected
step ~350–420 ms (≥1.9× vs today). Only worth starting once M1's parity
gates and A/B have held in real use.

## M2 spike results (2026-07-31 — measured, all gates passed)

- Pure-ops gather grid_sample: value parity 0.36/255 fp16, grad cosine
  1.000000; slower than torch's fused kernel standalone (+4.4 ms/step
  total) but irrelevant — inside the whole-step compile the entire
  non-tower chain measures 19.7 ms. No custom Metal kernel needed.
- Limited Palette decode in MLX: parity 1.5e-8, grads to all param
  groups, **3.2× faster than torch** (27.9 → 8.7 ms compiled).
- Whole-step mx.compile (decode→cutouts→augs→real fp16 towers→loss→
  Adam, RNG in state): nothing refused to compile, no retrace at steady
  state, 50 steps healthy. 512²/cutn40/2-towers: 399–411 ms;
  256²/cutn16/B32: **43.9–46.4 ms (~22 it/s)**.
- HONEST CAVEAT: at full settings M2 buys only ~5–10% over M1
  (453 → ~415 ms; both tower-bound — M1 already banked the tower win).
  The M2 payoff is drafts (1.4× over M1, 2.4× over torch) and deleting
  the DLPack boundary + sync discipline.
- Build inventory: docs/mlx-m2-seam-map.md (op equivalents, RNG moves,
  .bak round-trip contract, 7 slices ≈ 7–9 days). Lead rulings:
  adamw_sf fails loud under the whole-step path v1; animation/VQGAN
  keep the M1 bridge; gradient accumulation loops in-step; params/Adam
  fp32 with the fp16 cast at tower entry.

## Notes

- Two allocators share the GPU in M1: tune `mx.metal.set_cache_limit` if
  memory pressure shows; watch for pipeline bubbles at the 2-sync boundary.
- Contamination caveat: spike absolutes were measured with a background
  Metal app running; ratios were stable across rounds. Re-baseline on a
  quiet machine during M1 gate 4.
