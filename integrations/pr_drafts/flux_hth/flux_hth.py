"""Unified FLUX.1-dev head-to-head: vanilla / Spectrum (Chebyshev) / cache-dit DMD
/ cache-dit TaylorSeer. One arm per process. Identical pipeline + identical
generation call across arms -> only the forecaster patch differs.

Two-phase to fit a 32GB card with CLEAN denoising timing (no per-image offload):
  phase 1: text encoders on GPU, transformer on CPU -> encode all prompts -> embeds
  phase 2: text encoders off, transformer+VAE on GPU -> denoise from cached embeds
The transformer stays resident, so per-image time is the real denoising cost
(this is where a feature cache actually saves work), not PCIe transfer noise.

usage: CUDA_VISIBLE_DEVICES=1 python flux_hth.py <arm> <out_dir> <n_prompts> [rdt]
"""
import sys, os, json, time
import warnings; warnings.filterwarnings("ignore")
import torch
from diffusers import DiffusionPipeline

ARM, OUT, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
RDT = float(sys.argv[4]) if len(sys.argv) > 4 else 0.12
os.makedirs(OUT, exist_ok=True)
STEPS, SEED, GUID, WARMUP, RES = 50, 42, 3.5, 5, 1024
MODEL = "black-forest-labs/FLUX.1-dev"
DEV = "cuda:0"
prompts = [l.strip() for l in open("/tmp/Spectrum/prompts/DrawBench200.txt") if l.strip()][:N]

pipe = DiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.set_progress_bar_config(disable=True)

# ---- phase 1: encode every prompt with the text encoders on GPU ----
pipe.text_encoder.to(DEV); pipe.text_encoder_2.to(DEV)
embeds = []
with torch.no_grad():
    for p in prompts:
        pe, ppe, tid = pipe.encode_prompt(prompt=p, prompt_2=p, device=DEV,
                                          num_images_per_prompt=1, max_sequence_length=512)
        embeds.append((pe.cpu(), ppe.cpu()))
# Drop the text encoders entirely: embeds are cached, and leaving them on CPU
# would pin the pipe's _execution_device to CPU (latents created on the wrong
# device). With them gone, _execution_device resolves to the transformer (GPU).
pipe.text_encoder = None
pipe.text_encoder_2 = None
torch.cuda.empty_cache()

# ---- arm patch on the transformer ----
tag = ARM
if ARM == "spectrum":
    sys.path.insert(0, "/tmp/Spectrum/src")
    from pipelines.flux_forward import our_flux_forward
    from utils import set_method, set_w, set_lam, set_m
    set_method("spectrum"); set_w(0.5); set_lam(0.1); set_m(4)
    pipe.transformer.flex_w = False
    pipe.transformer.__class__.num_steps = STEPS
    pipe.transformer.__class__.forward = our_flux_forward
    pipe.transformer.cnt = 0
    pipe.transformer.num_consecutive_cached_steps = 0
    pipe.transformer.num_steps = STEPS
    pipe.transformer.warmup_steps = WARMUP
    pipe.transformer.window_size = 2
    pipe.transformer.curr_ws = 2
    pipe.transformer.flex_window = 0.75
elif ARM in ("dmd", "taylorseer"):
    import cache_dit
    calib = (cache_dit.DMDCalibratorConfig(dmd_history=6) if ARM == "dmd"
             else cache_dit.TaylorSeerCalibratorConfig(taylorseer_order=2))
    cache_dit.enable_cache(
        pipe,
        cache_config=cache_dit.DBCacheConfig(residual_diff_threshold=RDT,
                                             max_warmup_steps=WARMUP,
                                             num_inference_steps=STEPS),
        calibrator_config=calib)
    tag = f"{ARM}_rdt{RDT}"
elif ARM != "vanilla":
    raise SystemExit(f"unknown arm {ARM}")

# ---- phase 2: transformer + VAE resident, denoise from cached embeds ----
pipe.transformer.to(DEV); pipe.vae.to(DEV)

rows = []
for i, (pe, ppe) in enumerate(embeds):
    if ARM == "spectrum":
        pipe.transformer.actual_forward_counter = 0
    pe, ppe = pe.to(DEV), ppe.to(DEV)
    torch.cuda.synchronize(); t0 = time.time()
    img = pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe,
               num_inference_steps=STEPS, height=RES, width=RES,
               generator=torch.Generator().manual_seed(SEED),
               guidance_scale=GUID).images[0]
    torch.cuda.synchronize(); dt = time.time() - t0
    img.save(f"{OUT}/{i:03d}.png")
    fwd = getattr(pipe.transformer, "actual_forward_counter", None)
    rows.append({"i": i, "prompt": prompts[i][:80], "time_s": round(dt, 3), "fwd": fwd})
    print(f"[{tag}] {i:02d} {dt:6.2f}s fwd={fwd}  {prompts[i][:46]}", flush=True)

summary = None
if ARM in ("dmd", "taylorseer"):
    try:
        import cache_dit
        summary = cache_dit.strify(pipe)
    except Exception as e:
        summary = f"(summary failed: {e})"
mean_t = sum(r["time_s"] for r in rows) / len(rows)
json.dump({"arm": ARM, "tag": tag, "rdt": RDT if ARM in ("dmd", "taylorseer") else None,
           "steps": STEPS, "res": RES, "mean_time_s": round(mean_t, 3),
           "rows": rows, "cache_summary": summary},
          open(f"{OUT}/_stats.json", "w"), indent=2)
print(f"[{tag}] DONE mean={mean_t:.2f}s/img -> {OUT}", flush=True)
