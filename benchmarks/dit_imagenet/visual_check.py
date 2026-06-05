#!/usr/bin/env python3
"""Sanity grid: generate the SAME 8 classes/noise under baseline vs each cache, save a PNG grid
per method so the cache output can be eyeballed before a long FID run. A working cache should
look like coherent ImageNet samples ~matching the baseline; a broken one looks like noise."""
import os
import sys

import torch
from torchvision.utils import save_image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_dit import CachedCFG, REPO  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "..", "DiT"))
from diffusion import create_diffusion          # noqa: E402
from diffusers.models import AutoencoderKL       # noqa: E402
from models import DiT_models                     # noqa: E402

torch.set_grad_enabled(False)
dev = "cuda:0"
steps = 250
ckpt = os.path.join(REPO, "..", "..", "data", "weights", "DiT", "DiT-XL-2-256x256.pt")

model = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000).to(dev).eval()
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
diffusion = create_diffusion(str(steps))

classes = [207, 360, 387, 974, 88, 979, 417, 279]
out = "/tmp/dit_vis"; os.makedirs(out, exist_ok=True)

for method, interval in [("none", 1), ("hermite", 3), ("dmd", 3), ("dmd", 5)]:
    g = torch.Generator(device=dev).manual_seed(0)
    y = torch.tensor(classes, device=dev)
    z = torch.randn(len(classes), 4, 32, 32, generator=g, device=dev)
    z = torch.cat([z, z], 0)
    y_in = torch.cat([y, torch.full_like(y, 1000)], 0)
    cached = CachedCFG(model, method, interval, steps, 4, 0.5, 2, 6)
    cached.reset()
    s = diffusion.p_sample_loop(cached, z.shape, z, clip_denoised=False,
                                model_kwargs=dict(y=y_in, cfg_scale=1.5), progress=False, device=dev)
    s = s.chunk(2, 0)[0]
    img = vae.decode(s / 0.18215).sample
    tag = "baseline" if method == "none" else f"{method}_i{interval}"
    save_image(img, f"{out}/{tag}.png", nrow=4, normalize=True, value_range=(-1, 1))
    print(f"[{tag}] {cached.compute_calls}/{steps} compute -> {out}/{tag}.png")
