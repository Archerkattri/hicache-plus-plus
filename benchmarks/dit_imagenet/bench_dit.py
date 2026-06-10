#!/usr/bin/env python3
"""Literature-standard feature-cache benchmark: DiT-XL/2 on ImageNet-256 -> FID vs latency.

This is the canonical setting the feature-cache line (TaylorSeer, HiCache, FoCa) is evaluated in.
We wrap DiT's classifier-free-guidance forward with a velocity cache and, on skipped denoise
steps, *forecast* the model output instead of running the transformer. One cell = one
(method, interval): we generate ``--n`` class-conditional samples, accumulate Inception-V3
activations, and save (mu, sigma, latency, #compute-calls) so FID can be computed offline
against both the ImageNet reference and the uncached baseline.

  method ∈ {none, taylor, hermite, dmd}   (none = uncached baseline)
    taylor  — TaylorSeer monomial forecast        hermite — HiCache scaled-Hermite forecast
    dmd     — HiCache++ exponential (DMD) forecast

Run one cell per invocation (resumable, parallel-friendly); a driver sweeps the grid.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)                                   # hicache_pp
sys.path.insert(0, os.path.join(REPO, "..", "DiT"))       # DiT model code

from hicache_pp.hermite import (hicache_init, hicache_decide, hicache_update_derivatives,  # noqa: E402
                                hicache_forecast, scaled_hermite)
from hicache_pp.dmd import dmd_update_snapshots, dmd_forecast_state, auto_forecast_state                        # noqa: E402


def taylor_forecast(state):
    """TaylorSeer monomial forecast from the same backward finite differences HiCache uses:
    F_hat = F_t + sum_i (Delta^i F_t / i!) * k^i  (the un-scaled, un-damped Hermite limit)."""
    deriv = state["derivatives"]
    k = state["step"] - state["activated_steps"][-1]
    out = deriv[0].clone()
    o = 1
    while o in deriv:
        # Sign-convention fix (2026-06-10): evaluate at +k, not -k. The finite
        # differences are forward slopes, so forecasting k steps PAST the newest
        # anchor must use distance +k (upstream TaylorSeer's convention); (-k)**o
        # flips every odd-order term and extrapolates backwards. Any taylor cell
        # generated before this fix is an anti-TaylorSeer measurement.
        out = out + deriv[o] / math.factorial(o) * float(k) ** o
        o += 1
    return out


def atomic_savez(path, **arrays):
    """np.savez through a temp file + os.replace: a kill mid-write can never leave
    a truncated npz at the destination path."""
    tmp = path + ".tmp.npz"                     # ends in .npz so np.savez appends nothing
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


class CellCheckpoint:
    """Per-1k-image checkpointing of the FID stat accumulation, so a killed cell
    resumes from the last checkpoint instead of restarting from image 0.

    The partial state file ``<cell>.partial.npz`` (next to the final npz) holds the
    raw Inception activations accumulated so far plus the latency counter and a run
    fingerprint; it is written atomically every ``every`` images and removed when the
    final npz lands. The final npz format is UNCHANGED (same keys as before this
    class existed), and a resumed run produces bit-identical final stats to an
    uninterrupted one (the caller must fast-forward its RNG past completed batches;
    the per-step ancestral noise is already re-seeded per batch and needs nothing).
    """

    def __init__(self, out_npz, fingerprint, every=1000):
        self.out_npz = out_npz
        base = out_npz[:-4] if out_npz.endswith(".npz") else out_npz
        self.partial_path = base + ".partial.npz"
        self.fingerprint = str(fingerprint)
        self.every = int(every)
        self.feats = []                          # list of [b, 2048] activation arrays
        self.done = 0                            # images fully accumulated
        self.t_gen = 0.0                         # summed p_sample_loop wall-clock (s)
        self._last_saved = 0

    def resume(self):
        """Load the partial state if present. Returns ``(done, t_gen)`` (0, 0.0 on a
        fresh start). Refuses a partial written under different run parameters."""
        if os.path.exists(self.partial_path):
            with np.load(self.partial_path, allow_pickle=False) as z:
                fp = str(z["fingerprint"])
                if fp != self.fingerprint:
                    raise RuntimeError(
                        f"partial state {self.partial_path} was written by a different "
                        f"run configuration ({fp!r} != {self.fingerprint!r}); delete it "
                        f"or rerun with the original parameters")
                self.feats = [np.array(z["act"])]
                self.done = int(z["done"])
                self.t_gen = float(z["t_gen"])
            self._last_saved = self.done
        return self.done, self.t_gen

    def add(self, feats_batch, b, dt):
        """Accumulate one batch (``b`` images, ``dt`` seconds of generation) and
        checkpoint when a ``every``-image boundary is crossed."""
        self.feats.append(feats_batch)
        self.done += int(b)
        self.t_gen += float(dt)
        if self.done // self.every > self._last_saved // self.every:
            self.save_partial()

    def save_partial(self):
        atomic_savez(self.partial_path, act=np.concatenate(self.feats, 0),
                     done=self.done, t_gen=self.t_gen, fingerprint=self.fingerprint)
        self._last_saved = self.done

    def finalize(self, n, **meta):
        """Compute (mu, sigma) over the first ``n`` activations, atomically write the
        final npz (format unchanged) and remove the partial file."""
        act = np.concatenate(self.feats, 0)[:n]
        mu, sigma = act.mean(0), np.cov(act, rowvar=False)
        atomic_savez(self.out_npz, mu=mu, sigma=sigma, n=n, **meta)
        if os.path.exists(self.partial_path):
            os.remove(self.partial_path)
        return mu, sigma


class CachedCFG:
    """Wraps ``model.forward_with_cfg`` with a HiCache-style compute/forecast schedule."""

    def __init__(self, model, method, interval, num_steps, first_enhance, sigma, order, history):
        self.model, self.method = model, method
        self.p = dict(interval=interval, num_steps=num_steps, first_enhance=first_enhance,
                      sigma=sigma, order=order, history=history)
        self.reset()

    def reset(self):
        # interval is ignored for method="none"; clamp to a valid value so init never raises.
        self.state = hicache_init(num_steps=self.p["num_steps"], interval=max(1, self.p["interval"]),
                                  max_order=self.p["order"], first_enhance=self.p["first_enhance"],
                                  sigma=self.p["sigma"],
                                  backend={"dmd": "dmd", "auto": "auto"}.get(self.method, "hermite"))
        self.state["history"] = self.p["history"]
        self.state["step"] = 0
        self.compute_calls = 0

    def __call__(self, x, t, **kw):
        if self.method == "none":
            self.compute_calls += 1
            return self.model.forward_with_cfg(x, t, **kw)
        decision = hicache_decide(self.state)                      # also logs compute steps
        if decision == "forecast":
            if self.method in ("dmd", "auto"):
                out = (auto_forecast_state(self.state)
                       if self.state.get("backend") == "auto"
                       else dmd_forecast_state(self.state))
            elif self.method == "taylor":
                out = taylor_forecast(self.state)
            else:
                out = hicache_forecast(self.state)
            self.state["step"] += 1
            return out
        out = self.model.forward_with_cfg(x, t, **kw)
        self.compute_calls += 1
        hicache_update_derivatives(self.state, out.detach())
        if self.method in ("dmd", "auto"):
            dmd_update_snapshots(self.state, out.detach(), self.state["history"])
        self.state["step"] += 1
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["none", "taylor", "hermite", "dmd", "auto"])
    ap.add_argument("--interval", type=int, default=3)
    ap.add_argument("--n", type=int, default=10000, help="number of samples")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--cfg-scale", type=float, default=1.5)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--history", type=int, default=6)
    ap.add_argument("--first-enhance", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", default=os.path.join(REPO, "..", "..", "data", "weights", "DiT",
                                                   "DiT-XL-2-256x256.pt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    dev = f"cuda:{args.gpu}"
    cell = f"{args.method}_i{args.interval}" if args.method != "none" else "baseline"
    os.makedirs(args.out, exist_ok=True)
    out_npz = os.path.join(args.out, f"{cell}.npz")
    if os.path.exists(out_npz):
        print(f"[skip] {cell} exists"); return

    from diffusion import create_diffusion
    from diffusers.models import AutoencoderKL
    from models import DiT_models
    from pytorch_fid.inception import InceptionV3

    model = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False))
    model.eval()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
    diffusion = create_diffusion(str(args.steps))
    cached = CachedCFG(model, args.method, args.interval, args.steps,
                       args.first_enhance, args.sigma, args.order, args.history)

    fingerprint = (f"{args.method}|i{args.interval}|n{args.n}|b{args.batch}|s{args.steps}|"
                   f"cfg{args.cfg_scale}|o{args.order}|sg{args.sigma}|h{args.history}|"
                   f"fe{args.first_enhance}|seed{args.seed}")
    ckpt = CellCheckpoint(out_npz, fingerprint)
    done0, _ = ckpt.resume()

    g = torch.Generator(device=dev).manual_seed(args.seed)
    if done0:
        print(f"[resume] {cell}: {done0}/{args.n} images from {ckpt.partial_path}", flush=True)
        # Fast-forward the class/latent generator past the completed batches by
        # replaying (and discarding) their draws — deterministic, so a resumed run
        # is bit-identical to an uninterrupted one. The per-step ancestral noise is
        # already re-seeded per batch (below) and needs no fast-forward.
        d = 0
        while d < done0:
            b = min(args.batch, args.n - d)
            torch.randint(0, 1000, (b,), generator=g, device=dev)
            torch.randn(b, 4, 32, 32, generator=g, device=dev)
            d += b

    while ckpt.done < args.n:
        done = ckpt.done
        b = min(args.batch, args.n - done)
        y = torch.randint(0, 1000, (b,), generator=g, device=dev)
        z = torch.randn(b, 4, 32, 32, generator=g, device=dev)
        z = torch.cat([z, z], 0)
        y_in = torch.cat([y, torch.full_like(y, 1000)], 0)
        # Pair the per-step ancestral noise across cells. p_sample_loop's internal
        # th.randn_like draws from the GLOBAL generator, so seed it identically per
        # batch (same across every cell, since batch size and `done` match). Without
        # this each cell is a separate process with independent per-step noise, so
        # even a lossless cache scores a large spurious FID-vs-baseline — a pure
        # measurement floor, not cache drift. (DiT eval has no other global-RNG draws,
        # so this fully pairs the trajectories: a zero-forecast cell -> FID ~ 0.)
        torch.manual_seed(args.seed + done)
        torch.cuda.manual_seed_all(args.seed + done)
        cached.reset()
        torch.cuda.synchronize(dev); t0 = time.time()
        s = diffusion.p_sample_loop(cached, z.shape, z, clip_denoised=False,
                                    model_kwargs=dict(y=y_in, cfg_scale=args.cfg_scale),
                                    progress=False, device=dev)
        torch.cuda.synchronize(dev); dt = time.time() - t0
        s = s.chunk(2, 0)[0]
        img = vae.decode(s / 0.18215).sample                       # [-1,1]
        img = ((img + 1) * 0.5).clamp(0, 1)
        ckpt.add(inception(img)[0].squeeze(-1).squeeze(-1).cpu().numpy(), b, dt)
        done = ckpt.done
        if done % (args.batch * 8) < args.batch:
            print(f"  {cell}: {done}/{args.n}  ({ckpt.t_gen/done*1000:.0f} ms/img, "
                  f"{cached.compute_calls}/{args.steps} compute)", flush=True)

    if cached.compute_calls == 0:
        # Resumed with every image already accumulated (killed between the last
        # checkpoint and finalize): no batch ran, so recover the compute-call count
        # by replaying the deterministic schedule without touching the model.
        cached.reset()
        if args.method == "none":
            cached.compute_calls = args.steps
        else:
            for _ in range(args.steps):
                if hicache_decide(cached.state) == "full":
                    cached.compute_calls += 1
                cached.state["step"] += 1

    t_gen = ckpt.t_gen
    ckpt.finalize(args.n, latency_ms=t_gen / args.n * 1000,
                  compute_calls=cached.compute_calls, steps=args.steps, method=args.method,
                  interval=args.interval, cfg=args.cfg_scale)
    print(f"[done] {cell}: {t_gen/args.n*1000:.1f} ms/img, "
          f"{cached.compute_calls}/{args.steps} compute calls -> {out_npz}")


if __name__ == "__main__":
    main()
