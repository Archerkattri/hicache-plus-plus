"""Evaluate the FLUX head-to-head. For each arm: speedup vs its OWN uncached
vanilla (the fidelity a cache must hit), LPIPS + PSNR vs that vanilla, and an
absolute CLIP prompt-alignment score. Pairs images by index (same prompt+seed).

usage: python eval_hth.py  (pairs are configured below)
"""
import os, json
import warnings; warnings.filterwarnings("ignore")
import torch, numpy as np, lpips, open_clip
from PIL import Image

DEV = os.environ.get("EVAL_DEV", "cuda:0")
HTH = "/tmp/hth"
PROMPTS = [l.strip() for l in open("/tmp/Spectrum/prompts/DrawBench200.txt") if l.strip()]

lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_model = clip_model.to(DEV).eval()
clip_pp_n = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")[2]
clip_tok = open_clip.get_tokenizer("ViT-B-32")


def _t(img, sz=512):
    a = torch.from_numpy(np.array(img.resize((sz, sz)))).float().permute(2, 0, 1) / 127.5 - 1
    return a.unsqueeze(0).to(DEV)


def psnr(a, b):
    a = np.array(a).astype(np.float32); b = np.array(b).astype(np.float32)
    mse = ((a - b) ** 2).mean()
    return 100.0 if mse < 1e-9 else 20 * np.log10(255.0) - 10 * np.log10(mse)


@torch.no_grad()
def clip_score(img, prompt):
    im = clip_pp_n(img).unsqueeze(0).to(DEV)
    tx = clip_tok([prompt]).to(DEV)
    imf = clip_model.encode_image(im); txf = clip_model.encode_text(tx)
    imf = imf / imf.norm(dim=-1, keepdim=True); txf = txf / txf.norm(dim=-1, keepdim=True)
    return (imf * txf).sum(-1).item()


def evaluate(arm_dir, van_dir):
    st = json.load(open(f"{HTH}/{arm_dir}/_stats.json"))
    vst = json.load(open(f"{HTH}/{van_dir}/_stats.json"))
    n = len(st["rows"])
    lp, ps, cl, clv = [], [], [], []
    for i in range(n):
        ai = Image.open(f"{HTH}/{arm_dir}/{i:03d}.png").convert("RGB")
        vi = Image.open(f"{HTH}/{van_dir}/{i:03d}.png").convert("RGB")
        lp.append(lpips_fn(_t(ai), _t(vi)).item())
        ps.append(psnr(ai, vi))
        pr = PROMPTS[i]
        cl.append(clip_score(ai, pr)); clv.append(clip_score(vi, pr))
    speed = vst["mean_time_s"] / st["mean_time_s"]
    return {
        "arm": st["tag"], "time_s": round(st["mean_time_s"], 2),
        "speedup": round(speed, 2),
        "fwd": st["rows"][0].get("fwd"),
        "LPIPS": round(float(np.mean(lp)), 4),
        "PSNR": round(float(np.mean(ps)), 2),
        "CLIP": round(float(np.mean(cl)), 4),
        "CLIP_vanilla": round(float(np.mean(clv)), 4),
    }


if __name__ == "__main__":
    import sys
    # pairs: (arm_dir, its_own_vanilla_dir)
    pairs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
    # cross-env vanilla sanity (038 vs 034)
    out = []
    for arm, van in pairs:
        try:
            out.append(evaluate(arm, van))
        except Exception as e:
            out.append({"arm": arm, "error": str(e)})
    hdr = f"{'arm':<20}{'time_s':>8}{'speedup':>9}{'fwd':>5}{'LPIPS↓':>9}{'PSNR↑':>8}{'CLIP↑':>8}{'CLIP_van':>9}"
    print(hdr); print("-" * len(hdr))
    for r in out:
        if "error" in r:
            print(f"{r['arm']:<20} ERROR {r['error'][:50]}"); continue
        print(f"{r['arm']:<20}{r['time_s']:>8}{r['speedup']:>8}x{str(r['fwd']):>5}"
              f"{r['LPIPS']:>9}{r['PSNR']:>8}{r['CLIP']:>8}{r['CLIP_vanilla']:>9}")
    json.dump(out, open(f"{HTH}/_eval.json", "w"), indent=2)
