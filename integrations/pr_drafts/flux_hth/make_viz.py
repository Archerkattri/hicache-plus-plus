"""Compose comparison panels from the head-to-head outputs.
  1) dmd_vs_vanilla.png  -> for DefTruth's "show w/ and w/o the DMD calibrator"
  2) hth_panel.png       -> vanilla | Spectrum | DMD | TaylorSeer (the head-to-head)
"""
import json
from PIL import Image, ImageDraw, ImageFont

HTH = "/tmp/hth"


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def grid(cols, idxs, out, thumb=360, pad=10, labelh=34):
    ncol, nrow = len(cols), len(idxs)
    W = ncol * thumb + (ncol + 1) * pad
    H = labelh + nrow * thumb + (nrow + 1) * pad
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    f = _font(19)
    for c, (label, _dr) in enumerate(cols):
        x = pad + c * (thumb + pad)
        d.text((x + 4, 8), label, fill=(15, 15, 15), font=f)
    for r, i in enumerate(idxs):
        for c, (_label, dr) in enumerate(cols):
            try:
                im = Image.open(f"{HTH}/{dr}/{i:03d}.png").convert("RGB").resize((thumb, thumb))
            except Exception:
                im = Image.new("RGB", (thumb, thumb), (220, 220, 220))
            canvas.paste(im, (pad + c * (thumb + pad), labelh + pad + r * (thumb + pad)))
    canvas.save(out)
    print("saved", out, canvas.size)


if __name__ == "__main__":
    ev = {r["arm"]: r for r in json.load(open(f"{HTH}/_eval.json"))} if __import__("os").path.exists(f"{HTH}/_eval.json") else {}

    def lab(name, key):
        r = ev.get(key)
        return f"{name}  {r['speedup']}x  L{r['LPIPS']}" if (r and "LPIPS" in r) else name

    # DefTruth: vanilla vs DMD calibrator at a faithful operating point (~2.45x)
    grid([("uncached  (50 steps)", "vanilla038"),
          (lab("DMD calibrator", "dmd_rdt0.2"), "dmd_0.20")],
         [0, 1, 2, 5], f"{HTH}/dmd_vs_vanilla.png")

    # cache-dit in-framework: DMD vs the existing TaylorSeer calibrator (same DBCache)
    grid([("uncached", "vanilla038"),
          (lab("TaylorSeer calibrator", "taylorseer_rdt0.45"), "ts_0.45"),
          (lab("DMD calibrator", "dmd_rdt0.45"), "dmd_0.45")],
         [0, 3, 6, 9], f"{HTH}/cachedit_dmd_vs_taylorseer.png")

    # head-to-head: vanilla | Spectrum | DMD | TaylorSeer (matched ~3.2-3.5x)
    grid([("uncached", "vanilla038"),
          (lab("Spectrum", "spectrum"), "spectrum"),
          (lab("DMD", "dmd_rdt0.45"), "dmd_0.45"),
          (lab("TaylorSeer", "taylorseer_rdt0.45"), "ts_0.45")],
         [0, 3, 6, 9], f"{HTH}/hth_panel.png")
