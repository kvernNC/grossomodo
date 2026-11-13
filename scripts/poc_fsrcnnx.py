#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""FSRCNNX contre EASU (FSR 1 de gamescope) et les filtres classiques, en x2 et en x1,5, plus son coût théorique.

x2   : référence H, image basse résolution L = H / 2 (moyenne par zones), chaque méthode agrandit L en H.
x1,5 : L = H / 1,5 ; FSRCNNX fait x2 puis l'image est réduite à la taille de H (Lanczos), comme il le faudrait
       dans gamescope pour un rendu à 67 % ; les autres méthodes font x1,5 directement.

FSRCNNX n'agrandit que la luminance (Y, BT.709 sur les valeurs sRGB, comme mpv) ; la chrominance est agrandie en
Lanczos, puis on revient en RGB. Les shaders gamescope (EASU, Lanczos-3, SGSR, MAKO) passent par build/harness.

    python3 -I scripts/poc_fsrcnnx.py
"""

import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsrcnnx  # noqa: E402
import poc_a  # noqa: E402

HERE = poc_a.HERE
MODELS = {
    "fsrcnnx_8": os.path.join(HERE, "third_party", "fsrcnnx", "FSRCNNX_x2_8-0-4-1.glsl"),
    "fsrcnnx_16": os.path.join(HERE, "third_party", "fsrcnnx", "FSRCNNX_x2_16-0-4-1.glsl"),
}
GPU = {k: poc_a.SHADERS[k] for k in ("easu", "lanczos3_ar", "sgsr", "mako")}

# Steam Deck (APU Van Gogh, RDNA 2) : 8 CU x 64 x 2 opérations x 1,6 GHz = 1,64 TFLOPS en FP32 (x2 en FP16 groupé),
# LPDDR5 88 Go/s. Une multiplication-addition = 2 opérations.
DECK_MAC_FP32 = 1.64e12 / 2
DECK_MAC_FP16 = 2 * DECK_MAC_FP32
DECK_BW = 88e9


def to_ycbcr(a):
    a = a.astype(np.float64) / 255.0
    y = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    cb = (a[..., 2] - y) / 1.8556
    cr = (a[..., 0] - y) / 1.5748
    return y, cb, cr


def to_rgb(y, cb, cr):
    r = y + 1.5748 * cr
    b = y + 1.8556 * cb
    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return Image.fromarray((np.clip(np.stack([r, g, b], -1), 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


def resize_plane(p, size):
    img = Image.fromarray(p.astype(np.float32), "F")
    return np.asarray(img.resize(size, Image.LANCZOS), dtype=np.float64)


def fsrcnnx_upscale(passes, low, size):
    """x2 sur la luminance avec FSRCNNX, chrominance en Lanczos, puis réduction à `size` si besoin."""
    y, cb, cr = to_ycbcr(np.asarray(low))
    y2 = fsrcnnx.run(passes, y)
    big = (low.width * 2, low.height * 2)
    out = to_rgb(y2, resize_plane(cb, big), resize_plane(cr, big))
    return out if out.size == size else out.resize(size, Image.LANCZOS)


def gpu_upscale(spv, low, size, tag, work):
    src, dst = os.path.join(work, f"{tag}_low.rgba"), os.path.join(work, f"{tag}_out.rgba")
    poc_a.write_rgba(src, low)
    subprocess.run([poc_a.HARNESS, spv, src, dst, str(size[0]), str(size[1]), "1"], check=True,
                   stdout=subprocess.DEVNULL)
    return poc_a.read_rgba(dst)


def evaluate(refs, factor, models, work):
    methods = ["bilineaire", "lanczos_pil"] + list(GPU) + list(models)
    scores = {m: [] for m in methods}
    for name, ref in refs.items():
        low = ref.resize((round(ref.width / factor), round(ref.height / factor)), Image.BOX)
        out = {
            "bilineaire": low.resize(ref.size, Image.BILINEAR),
            "lanczos_pil": low.resize(ref.size, Image.LANCZOS),
        }
        for key, spv in GPU.items():
            out[key] = gpu_upscale(spv, low, ref.size, f"{name}_{key}", work)
        for key, passes in models.items():
            out[key] = fsrcnnx_upscale(passes, low, ref.size)
        for m in methods:
            scores[m].append((poc_a.psnr(ref, out[m]), poc_a.ssim(ref, out[m])))
        if name in ("photo_astronaut", "hud_texte", "photo_deck_photo"):
            save_sheet(ref, out, os.path.join(work, f"x{factor:g}_{name}_planche.png"))
    return methods, scores


def save_sheet(ref, out, path):
    from PIL import ImageDraw, ImageFont
    keys = ["reference", "easu", "lanczos3_ar", "fsrcnnx_8", "fsrcnnx_16"]
    imgs = {"reference": ref, **out}
    cw, ch = min(200, ref.width), min(140, ref.height)
    cx, cy = (ref.width - cw) // 3, (ref.height - ch) // 3
    sheet = Image.new("RGB", (len(keys) * cw * 2, ch * 2 + 22))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(poc_a.FONT, 14)
    for i, k in enumerate(keys):
        sheet.paste(imgs[k].crop((cx, cy, cx + cw, cy + ch)).resize((cw * 2, ch * 2), Image.NEAREST), (i * cw * 2, 22))
        d.text((i * cw * 2 + 6, 3), k, font=f, fill=(255, 255, 255))
    sheet.save(path)


def print_scores(title, methods, scores):
    print(title)
    print(f"  {'méthode':14} {'PSNR moyen':>11} {'SSIM moyen':>11}")
    for m in methods:
        p = np.mean([s[0] for s in scores[m]])
        s = np.mean([s[1] for s in scores[m]])
        print(f"  {m:14} {p:9.2f} dB {s:11.4f}")
    print()


def print_cost(models):
    print("Coût théorique (Steam Deck : 0,82 TMAC/s en FP32, 1,64 en FP16, 88 Go/s ; efficacité réelle 30-50 %)")
    scenarios = [("x2  : 640x400 -> 1280x800", 640 * 400), ("x1,5 via x2 : 853x533 -> 1706x1066 -> 1280x800", 853 * 533)]
    for key, passes in models.items():
        macs, fetches, writes = fsrcnnx.cost(passes)
        n = sum(1 for p in passes if not p.aggregate)
        print(f"  {key} : {n} passes + agrégation, {macs} mult.-add. et {fetches} lectures par pixel basse résolution")
        for label, px in scenarios:
            gmac = macs * px / 1e9
            # écritures RGBA16F (8 o) par passe et pixel + lectures (voisins en cache : ~2 textures lues par pixel)
            mb = (writes / 2 + 2 * 8 * n) * px / 1e6
            lo32, hi32 = gmac * 1e9 / (DECK_MAC_FP32 * 0.5) * 1e3, gmac * 1e9 / (DECK_MAC_FP32 * 0.3) * 1e3
            lo16, hi16 = gmac * 1e9 / (DECK_MAC_FP16 * 0.5) * 1e3, gmac * 1e9 / (DECK_MAC_FP16 * 0.3) * 1e3
            bw = mb * 1e6 / (DECK_BW * 0.7) * 1e3
            print(f"    {label} : {gmac:.2f} G mult.-add., ~{mb:.0f} Mo de trafic -> "
                  f"FP32 {lo32:.1f}-{hi32:.1f} ms, FP16 {lo16:.1f}-{hi16:.1f} ms (mémoire >= {bw:.1f} ms)")
    # EASU : 12 texels lus (3 textureGather) et ~150 opérations par pixel de sortie (ordre de grandeur, ffx_fsr1.h).
    easu_ops = 150 * 1280 * 800
    print(f"  easu (référence) : ~150 opérations et 12 lectures par pixel de sortie -> "
          f"{easu_ops / 1e9:.2f} G op. à 1280x800, ~{easu_ops / (1.64e12 * 0.4) * 1e3:.2f} ms")


def main():
    work = os.path.join(poc_a.BUILD, "poc_fsrcnnx")
    os.makedirs(work, exist_ok=True)
    models = {k: fsrcnnx.parse(p) for k, p in MODELS.items()}
    refs = poc_a.synthetic_images()
    refs.update(poc_a.photo_images(os.path.join(HERE, "tests", "images")))
    refs.update(poc_a.skimage_photos())
    # x2 et x1,5 exigent des tailles divisibles : on recadre à un multiple de 12.
    refs = {k: v.crop((0, 0, v.width - v.width % 12, v.height - v.height % 12)) for k, v in refs.items()}
    for factor in (2.0, 1.5):
        methods, scores = evaluate(refs, factor, models, work)
        print_scores(f"x{factor:g} (rendu à {100 / factor:.0f} %), {len(refs)} images, plus haut = mieux", methods, scores)
    print_cost(models)


if __name__ == "__main__":
    main()
