#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""PoC-A : compare EASU (FSR 1 de gamescope) et le MAKO Scaler adapté à l'interface d'EASU.

Pour chaque image de référence H (taille de sortie), on fabrique une image basse résolution L (H / facteur, moyenne
par zones, comme un jeu rendu plus petit et antialiasé), on l'agrandit à la taille de H avec chaque méthode, puis on
compare au H d'origine : PSNR (plus haut = plus fidèle) et SSIM sur la luminance (plus proche de 1 = structure mieux
gardée). Aucune netteté n'est appliquée : dans gamescope, le RCAS de Valve passe derrière les deux.

    python3 -I scripts/poc_a.py [--factor 1.5] [--out build/poc_a]
"""

import argparse
import os
import struct
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(HERE, "build")
HARNESS = os.path.join(BUILD, "harness")
SHADERS = {
    "easu": os.path.join(BUILD, "cs_easu.spv"),
    "easu_fp16": os.path.join(BUILD, "cs_easu_fp16.spv"),
    "mako": os.path.join(BUILD, "cs_mako_easu.spv"),
    "sgsr": os.path.join(BUILD, "cs_sgsr_easu.spv"),
    "lanczos3_ar": os.path.join(BUILD, "cs_lanczos3_easu.spv"),
    "ewa_sharp": os.path.join(BUILD, "cs_ewa_easu.spv"),
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def write_rgba(path, img):
    a = np.asarray(img.convert("RGBA"), dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", a.shape[1], a.shape[0]))
        f.write(a.tobytes())


def read_rgba(path):
    with open(path, "rb") as f:
        w, h = struct.unpack("<II", f.read(8))
        a = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w, 4)
    return Image.fromarray(a[:, :, :3].copy(), "RGB")


def luma(a):
    a = a.astype(np.float64)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def psnr(a, b):
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    """SSIM de luminance, fenêtre gaussienne 11x11 (sigma 1,5), version simple sans dépendance."""
    x, y = luma(np.asarray(a)), luma(np.asarray(b))
    k = np.exp(-0.5 * (np.arange(-5, 6) / 1.5) ** 2)
    k /= k.sum()

    def blur(z):
        z = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, z)
        return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, z)

    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s.mean())


def synthetic_images():
    """Cas difficiles pour un agrandisseur spatial : texte d'interface, diagonales, motif fin, dégradés."""
    w, h = 960, 600
    out = {}

    img = Image.new("RGB", (w, h), (24, 26, 32))
    d = ImageDraw.Draw(img)
    y = 10
    for size in (12, 14, 16, 20, 28, 40):
        f = ImageFont.truetype(FONT, size)
        d.text((12, y), f"PV 1234/1500  Runes 98765  Fiole x4  ({size}px)", font=f, fill=(230, 220, 190))
        y += size + 14
    d.rectangle((12, y + 10, 600, y + 30), fill=(150, 20, 20))
    d.rectangle((12, y + 40, 420, y + 56), fill=(30, 90, 170))
    out["hud_texte"] = img

    img = Image.new("RGB", (w, h), (200, 200, 200))
    d = ImageDraw.Draw(img)
    for i in range(0, 2 * w, 9):
        d.line((i, 0, i - h, h), fill=(40, 40, 40), width=1)
    for r in range(10, 300, 12):
        d.ellipse((w - 320 - r // 2, 300 - r // 2, w - 320 + r // 2, 300 + r // 2), outline=(150, 30, 30))
    out["diagonales_cercles"] = img

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w * 0.9)
    zone = (127.5 + 127.5 * np.cos(r2)).astype(np.uint8)
    out["mire_zone"] = Image.fromarray(np.stack([zone] * 3, axis=-1), "RGB")

    rng = np.random.default_rng(7)
    base = rng.random((h // 6, w // 6, 3)) * 255
    tex = Image.fromarray(base.astype(np.uint8), "RGB").resize((w, h), Image.BICUBIC)
    g = np.asarray(tex, np.float64)
    fine = rng.random((h, w, 1)) * 60 - 30
    grad = np.linspace(0, 1, w)[None, :, None] * np.array([40, 60, 20])[None, None, :]
    out["texture_feuillage"] = Image.fromarray(np.clip(g * 0.6 + fine + grad, 0, 255).astype(np.uint8), "RGB")
    return out


def photo_images(folder):
    out = {}
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                img = Image.open(os.path.join(folder, name)).convert("RGB")
                img = img.crop((0, 0, img.width - img.width % 6, img.height - img.height % 6))
                out["photo_" + os.path.splitext(name)[0]] = img
    return out


def skimage_photos():
    """Photos libres de droits livrées avec scikit-image (si installé) : NASA, CC0."""
    try:
        import skimage.data as data
    except ImportError:
        return {}
    out = {}
    for name in ("astronaut", "coffee", "rocket", "chelsea", "hubble_deep_field"):
        a = getattr(data, name)()
        h, w = a.shape[0] - a.shape[0] % 6, a.shape[1] - a.shape[1] % 6
        out["photo_" + name] = Image.fromarray(a[:h, :w, :3], "RGB")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=float, default=1.5, help="facteur d'agrandissement (1,5 = rendu à 66 %%)")
    ap.add_argument("--out", default=os.path.join(BUILD, "poc_a"))
    ap.add_argument("--photos", default=os.path.join(HERE, "tests", "images"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    refs = synthetic_images()
    refs.update(photo_images(args.photos))
    refs.update(skimage_photos())
    methods = ["bilineaire", "lanczos_pil"] + list(SHADERS)
    rows = []
    for name, ref in refs.items():
        lw, lh = round(ref.width / args.factor), round(ref.height / args.factor)
        low = ref.resize((lw, lh), Image.BOX)
        results = {
            "bilineaire": low.resize(ref.size, Image.BILINEAR),
            "lanczos_pil": low.resize(ref.size, Image.LANCZOS),
        }
        src = os.path.join(args.out, f"{name}_low.rgba")
        write_rgba(src, low)
        for key, spv in SHADERS.items():
            dst = os.path.join(args.out, f"{name}_{key}.rgba")
            subprocess.run([HARNESS, spv, src, dst, str(ref.width), str(ref.height), "3"], check=True,
                           stdout=subprocess.DEVNULL)
            results[key] = read_rgba(dst)
        for key in methods:
            rows.append((name, key, psnr(ref, results[key]), ssim(ref, results[key])))

        # Planche de comparaison : un recadrage 1:1 agrandi x2 (pixels nets) par méthode.
        cw, ch = min(240, ref.width), min(150, ref.height)
        cx, cy = (ref.width - cw) // 3, (ref.height - ch) // 3
        tiles = [("reference", ref)] + [(k, results[k]) for k in ("easu", "sgsr", "lanczos3_ar", "ewa_sharp")]
        sheet = Image.new("RGB", (len(tiles) * cw * 2, ch * 2 + 22), (0, 0, 0))
        dd = ImageDraw.Draw(sheet)
        f = ImageFont.truetype(FONT, 14)
        for i, (label, im) in enumerate(tiles):
            crop = im.crop((cx, cy, cx + cw, cy + ch)).resize((cw * 2, ch * 2), Image.NEAREST)
            sheet.paste(crop, (i * cw * 2, 22))
            dd.text((i * cw * 2 + 6, 3), label, font=f, fill=(255, 255, 255))
        sheet.save(os.path.join(args.out, f"{name}_planche.png"))

    print(f"facteur {args.factor} (rendu à {100 / args.factor:.0f} %), PSNR dB / SSIM luminance, plus haut = mieux\n")
    print(f"{'image':24} " + " ".join(f"{m:>17}" for m in methods))  # PSNR / SSIM
    for name in refs:
        cells = []
        for m in methods:
            p, s = next((r[2], r[3]) for r in rows if r[0] == name and r[1] == m)
            cells.append(f"{p:6.2f} / {s:.4f}")
        print(f"{name:24} " + " ".join(f"{c:>17}" for c in cells))
    print("\nmoyenne                  " + " ".join(
        f"{np.mean([r[2] for r in rows if r[1] == m]):6.2f} / {np.mean([r[3] for r in rows if r[1] == m]):.4f}"
        for m in methods))


if __name__ == "__main__":
    sys.exit(main())
