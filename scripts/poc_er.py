#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Banc sur de vraies images d'Elden Ring, aux résolutions que le jeu propose, vers 1280x720 (l'image du jeu sur le
Steam Deck, qui n'a pas de mode 16:10).

Les captures 4K (data/er_4k/, hors dépôt) sont réduites par moyenne de zones : 1280x720 sert de référence, et
1152x648 / 1024x576 / 800x450 d'images rendues plus petites (un rendu idéalement antialiasé : un vrai rendu en
basse résolution est un peu plus crénelé). Chaque méthode agrandit vers 1280x720, avant RCAS.

    python3 -I scripts/poc_er.py [--images data/er_4k]
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsrcnnx  # noqa: E402
import poc_a  # noqa: E402
import poc_fsrcnnx  # noqa: E402

TARGET = (1280, 720)
SOURCES = [(1152, 648), (1024, 576), (800, 450)]


def upscale_all(low, models, work, tag):
    out = {
        "bilineaire": low.resize(TARGET, Image.BILINEAR),
        "lanczos_pil": low.resize(TARGET, Image.LANCZOS),
    }
    for key, spv in poc_fsrcnnx.GPU.items():
        out[key] = poc_fsrcnnx.gpu_upscale(spv, low, TARGET, f"{tag}_{key}", work)
    out["ewa_sharp"] = poc_fsrcnnx.gpu_upscale(poc_a.SHADERS["ewa_sharp"], low, TARGET, f"{tag}_ewa", work)
    for key, passes in models.items():
        out[key] = poc_fsrcnnx.fsrcnnx_upscale(passes, low, TARGET)  # x2 puis réduction vers 1280x720
    return out


def sheet(ref, out, keys, box, path):
    from PIL import ImageDraw, ImageFont
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    img = Image.new("RGB", (len(keys) * cw * 2, ch * 2 + 22))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(poc_a.FONT, 14)
    tiles = {"reference": ref, **out}
    for i, k in enumerate(keys):
        img.paste(tiles[k].crop(box).resize((cw * 2, ch * 2), Image.NEAREST), (i * cw * 2, 22))
        d.text((i * cw * 2 + 6, 3), k, font=f, fill=(255, 255, 255))
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(poc_a.HERE, "data", "er_4k"))
    args = ap.parse_args()
    work = os.path.join(poc_a.BUILD, "poc_er")
    os.makedirs(work, exist_ok=True)
    models = {k: fsrcnnx.parse(p) for k, p in poc_fsrcnnx.MODELS.items()}

    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    hi = {os.path.splitext(f)[0][-12:]: Image.open(os.path.join(args.images, f)).convert("RGB") for f in files}
    print(f"{len(hi)} captures 4K d'Elden Ring -> 1280x720 ; PSNR dB / SSIM, plus haut = mieux, avant RCAS\n")

    for size in SOURCES:
        factor = TARGET[0] / size[0]
        scores = {}
        for name, im in hi.items():
            ref = im.resize(TARGET, Image.BOX)
            low = im.resize(size, Image.BOX)
            out = upscale_all(low, models, work, f"{name}_{size[0]}")
            for k, v in out.items():
                scores.setdefault(k, []).append((poc_a.psnr(ref, v), poc_a.ssim(ref, v)))
            if size == (800, 450):
                # un recadrage au centre de chaque image (souvent végétation, architecture)
                box = (540, 300, 540 + 200, 300 + 120)
                sheet(ref, out, ["reference", "bilineaire", "easu", "lanczos3_ar", "fsrcnnx_8"], box,
                      os.path.join(work, f"{name}_800x450_planche.png"))
        print(f"{size[0]}x{size[1]} -> 1280x720 (x{factor:.2f}, {100 * size[0] * size[1] / (TARGET[0] * TARGET[1]):.0f} % des pixels)")
        easu = np.mean([s[0] for s in scores["easu"]])
        for k, v in sorted(scores.items(), key=lambda kv: -np.mean([s[0] for s in kv[1]])):
            p, s = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
            print(f"  {k:14} {p:6.2f} dB ({p - easu:+.2f} vs EASU)  SSIM {s:.4f}")
        print()


if __name__ == "__main__":
    main()
