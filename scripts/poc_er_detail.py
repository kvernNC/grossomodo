#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mesure du détail fin sur les captures d'Elden Ring (800x450 -> 1280x720, et 1024x576), avant RCAS.

- énergie du détail : écart-type de la luminance après passe-haut (image - flou gaussien sigma 1), rapporté à celui
  de la référence (100 % = autant de détail que l'original, moins = plus doux, plus = sur-accentué) ;
- PSNR du détail : fidélité de cette seule bande de hautes fréquences (le bon détail au bon endroit).

    python3 -I scripts/poc_er_detail.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsrcnnx  # noqa: E402
import poc_a  # noqa: E402
import poc_er  # noqa: E402
import poc_fsrcnnx  # noqa: E402


def detail(img):
    y = poc_a.luma(np.asarray(img))
    blur = poc_a.luma(np.asarray(img.filter(ImageFilter.GaussianBlur(1.0))))
    return y - blur


def main():
    work = os.path.join(poc_a.BUILD, "poc_er")
    models = {"fsrcnnx_8": fsrcnnx.parse(poc_fsrcnnx.MODELS["fsrcnnx_8"])}
    folder = os.path.join(poc_a.HERE, "data", "er_4k")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for size in ((1024, 576), (800, 450)):
        acc = {}
        for f in files:
            im = Image.open(os.path.join(folder, f)).convert("RGB")
            ref, low = im.resize(poc_er.TARGET, Image.BOX), im.resize(size, Image.BOX)
            out = {
                "bilineaire": low.resize(poc_er.TARGET, Image.BILINEAR),
                "easu": poc_fsrcnnx.gpu_upscale(poc_a.SHADERS["easu"], low, poc_er.TARGET, "d_easu", work),
                "lanczos3_ar": poc_fsrcnnx.gpu_upscale(poc_a.SHADERS["lanczos3_ar"], low, poc_er.TARGET, "d_l3", work),
                "fsrcnnx_8": poc_fsrcnnx.fsrcnnx_upscale(models["fsrcnnx_8"], low, poc_er.TARGET),
            }
            dref = detail(ref)
            for k, v in out.items():
                dv = detail(v)
                energy = dv.std() / dref.std()
                mse = np.mean((dv - dref) ** 2)
                acc.setdefault(k, []).append((energy, 10 * np.log10(255 ** 2 / mse)))
        print(f"{size[0]}x{size[1]} -> 1280x720 : détail conservé (énergie) / PSNR du détail")
        for k, v in acc.items():
            print(f"  {k:12} {100 * np.mean([x[0] for x in v]):5.1f} %   {np.mean([x[1] for x in v]):6.2f} dB")
        print()


if __name__ == "__main__":
    main()
