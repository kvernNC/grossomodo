#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Images finales comme dans gamescope : agrandisseur, puis RCAS (shaders/cs_rcas.comp), sur les captures d'Elden Ring.

Netteté de gamescope : nUpscaleSharpness de 0 (maximum) à 20 (minimum), 2 par défaut ; RCAS reçoit
nUpscaleSharpness / 10 stops. On compare EASU (le FSR 1 actuel), Lanczos-3 + anti-ringing et FSRCNNX 8, chacun
suivi de RCAS à plusieurs réglages, contre la référence 1280x720. Mesures : PSNR, SSIM, énergie du détail fin
(100 % = autant que la référence, plus = sur-accentué) et PSNR du détail (le bon détail au bon endroit).

    python3 -I scripts/poc_er_rcas.py
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsrcnnx  # noqa: E402
import poc_a  # noqa: E402
import poc_er  # noqa: E402
import poc_er_detail  # noqa: E402
import poc_fsrcnnx  # noqa: E402

RCAS_SPV = os.path.join(poc_a.BUILD, "cs_rcas.spv")
SETTINGS = [None, 20, 10, 5, 2, 0]  # None : sans RCAS ; sinon nUpscaleSharpness de gamescope


def rcas(img, sharpness, work, tag):
    src, dst = os.path.join(work, f"{tag}_rcas_in.rgba"), os.path.join(work, f"{tag}_rcas_out.rgba")
    poc_a.write_rgba(src, img)
    subprocess.run([poc_a.HARNESS, RCAS_SPV, src, dst, str(img.width), str(img.height), "1",
                    f"rcas={sharpness / 10:g}"], check=True, stdout=subprocess.DEVNULL)
    return poc_a.read_rgba(dst)


def metrics(ref, img):
    dref, dimg = poc_er_detail.detail(ref), poc_er_detail.detail(img)
    return (poc_a.psnr(ref, img), poc_a.ssim(ref, img), dimg.std() / dref.std(),
            10 * np.log10(255 ** 2 / np.mean((dimg - dref) ** 2)))


def main():
    work = os.path.join(poc_a.BUILD, "poc_er_rcas")
    os.makedirs(work, exist_ok=True)
    model = fsrcnnx.parse(poc_fsrcnnx.MODELS["fsrcnnx_8"])
    folder = os.path.join(poc_a.HERE, "data", "er_4k")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for size in ((1024, 576), (800, 450)):
        acc = {}
        for f in files:
            name = os.path.splitext(f)[0][-12:]
            im = Image.open(os.path.join(folder, f)).convert("RGB")
            ref, low = im.resize(poc_er.TARGET, Image.BOX), im.resize(size, Image.BOX)
            base = {
                "easu": poc_fsrcnnx.gpu_upscale(poc_a.SHADERS["easu"], low, poc_er.TARGET, "r_easu", work),
                "lanczos3_ar": poc_fsrcnnx.gpu_upscale(poc_a.SHADERS["lanczos3_ar"], low, poc_er.TARGET, "r_l3", work),
                "fsrcnnx_8": poc_fsrcnnx.fsrcnnx_upscale(model, low, poc_er.TARGET),
            }
            finals = {}
            for k, img in base.items():
                for sh in SETTINGS:
                    out = img if sh is None else rcas(img, sh, work, f"{k}_{sh}")
                    finals[(k, sh)] = out
                    acc.setdefault((k, sh), []).append(metrics(ref, out))
            if size == (800, 450):
                box = (540, 300, 740, 420)
                poc_er.sheet(ref, {f"{k} + rcas {sh}": finals[(k, sh)] for k, sh in
                                   (("easu", 2), ("lanczos3_ar", 2), ("fsrcnnx_8", 10), ("fsrcnnx_8", 2))},
                             ["reference", "easu + rcas 2", "lanczos3_ar + rcas 2", "fsrcnnx_8 + rcas 10",
                              "fsrcnnx_8 + rcas 2"], box, os.path.join(work, f"{name}_800x450_final.png"))
        print(f"{size[0]}x{size[1]} -> 1280x720, image finale (netteté gamescope 0 = max, 20 = min, 2 par défaut)")
        print(f"  {'méthode':12} {'netteté':>8} {'PSNR':>8} {'SSIM':>7} {'détail':>8} {'PSNR dét.':>10}")
        for (k, sh), v in acc.items():
            p, s, e, dp = (np.mean([x[i] for x in v]) for i in range(4))
            label = "sans" if sh is None else str(sh)
            print(f"  {k:12} {label:>8} {p:6.2f}dB {s:7.4f} {100 * e:6.1f} % {dp:8.2f}dB")
        print()


if __name__ == "__main__":
    main()
