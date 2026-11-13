#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Détail contre scintillement selon l'intensité du mélange FSR 1 / FSRCNNX (strength), sur les captures d'Elden Ring.

Panoramique simulé : la capture 4K est recadrée avec un décalage d'un pixel 4K par image (1/3 de pixel de sortie,
0,2 pixel de rendu), puis réduite en référence 1200x675 et en rendu 750x422 (x1,6). Chaque rendu est agrandi par le
shader final (strength=0 : FSR 1 ; strength=1 : FSRCNNX ; entre les deux : mélange linéaire, calculé ici à partir des
deux extrêmes, ce qui est exact puisque le shader mélange avant de borner dans [0, 1]).

Mesures (moyennes) : PSNR, énergie du détail fin (100 % = référence), et **erreur temporelle** : écart entre la
variation d'une image à la suivante en sortie et la variation réelle de la référence (luminance, plus bas = moins de
scintillement).

    python3 -I scripts/poc_strength.py
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poc_a  # noqa: E402
import poc_er_detail  # noqa: E402

FRAMES = 8
REF = (1200, 675)
LOW = (750, 422)
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def upscale(low, strength, work):
    src, dst = os.path.join(work, "s_low.rgba"), os.path.join(work, "s_out.rgba")
    poc_a.write_rgba(src, low)
    subprocess.run([os.path.join(poc_a.BUILD, "fsrcnnx_vk"), os.path.join(poc_a.BUILD, "fsrcnnx8"), src, dst,
                    str(REF[0]), str(REF[1]), "1", "fp16", f"strength={strength}"], check=True, stdout=subprocess.DEVNULL)
    return np.asarray(poc_a.read_rgba(dst), np.float64)


def main():
    work = os.path.join(poc_a.BUILD, "poc_strength")
    os.makedirs(work, exist_ok=True)
    folder = os.path.join(poc_a.HERE, "data", "er_4k")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    acc = {s: {"psnr": [], "detail": [], "temporal": []} for s in STRENGTHS}
    for f in files:
        big = Image.open(os.path.join(folder, f)).convert("RGB")
        refs, easu, net = [], [], []
        for t in range(FRAMES):
            crop = big.crop((t, 0, t + 3600, 2025))
            refs.append(np.asarray(crop.resize(REF, Image.BOX), np.float64))
            low = crop.resize(LOW, Image.BOX)
            easu.append(upscale(low, 0.0, work))
            net.append(upscale(low, 1.0, work))
        for s in STRENGTHS:
            outs = [np.clip(np.rint(e + s * (n - e)), 0, 255) for e, n in zip(easu, net)]
            for r, o in zip(refs, outs):
                ri, oi = Image.fromarray(r.astype(np.uint8)), Image.fromarray(o.astype(np.uint8))
                acc[s]["psnr"].append(poc_a.psnr(ri, oi))
                dr, do = poc_er_detail.detail(ri), poc_er_detail.detail(oi)
                acc[s]["detail"].append(do.std() / dr.std())
            for t in range(1, FRAMES):
                dref = poc_a.luma(refs[t]) - poc_a.luma(refs[t - 1])
                dout = poc_a.luma(outs[t]) - poc_a.luma(outs[t - 1])
                acc[s]["temporal"].append(np.mean(np.abs(dout - dref)))
        print(f"  {f} : fait", flush=True)
    print(f"\n{len(files)} captures, panoramique de {FRAMES} images, 750x422 -> 1200x675 (x1,6), avant RCAS")
    print(f"  {'intensité':>9} {'PSNR':>8} {'détail':>8} {'erreur temporelle':>18}")
    for s in STRENGTHS:
        a = acc[s]
        print(f"  {s:9.2f} {np.mean(a['psnr']):6.2f}dB {100 * np.mean(a['detail']):6.1f} % {np.mean(a['temporal']):14.3f}")


if __name__ == "__main__":
    main()
