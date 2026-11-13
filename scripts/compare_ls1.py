#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Comparaison avec LS1 (scaler spatial de Lossless Scaling), mêmes conditions que sweep_800p.py.

LS1 n'est pas distribué ici : --ls1-spv pointe vers les SPIR-V que l'utilisateur a traduits depuis SON Lossless.dll
(build/ls1_vk les exécute avec la chaîne de MAKO). LS1 inclut sa propre netteté (variantes 0 à 4) ; FSR 1 et
GrossoModo sont mesurés avant RCAS, comme dans sweep_800p.py.

    python3 -I scripts/compare_ls1.py --ls1-spv DOSSIER [--frames 6] [--jobs 4] [--variants 0,2,4]
"""
import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poc_a  # noqa: E402
import poc_er_detail  # noqa: E402
from sweep_800p import CROP, OUT, SIZES  # noqa: E402


def run_one(args):
    path, size, frames, spv, variants = args
    big = Image.open(path).convert("RGB")
    x0, y0 = (big.width - CROP[0] - frames) // 2, (big.height - CROP[1]) // 2
    tmp = tempfile.mkdtemp(prefix="ls1_")
    methods = ["FSR1", "GM0.5"] + [f"LS1-{m[0].upper()}{v}" for m in ("quality", "performance") for v in variants]
    refs, outs = [], {m: [] for m in methods}
    for t in range(frames):
        crop = big.crop((x0 + t, y0, x0 + t + CROP[0], y0 + CROP[1]))
        refs.append(np.asarray(crop.resize(OUT, Image.BOX), np.float64))
        src = os.path.join(tmp, "low.rgba")
        poc_a.write_rgba(src, crop.resize(size, Image.BOX))
        dst = os.path.join(tmp, "o.rgba")
        res = {}
        for s in (0.0, 1.0):
            subprocess.run([os.path.join(poc_a.BUILD, "fsrcnnx_vk"), os.path.join(poc_a.BUILD, "fsrcnnx8"), src, dst,
                            str(OUT[0]), str(OUT[1]), "1", "fp16", f"strength={s}"], check=True, stdout=subprocess.DEVNULL)
            res[s] = np.asarray(poc_a.read_rgba(dst).convert("RGB"), np.float64)
        outs["FSR1"].append(res[0.0])
        outs["GM0.5"].append(np.clip(np.rint(res[0.0] + 0.5 * (res[1.0] - res[0.0])), 0, 255))
        for m in ("quality", "performance"):
            for v in variants:
                subprocess.run([os.path.join(poc_a.BUILD, "ls1_vk"), spv, m, str(v), src, dst, str(OUT[0]), str(OUT[1])],
                               check=True, stdout=subprocess.DEVNULL)
                outs[f"LS1-{m[0].upper()}{v}"].append(np.asarray(poc_a.read_rgba(dst).convert("RGB"), np.float64))
    stats = {}
    for m in methods:
        p, d, te = [], [], []
        for r, o in zip(refs, outs[m]):
            ri, oi = Image.fromarray(r.astype(np.uint8)), Image.fromarray(o.astype(np.uint8))
            p.append(poc_a.psnr(ri, oi))
            d.append(poc_er_detail.detail(oi).std() / poc_er_detail.detail(ri).std())
        for t in range(1, frames):
            dref = poc_a.luma(refs[t]) - poc_a.luma(refs[t - 1])
            dout = poc_a.luma(outs[m][t]) - poc_a.luma(outs[m][t - 1])
            te.append(np.mean(np.abs(dout - dref)))
        stats[m] = (np.mean(p), np.mean(d), np.mean(te))
    return size, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls1-spv", required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--variants", default="0,2,4")
    args = ap.parse_args()
    variants = [int(v) for v in args.variants.split(",")]
    folder = os.path.join(poc_a.HERE, "data", "er_4k")
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    tasks = [(f, s, args.frames, args.ls1_spv, variants) for s in SIZES for f in files]
    acc = {}
    with ProcessPoolExecutor(args.jobs) as ex:
        for size, stats in ex.map(run_one, tasks):
            acc.setdefault(size, []).append(stats)
    print(f"{len(files)} captures, panoramique de {args.frames} images, vers 1280x800 ; FSR1/GM avant RCAS, LS1 avec sa netteté")
    print(f"  {'rendu':>9} {'méthode':>8} {'PSNR':>8} {'détail':>7} {'scint.':>7}")
    for size in SIZES:
        for m in acc[size][0]:
            p, d, te = (np.mean([x[m][i] for x in acc[size]]) for i in range(3))
            print(f"  {size[0]:>4}x{size[1]:<4} {m:>8} {p:6.2f}dB {100 * d:6.1f}% {te:7.3f}")
        print()


if __name__ == "__main__":
    main()
