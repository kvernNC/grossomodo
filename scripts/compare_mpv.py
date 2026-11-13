#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mesure des shaders mpv tiers (CuNNy, RAVU, Anime4K, FSRCNNX...) dans le protocole de sweep_800p.py, exécutés par
ffmpeg + libplacebo (Vulkan) : shader de luminance, puis réduction Catmull-Rom de libplacebo vers 1280x800.
Calibrage : FSRCNNX par ce chemin donne le même PSNR que notre implémentation (36,57 contre 36,55 dB).

    python3 -I scripts/compare_mpv.py --shaders a.glsl,b.hook [--sizes 640x400,800x500] [--frames 6]
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
from sweep_800p import CROP, OUT  # noqa: E402

VF = ("format=yuv444p16le,libplacebo=w={w}:h={h}:format=rgb24:upscaler=ewa_lanczos:downscaler=catmull_rom"
      ":colorspace=bt709:color_primaries=bt709:color_trc=iec61966-2-1:range=pc{shader}")


def run_one(args):
    path, size, frames, shaders = args
    big = Image.open(path).convert("RGB")
    x0, y0 = (big.width - CROP[0] - frames) // 2, (big.height - CROP[1]) // 2
    tmp = tempfile.mkdtemp(prefix="mpv_")
    refs = []
    for t in range(frames):
        crop = big.crop((x0 + t, y0, x0 + t + CROP[0], y0 + CROP[1]))
        refs.append(np.asarray(crop.resize(OUT, Image.BOX), np.float64))
        crop.resize(size, Image.BOX).save(os.path.join(tmp, f"low_{t}.png"))
    stats = {}
    for sh in ["(aucun: EWA Lanczos)"] + shaders:
        opt = "" if sh.startswith("(") else f":custom_shader_path={sh}"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-init_hw_device", "vulkan", "-i", os.path.join(tmp, "low_%d.png"),
                        "-vf", VF.format(w=OUT[0], h=OUT[1], shader=opt), os.path.join(tmp, "o_%d.png")], check=True)
        outs = [np.asarray(Image.open(os.path.join(tmp, f"o_{t}.png")).convert("RGB"), np.float64) for t in range(1, frames + 1)]
        p = np.mean([poc_a.psnr(Image.fromarray(r.astype(np.uint8)), Image.fromarray(o.astype(np.uint8))) for r, o in zip(refs, outs)])
        d = np.mean([poc_er_detail.detail(Image.fromarray(o.astype(np.uint8))).std() /
                     poc_er_detail.detail(Image.fromarray(r.astype(np.uint8))).std() for r, o in zip(refs, outs)])
        te = np.mean([np.mean(np.abs((poc_a.luma(outs[t]) - poc_a.luma(outs[t - 1])) - (poc_a.luma(refs[t]) - poc_a.luma(refs[t - 1]))))
                      for t in range(1, frames)])
        stats[os.path.basename(sh)] = (p, d, te)
    return size, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shaders", required=True)
    ap.add_argument("--sizes", default="640x400,800x500")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--files", default="")
    a = ap.parse_args()
    sizes = [tuple(int(v) for v in s.split("x")) for s in a.sizes.split(",")]
    folder = os.path.join(poc_a.HERE, "data", "er_4k")
    files = a.files.split(",") if a.files else sorted(os.path.join(folder, f) for f in os.listdir(folder)
                                                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
    acc = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for size, stats in ex.map(run_one, [(f, s, a.frames, a.shaders.split(",")) for s in sizes for f in files]):
            acc.setdefault(size, []).append(stats)
    print(f"{len(files)} captures, panoramique de {a.frames} images, vers 1280x800 (libplacebo, avant RCAS)")
    for size in sizes:
        for m in acc[size][0]:
            p, d, te = (np.mean([x[m][i] for x in acc[size]]) for i in range(3))
            print(f"  {size[0]:>4}x{size[1]:<4} {m:>28} {p:6.2f}dB {100 * d:6.1f}% {te:7.3f}")
        print()


if __name__ == "__main__":
    main()
