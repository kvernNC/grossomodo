#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Quelle résolution de rendu pour un Steam Deck qui affiche en 1280x800 ? Balayage des résolutions 16:10 candidates.

Pour chaque capture 4K (data/er_4k/, recadrée en 16:10 : 3456x2160 = 1280x800 x 2,7) et chaque résolution de rendu :
panoramique simulé de FRAMES images (décalage d'un pixel 4K par image), rendu par moyenne de zones, agrandi vers
1280x800 par FSR 1 (EASU, strength=0) et par GrossoModo (strength 0,5 et 1), avant RCAS. Mesures moyennes : PSNR,
détail fin (100 % = référence 1280x800) et erreur temporelle (scintillement). Le coût GPU de GrossoModo sur le Deck
est estimé à partir des mesures (results/deck-2026-11-13-rapport3.txt) : réseau proportionnel aux pixels rendus,
réduction proportionnelle aux pixels de sortie.

    python3 -I scripts/sweep_800p.py [--frames 6] [--jobs 4]
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

OUT = (1280, 800)
CROP = (3456, 2160)
SIZES = [(640, 400), (720, 450), (800, 500), (854, 534), (960, 600), (1024, 640)]
STRENGTHS = (0.0, 0.5, 1.0)

# Steam Deck, FP16, rapport n°3 : réseau (passes 0-6) à 800x450 = 1,92 ms ; réduction Catmull-Rom = 0,289 ms à 1280x720.
NET_NS_PER_PX = 1.92e6 / (800 * 450)
DOWN_NS_PER_PX = 0.289e6 / (1280 * 720)
EASU_MS = 0.47 * (OUT[0] * OUT[1]) / (1280 * 720)


def deck_cost_ms(w, h):
    net = NET_NS_PER_PX * w * h / 1e6
    down = 0.0 if (2 * w, 2 * h) == OUT else DOWN_NS_PER_PX * OUT[0] * OUT[1] / 1e6
    return net + down


def run_one(args):
    path, size, frames = args
    big = Image.open(path).convert("RGB")
    x0, y0 = (big.width - CROP[0] - frames) // 2, (big.height - CROP[1]) // 2
    tmp = tempfile.mkdtemp(prefix="sweep_")
    refs, outs = [], {s: [] for s in STRENGTHS}
    for t in range(frames):
        crop = big.crop((x0 + t, y0, x0 + t + CROP[0], y0 + CROP[1]))
        refs.append(np.asarray(crop.resize(OUT, Image.BOX), np.float64))
        low = crop.resize(size, Image.BOX)
        src = os.path.join(tmp, "low.rgba")
        poc_a.write_rgba(src, low)
        res = {}
        for s in (0.0, 1.0):
            dst = os.path.join(tmp, f"o{s}.rgba")
            subprocess.run([os.path.join(poc_a.BUILD, "fsrcnnx_vk"), os.path.join(poc_a.BUILD, "fsrcnnx8"), src, dst,
                            str(OUT[0]), str(OUT[1]), "1", "fp16", f"strength={s}"], check=True, stdout=subprocess.DEVNULL)
            res[s] = np.asarray(poc_a.read_rgba(dst), np.float64)
        for s in STRENGTHS:
            outs[s].append(np.clip(np.rint(res[0.0] + s * (res[1.0] - res[0.0])), 0, 255))
    stats = {}
    for s in STRENGTHS:
        p, d, te = [], [], []
        for r, o in zip(refs, outs[s]):
            ri, oi = Image.fromarray(r.astype(np.uint8)), Image.fromarray(o.astype(np.uint8))
            p.append(poc_a.psnr(ri, oi))
            d.append(poc_er_detail.detail(oi).std() / poc_er_detail.detail(ri).std())
        for t in range(1, frames):
            dref = poc_a.luma(refs[t]) - poc_a.luma(refs[t - 1])
            dout = poc_a.luma(outs[s][t]) - poc_a.luma(outs[s][t - 1])
            te.append(np.mean(np.abs(dout - dref)))
        stats[s] = (np.mean(p), np.mean(d), np.mean(te))
    return size, os.path.basename(path), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    folder = os.path.join(poc_a.HERE, "data", "er_4k")
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    tasks = [(f, s, args.frames) for s in SIZES for f in files]
    acc = {}
    with ProcessPoolExecutor(args.jobs) as ex:
        for size, name, stats in ex.map(run_one, tasks):
            acc.setdefault(size, []).append(stats)
            print(f"  {size[0]}x{size[1]} {name} : fait", flush=True)

    print(f"\n{len(files)} captures, panoramique de {args.frames} images, vers 1280x800, avant RCAS")
    print(f"  coût Deck estimé : EASU {EASU_MS:.2f} ms ; GrossoModo = réseau + réduction")
    hdr = (f"  {'rendu':>9} {'facteur':>7} {'pixels':>6} {'coût GM':>8} | {'FSR1 PSNR':>9} {'détail':>6} {'scint.':>6} |"
           f" {'GM0,5 PSNR':>10} {'détail':>6} {'scint.':>6} | {'GM1 PSNR':>8} {'détail':>6} {'scint.':>6}")
    print(hdr)
    for size in SIZES:
        v = acc[size]
        row = f"  {size[0]:>4}x{size[1]:<4} x{OUT[0] / size[0]:<6.2f} {100 * size[0] * size[1] / (OUT[0] * OUT[1]):5.0f}% {deck_cost_ms(*size):6.2f}ms |"
        for s in STRENGTHS:
            p, d, te = (np.mean([x[s][i] for x in v]) for i in range(3))
            row += f" {p:8.2f}dB {100 * d:5.1f}% {te:6.3f} |"
        print(row.rstrip("|"))


if __name__ == "__main__":
    main()
