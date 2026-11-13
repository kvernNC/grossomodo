#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Évalue un ou plusieurs modèles (train_gm.py) contre FSR 1, GrossoModo/FSRCNNX et LS1, protocole de sweep_800p.py.

    python3 -I train/eval_gm.py --test a.jpg,b.jpg --models m1.pt,m2.pt [--ls1-spv DOSSIER] [--sizes 640x400,800x500]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)
import poc_a  # noqa: E402
import poc_er_detail  # noqa: E402
from sweep_800p import CROP, OUT  # noqa: E402
from train_gm import Net  # noqa: E402

torch.set_grad_enabled(False)


def load(p):
    d = torch.load(p)
    n = Net(d["C"], d["L"])
    n.load_state_dict(d["state"])
    return n.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--models", default="")
    ap.add_argument("--ls1-spv", default="")
    ap.add_argument("--ravu", default="")
    ap.add_argument("--sizes", default="640x400,800x500,960x600")
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()
    sizes = [tuple(int(v) for v in s.split("x")) for s in a.sizes.split(",")]
    models = {os.path.basename(p): load(p) for p in a.models.split(",") if p}
    tmp = tempfile.mkdtemp()
    acc = {}
    for path in a.test.split(","):
        big = Image.open(path).convert("RGB")
        x0, y0 = (big.width - CROP[0] - a.frames) // 2, (big.height - CROP[1]) // 2
        for size in sizes:
            refs, outs = [], {}
            for t in range(a.frames):
                crop = big.crop((x0 + t, y0, x0 + t + CROP[0], y0 + CROP[1]))
                refs.append(np.asarray(crop.resize(OUT, Image.BOX), np.float64))
                low = crop.resize(size, Image.BOX)
                src, dst = os.path.join(tmp, "l.rgba"), os.path.join(tmp, "o.rgba")
                poc_a.write_rgba(src, low)
                rd = lambda: np.asarray(poc_a.read_rgba(dst).convert("RGB"), np.float64)
                res = {}
                for s in (0.0, 1.0):
                    subprocess.run([os.path.join(poc_a.BUILD, "fsrcnnx_vk"), os.path.join(poc_a.BUILD, "fsrcnnx8"), src, dst,
                                    str(OUT[0]), str(OUT[1]), "1", "fp16", f"strength={s}"], check=True, stdout=subprocess.DEVNULL)
                    res[s] = rd()
                outs.setdefault("FSR1", []).append(res[0.0])
                outs.setdefault("GM0.5", []).append(np.clip(np.rint(res[0.0] + 0.5 * (res[1.0] - res[0.0])), 0, 255))
                if a.ls1_spv:
                    for m, v in (("quality", 0), ("quality", 2), ("performance", 0)):
                        subprocess.run([os.path.join(poc_a.BUILD, "ls1_vk"), a.ls1_spv, m, str(v), src, dst, str(OUT[0]), str(OUT[1])],
                                       check=True, stdout=subprocess.DEVNULL)
                        outs.setdefault(f"LS1-{m[0].upper()}{v}", []).append(rd())
                for r in [r for r in a.ravu.split(",") if r]:
                    subprocess.run([os.path.join(poc_a.BUILD, "ravu_vk"), r, src, dst, str(OUT[0]), str(OUT[1])], check=True, stdout=subprocess.DEVNULL)
                    outs.setdefault(os.path.basename(r), []).append(rd())
                x = torch.from_numpy(np.asarray(low, np.float32) / 255).permute(2, 0, 1)[None]
                for name, net in models.items():
                    o = net(x, (OUT[1], OUT[0]))[0].permute(1, 2, 0).numpy()
                    outs.setdefault(name, []).append(np.clip(np.rint(o * 255), 0, 255).astype(np.float64))
            for m, os_ in outs.items():
                p = np.mean([poc_a.psnr(Image.fromarray(r.astype(np.uint8)), Image.fromarray(o.astype(np.uint8))) for r, o in zip(refs, os_)])
                d = np.mean([poc_er_detail.detail(Image.fromarray(o.astype(np.uint8))).std() /
                             poc_er_detail.detail(Image.fromarray(r.astype(np.uint8))).std() for r, o in zip(refs, os_)])
                te = np.mean([np.mean(np.abs((poc_a.luma(os_[t]) - poc_a.luma(os_[t - 1])) - (poc_a.luma(refs[t]) - poc_a.luma(refs[t - 1]))))
                              for t in range(1, a.frames)])
                acc.setdefault((size, m), []).append((p, d, te))
    print(f"test : {', '.join(os.path.basename(p) for p in a.test.split(','))} -> 1280x800")
    for size in sizes:
        for (s, m), v in acc.items():
            if s == size:
                p, d, te = np.mean(v, 0)
                print(f"  {size[0]}x{size[1]:<4} {m:>14} {p:6.2f}dB {100 * d:6.1f}% {te:7.3f}")
        print()


if __name__ == "__main__":
    main()
