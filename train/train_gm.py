#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Petit réseau GrossoModo : entraîné pour le Deck, avec pénalité de scintillement.

Architecture (pensée pour 2-4 passes Vulkan) : sur la luminance à la résolution du jeu, conv3x3 1->C, L convs 3x3
C->C (PReLU), conv3x3 C->4 = carte de détail 2x (pixel shuffle). Reconstruction à la résolution de sortie :
Catmull-Rom de l'image RGB + détail échantillonné bilinéairement (identique pour R, G, B). Toute la chaîne est
différentiable, entraînée de bout en bout aux facteurs visés (x1,25 à x2).

Perte : L1 sur deux images décalées d'une fraction de pixel + LAMBDA x L1 de (variation de sortie - variation de
référence), c'est-à-dire exactement la mesure de scintillement de sweep_800p.py.

    python3 -I train/train_gm.py --train a.jpg,b.jpg --out model.pt [--C 8 --L 2 --steps 3000]
"""
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

LUMA = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


class Net(nn.Module):
    def __init__(self, C=8, L=2):
        super().__init__()
        self.first = nn.Conv2d(1, C, 3, padding=1, padding_mode="replicate")
        self.body = nn.ModuleList(nn.Conv2d(C, C, 3, padding=1, padding_mode="replicate") for _ in range(L))
        self.acts = nn.ModuleList(nn.PReLU(C) for _ in range(L + 1))
        self.last = nn.Conv2d(C, 4, 3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def detail(self, rgb):
        y = (rgb * LUMA).sum(1, keepdim=True)
        x = self.acts[0](self.first(y))
        for conv, act in zip(self.body, self.acts[1:]):
            x = act(conv(x))
        return F.pixel_shuffle(self.last(x), 2)  # 1 canal, 2x

    def forward(self, rgb, out_hw):
        base = F.interpolate(rgb, size=out_hw, mode="bicubic", align_corners=False)  # Catmull-Rom (a=-0.75 ~ cr)
        d = F.interpolate(self.detail(rgb), size=out_hw, mode="bilinear", align_corners=False)
        return base + d


def box(img, size):
    return np.asarray(img.resize(size, Image.BOX), np.float32) / 255.0


def to_t(a):
    return torch.from_numpy(a).permute(2, 0, 1)


def make_batch(imgs, n, out=96, scales=(1.25, 2.0)):
    xs1, xs2, rs1, rs2 = [], [], [], []
    s = random.uniform(*scales)
    inn = max(8, round(out / s))
    for _ in range(n):
        im = random.choice(imgs)
        K = random.randint(int(out * 2), int(out * 4))
        sh = random.randint(1, 3)
        x0, y0 = random.randint(0, im.width - K - sh - 1), random.randint(0, im.height - K - 1)
        c1 = im.crop((x0, y0, x0 + K, y0 + K))
        c2 = im.crop((x0 + sh, y0, x0 + sh + K, y0 + K))
        k = random.randint(0, 7)
        def aug(a):
            a = np.rot90(a, k % 4)
            return np.ascontiguousarray(a[:, ::-1] if k >= 4 else a)
        xs1.append(to_t(aug(box(c1, (inn, inn))))); xs2.append(to_t(aug(box(c2, (inn, inn)))))
        rs1.append(to_t(aug(box(c1, (out, out))))); rs2.append(to_t(aug(box(c2, (out, out)))))
    st = torch.stack
    return st(xs1), st(xs2), st(rs1), st(rs2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    imgs = [Image.open(p).convert("RGB") for p in a.train.split(",")]
    net = Net(a.C, a.L)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    for step in range(a.steps):
        x1, x2, r1, r2 = make_batch(imgs, a.batch)
        hw = r1.shape[-2:]
        o1, o2 = net(x1, hw), net(x2, hw)
        y = lambda t: (t * LUMA).sum(1)
        loss = (o1 - r1).abs().mean() + (o2 - r2).abs().mean() + a.lam * ((y(o2) - y(o1)) - (y(r2) - y(r1))).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps - 1:
            print(f"  pas {step:5d} perte {loss.item():.5f}", flush=True)
    torch.save({"C": a.C, "L": a.L, "state": net.state_dict()}, a.out)
    print("params :", sum(p.numel() for p in net.parameters()))


if __name__ == "__main__":
    main()
