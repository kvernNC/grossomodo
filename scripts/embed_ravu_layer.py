#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Embarque RAVU-Lite (sortie de gen_ravu_vk.py : ravu.spv, final.spv, table rgba16f) dans un en-tête C++ pour la couche.

    python3 -I scripts/embed_ravu_layer.py build/ravu_lite_ar_r3 layer/ravu_spv.h
"""
import os
import sys

src, out = sys.argv[1], sys.argv[2]
luts = [l.split() for l in open(os.path.join(src, "luts.txt")) if l.strip()]
assert len(luts) == 1, "la couche prend une seule table (RAVU-Lite)"
name, w, h = luts[0][0], int(luts[0][1]), int(luts[0][2])
gw, gh, kind = open(os.path.join(src, "group.txt")).read().split()
assert kind == "lite"


def words(path):
    data = open(path, "rb").read()
    return [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]


def array(f, cname, ws, ctype="uint32_t", fmt="0x{:08x}u", per=8):
    f.write(f"static const {ctype} {cname}[] = {{\n")
    for i in range(0, len(ws), per):
        f.write("    " + ", ".join(fmt.format(x) for x in ws[i:i + per]) + ",\n")
    f.write("};\n")


with open(out, "w") as f:
    f.write(f"// Généré par scripts/embed_ravu_layer.py depuis {src}. Ne pas modifier.\n")
    f.write("// RAVU-Lite de bjin (mpv-prescalers), shader et table sous LGPL-3.0-or-later (third_party/mpv-prescalers/).\n")
    f.write("#pragma once\n#include <cstddef>\n#include <cstdint>\n\nnamespace fsup {\n\n")
    array(f, "k_ravu_lite", words(os.path.join(src, "ravu.spv")))
    array(f, "k_ravu_final", words(os.path.join(src, "final.spv")))
    lut = open(os.path.join(src, f"{name}.bin"), "rb").read()
    assert len(lut) == w * h * 8
    halves = [int.from_bytes(lut[i:i + 2], "little") for i in range(0, len(lut), 2)]
    array(f, "k_ravu_lut", halves, "uint16_t", "0x{:04x}", 12)
    f.write(f"\nconstexpr uint32_t kRavuLutW = {w}, kRavuLutH = {h};  // rgba16f\n")
    f.write(f"constexpr uint32_t kRavuGroupW = {gw}, kRavuGroupH = {gh};  // pixels source par groupe (passe x2)\n")
    f.write(f'constexpr const char *kRavuName = "{os.path.basename(src.rstrip("/"))}";\n')
    f.write("\n} // namespace fsup\n")
print(out)
