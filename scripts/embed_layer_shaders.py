#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Embarque les shaders FSRCNNX générés (plan.txt + pass_NN[_fp16].spv) dans un en-tête C++ pour la couche Vulkan.

    python3 -I scripts/embed_layer_shaders.py build/fsrcnnx8 layer/fsrcnnx8_spv.h
"""
import os
import sys

src, out = sys.argv[1], sys.argv[2]
plan = [line.split() for line in open(os.path.join(src, "plan.txt")) if line.strip()]
with open(out, "w") as f:
    f.write("// Généré par scripts/embed_layer_shaders.py depuis %s. Ne pas modifier.\n" % src)
    f.write("// Shaders : FSRCNNX 8-0-4-1 (igv), réseau et poids sous LGPL-3.0-or-later (third_party/fsrcnnx/).\n")
    f.write("#pragma once\n#include <cstddef>\n#include <cstdint>\n\nnamespace fsup {\n\n")
    for name, *_ in plan:
        for suffix in ("", "_fp16"):
            data = open(os.path.join(src, name + suffix + ".spv"), "rb").read()
            words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
            f.write(f"static const uint32_t k_{name}{suffix}[] = {{\n")
            for i in range(0, len(words), 8):
                f.write("    " + ", ".join(f"0x{w:08x}u" for w in words[i:i + 8]) + ",\n")
            f.write("};\n")
    f.write("\nstruct PassInfo {\n    const char *name;\n    const uint32_t *spv32;\n    size_t size32;\n"
            "    const uint32_t *spv16;\n    size_t size16;\n    const char *in, *skip, *out, *grid;\n};\n\n")
    f.write(f"static const PassInfo k_passes[{len(plan)}] = {{\n")
    for name, a, b, c, g in plan:
        f.write(f'    {{"{name}", k_{name}, sizeof(k_{name}), k_{name}_fp16, sizeof(k_{name}_fp16), '
                f'"{a}", "{b}", "{c}", "{g}"}},\n')
    f.write("};\n\n} // namespace fsup\n")
print(f"{len(plan)} passes -> {out}")
