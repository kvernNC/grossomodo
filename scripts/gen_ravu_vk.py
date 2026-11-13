#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Convertit un shader RAVU-Zoom de mpv-prescalers (bjin, LGPL-3, dossier compute/, variante -rgb) en shader de
calcul Vulkan autonome + tables binaires (rgba16f) :
  liaison 0 : image source (sampler2D), 1 : sortie (image de stockage), 2 et 3 : tables (sampler2D linéaire).
  constantes poussées : vec4(largeur source, hauteur source, largeur sortie, hauteur sortie).

    python3 -I scripts/gen_ravu_vk.py ravu-zoom-ar-r3-rgb.hook DOSSIER_SORTIE
"""
import os
import re
import subprocess
import sys

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
text = open(src).read()
blocks = re.split(r"^(?=//!(?:DESC|TEXTURE))", text, flags=re.M)
code, luts = None, []
for b in blocks:
    lines = b.splitlines()
    if b.startswith("//!DESC"):
        code = "\n".join(l for l in lines if not l.startswith("//!"))
        group = re.search(r"^//!COMPUTE (\d+) (\d+)", b, re.M).groups()
    elif b.startswith("//!TEXTURE"):
        name = lines[0].split()[1]
        w, h = map(int, re.search(r"^//!SIZE (\d+) (\d+)", b, re.M).groups())
        data = bytes.fromhex("".join(l.strip() for l in lines if not l.startswith("//!")))
        assert len(data) == w * h * 16, (name, len(data))  # flottants 32 bits malgré FORMAT rgba16f
        # Format voulu par le shader : rgba16f (libplacebo convertit) ; on stocke directement en demi-flottants.
        import array
        f = array.array("f"); f.frombytes(data)
        import struct
        open(os.path.join(out, f"{name}.bin"), "wb").write(struct.pack(f"<{len(f)}e", *f))
        luts.append((name, w, h))
assert code and len(luts) in (1, 2)
head = f"""#version 450
// Généré par scripts/gen_ravu_vk.py depuis {os.path.basename(src)} (mpv-prescalers, bjin, LGPL-3.0-or-later).
layout(local_size_x = {group[0]}, local_size_y = {group[1]}) in;
layout(binding = 0) uniform sampler2D src_tex;
layout(binding = 1) uniform writeonly image2D out_image;
""" + "".join(f"layout(binding = {2 + i}) uniform sampler2D {n};\n" for i, (n, _, _) in enumerate(luts)) + f"""layout(push_constant) uniform PC {{ vec4 sizes; }};
#define HOOKED_size (sizes.xy)
#define HOOKED_pt (1.0 / sizes.xy)
#define HOOKED_map(id) ((vec2(id) + 0.5) / sizes.zw)
#define HOOKED_tex(p) TEX(p)
const vec3 LUMA_W = vec3(0.2126, 0.7152, 0.0722);
"""
code = code.replace("void hook() {", "void main() {\nif (any(greaterThanEqual(gl_GlobalInvocationID.xy, uvec2(sizes.zw)))) return;", 1)
# Le retour anticipé avant barrier() est interdit : on borne l'écriture à la place.
code = code.replace("void main() {\nif (any(greaterThanEqual(gl_GlobalInvocationID.xy, uvec2(sizes.zw)))) return;", "void main() {", 1)
luma = "vec4(res, 0.0, 0.0, 0.0)" in code
# Variante luminance : RAVU sur la luminance, couleur = bilinéaire + (luminance RAVU - luminance bilinéaire).
head += ("#define TEX(p) vec4(dot(texture(src_tex, (p)).rgb, LUMA_W))\n" if luma else "#define TEX(p) texture(src_tex, (p))\n")
code = code.replace("imageStore(out_image, ivec2(gl_GlobalInvocationID), vec4(res, 0.0, 0.0, 0.0));",
                    "vec3 base = texture(src_tex, (vec2(gl_GlobalInvocationID.xy) + 0.5) / sizes.zw).rgb;\n"
                    "if (all(lessThan(gl_GlobalInvocationID.xy, uvec2(sizes.zw)))) imageStore(out_image, ivec2(gl_GlobalInvocationID), "
                    "vec4(clamp(base + (res - dot(base, LUMA_W)), 0.0, 1.0), 1.0));")
code = code.replace("imageStore(out_image, ivec2(gl_GlobalInvocationID), vec4(res, 1.0));",
                    "if (all(lessThan(gl_GlobalInvocationID.xy, uvec2(sizes.zw)))) imageStore(out_image, ivec2(gl_GlobalInvocationID), vec4(clamp(res, 0.0, 1.0), 1.0));")
open(os.path.join(out, "ravu.comp"), "w").write(head + code)
open(os.path.join(out, "luts.txt"), "w").write("".join(f"{n} {w} {h}\n" for n, w, h in luts))
subprocess.run(["glslangValidator", "-V", "--quiet", "--target-env", "vulkan1.1", os.path.join(out, "ravu.comp"),
                "-o", os.path.join(out, "ravu.spv")], check=True)
print("ok", luts, "groupe", group)
