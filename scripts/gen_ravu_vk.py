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
        nums = re.search(r"^//!COMPUTE ([\d ]+)$", b, re.M).group(1).split()
        group = nums[2:4] if len(nums) == 4 else nums[:2]  # "bw bh tw th" : taille du groupe de fils = tw x th
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
luma = "vec4(res, 0.0, 0.0, 0.0)" in code or "vec4(res[0], 0.0, 0.0, 0.0)" in code
lite = "//!WIDTH 2 HOOKED.w *" in text
if lite:
    head = head.replace("layout(binding = 1) uniform writeonly image2D out_image;", "layout(binding = 1, r16f) uniform writeonly image2D out_image;")
# Variante luminance : RAVU sur la luminance, couleur = bilinéaire + (luminance RAVU - luminance bilinéaire).
head += ("#define TEX(p) vec4(dot(texture(src_tex, (p)).rgb, LUMA_W))\n" if luma else "#define TEX(p) texture(src_tex, (p))\n")
code = code.replace("imageStore(out_image, ivec2(gl_GlobalInvocationID), vec4(res, 0.0, 0.0, 0.0));",
                    "vec3 base = texture(src_tex, (vec2(gl_GlobalInvocationID.xy) + 0.5) / sizes.zw).rgb;\n"
                    "if (all(lessThan(gl_GlobalInvocationID.xy, uvec2(sizes.zw)))) imageStore(out_image, ivec2(gl_GlobalInvocationID), "
                    "vec4(clamp(base + (res - dot(base, LUMA_W)), 0.0, 1.0), 1.0));")
code = code.replace("imageStore(out_image, ivec2(gl_GlobalInvocationID), vec4(res, 1.0));",
                    "if (all(lessThan(gl_GlobalInvocationID.xy, uvec2(sizes.zw)))) imageStore(out_image, ivec2(gl_GlobalInvocationID), vec4(clamp(res, 0.0, 1.0), 1.0));")
open(os.path.join(out, "ravu.comp"), "w").write(head + code)
if lite:
    open(os.path.join(out, "group.txt"), "w").write(f"{group[0]} {group[1]} lite\n")
    final = """#version 450
// Réduction de l'image x2 de RAVU-Lite (luminance) vers la sortie : Catmull-Rom en 9 lectures bilinéaires,
// couleur = bilinéaire de la source + (luminance réduite - luminance bilinéaire).
layout(local_size_x = 8, local_size_y = 8) in;
// SPLIT = 1 : écran partagé, la moitié gauche est laissée à FSR 1 (déjà écrite), trait blanc au milieu.
layout(constant_id = 0) const int SPLIT = 0;
layout(binding = 0) uniform sampler2D src_tex;
layout(binding = 1) uniform writeonly image2D out_image;
layout(binding = 2) uniform sampler2D x2_tex;
layout(push_constant) uniform PC { vec4 sizes; };
const vec3 LUMA_W = vec3(0.2126, 0.7152, 0.0722);
float cr9(vec2 uv, vec2 size)
{
    vec2 p = uv * size, t1 = floor(p - 0.5) + 0.5, f = p - t1;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2, o12 = w2 / w12;
    vec2 t0 = (t1 - 1.0) / size, t3 = (t1 + 2.0) / size, t12 = (t1 + o12) / size;
    float r = 0.0;
    r += texture(x2_tex, vec2(t0.x, t0.y)).x * w0.x * w0.y;   r += texture(x2_tex, vec2(t12.x, t0.y)).x * w12.x * w0.y;
    r += texture(x2_tex, vec2(t3.x, t0.y)).x * w3.x * w0.y;   r += texture(x2_tex, vec2(t0.x, t12.y)).x * w0.x * w12.y;
    r += texture(x2_tex, vec2(t12.x, t12.y)).x * w12.x * w12.y; r += texture(x2_tex, vec2(t3.x, t12.y)).x * w3.x * w12.y;
    r += texture(x2_tex, vec2(t0.x, t3.y)).x * w0.x * w3.y;   r += texture(x2_tex, vec2(t12.x, t3.y)).x * w12.x * w3.y;
    r += texture(x2_tex, vec2(t3.x, t3.y)).x * w3.x * w3.y;
    return r;
}
void main()
{
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pos, ivec2(sizes.zw)))) return;
    if (SPLIT != 0) {
        int mid = int(sizes.z) / 2;
        if (pos.x < mid) return;
        if (pos.x == mid) { imageStore(out_image, pos, vec4(1.0)); return; }
    }
    vec2 uv = (vec2(pos) + 0.5) / sizes.zw;
    vec2 x2 = 2.0 * sizes.xy;
    float y = all(equal(sizes.zw, x2)) ? texelFetch(x2_tex, pos, 0).x : cr9(uv, x2);
    vec3 base = texture(src_tex, uv).rgb;
    imageStore(out_image, pos, vec4(clamp(base + (y - dot(base, LUMA_W)), 0.0, 1.0), 1.0));
}
"""
    open(os.path.join(out, "final.comp"), "w").write(final)
    subprocess.run(["glslangValidator", "-V", "--quiet", "--target-env", "vulkan1.1", os.path.join(out, "final.comp"),
                    "-o", os.path.join(out, "final.spv")], check=True)
open(os.path.join(out, "luts.txt"), "w").write("".join(f"{n} {w} {h}\n" for n, w, h in luts))
subprocess.run(["glslangValidator", "-V", "--quiet", "--target-env", "vulkan1.1", os.path.join(out, "ravu.comp"),
                "-o", os.path.join(out, "ravu.spv")], check=True)
print("ok", luts, "groupe", group)
