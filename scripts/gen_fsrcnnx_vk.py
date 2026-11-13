#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Génère FSRCNNX en shaders de calcul Vulkan (GLSL) à partir du fichier mpv, poids écrits en dur.

Les passes mpv vont par paires (les deux moitiés de 4 canaux d'une couche de 8) : chaque paire devient un seul
dispatch qui écrit une carte de caractéristiques 8 canaux (image2DArray RGBA16F à 2 couches, comme d2-webgpu).
La convolution sous-pixel et l'agrégation deviennent un dispatch qui écrit directement le bloc 2x2 de sortie, avec la
couleur recombinée : rgb_bilinéaire + (Y_réseau - Y_bilinéaire). Une dernière passe réduit l'image x2 à la taille
voulue (Lanczos-2 adapté à la réduction), pour les facteurs non entiers (800x450 -> 1600x900 -> 1280x720).

Sorties dans build/<nom>/ : pass_NN.comp, pass_NN.spv (FP32 et _fp16) et plan.txt, lu par build/fsrcnnx_vk.

    python3 -I scripts/gen_fsrcnnx_vk.py third_party/fsrcnnx/FSRCNNX_x2_8-0-4-1.glsl build/fsrcnnx8
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsrcnnx  # noqa: E402

HEADER = """#version 450
// Généré par scripts/gen_fsrcnnx_vk.py depuis {src}.
// FSRCNNX (igv), réseau et poids sous LGPL-3.0-or-later ; voir third_party/fsrcnnx/.
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_GOOGLE_include_directive : enable
layout(local_size_x = 8, local_size_y = 8) in;
#ifdef USE_FP16
#define V4 f16vec4
#define M4 f16mat4
#define SC float16_t
#else
#define V4 vec4
#define M4 mat4
#define SC float
#endif
layout(binding = 0) uniform sampler2D srcRgb;                      // image du jeu (RGBA8)
layout(binding = 1, rgba16f) uniform readonly image2DArray inMap;   // carte d'entrée, 8 canaux
layout(binding = 2, rgba16f) uniform readonly image2DArray inSkip;  // carte de la connexion résiduelle
layout(binding = 3, rgba16f) uniform writeonly image2DArray outMap; // carte de sortie
layout(binding = 4) uniform writeonly image2D outRgb;               // image de sortie (sans format : BGRA dans gamescope)
layout(binding = 5, rgba8) uniform readonly image2D inRgb;          // image x2 (passe de réduction)
layout(binding = 6) uniform sampler2D inRgbTex;                     // la même, échantillonnée (filtrage bilinéaire)

const vec3 LUMA709 = vec3(0.2126, 0.7152, 0.0722);
ivec2 lrSize() {{ return textureSize(srcRgb, 0); }}
ivec2 clampLr(ivec2 p) {{ return clamp(p, ivec2(0), lrSize() - 1); }}
V4 mapAt(ivec2 p, int layer) {{ return V4(imageLoad(inMap, ivec3(clampLr(p), layer))); }}
V4 skipAt(ivec2 p, int layer) {{ return V4(imageLoad(inSkip, ivec3(clampLr(p), layer))); }}
float lumaAt(ivec2 p) {{ return dot(texelFetch(srcRgb, clampLr(p), 0).rgb, LUMA709); }}
V4 prelu(V4 x, V4 a) {{ return max(x, V4(0)) + a * min(x, V4(0)); }}
"""


def f(x):
    return f"{x:.9g}"


def vec(ws):
    return "V4(" + ",".join(f(x) for x in ws) + ")"


def mat(cols):
    # cols : (4, 4), une ligne = une colonne de la mat4 GLSL (ordre du fichier mpv)
    return "M4(" + ",".join(f(x) for x in cols.reshape(-1)) + ")"


def half_body(p, name_loc, out_var):
    lines = [f"    V4 {out_var} = {vec(p.bias)};"]
    for src, dx, dy, w in p.terms:
        if src == "LUMA":
            lines.append(f"    {out_var} += {vec(w)} * luma[{(dx + 2) * 5 + (dy + 2)}];")
        else:
            tex, layer = name_loc[src]
            lines.append(f"    {out_var} += {mat(w)} * {tex}[{(dx + 1) * 3 + (dy + 1)}][{layer}];")
    if p.residual:
        lines.append(f"    {out_var} += skp[{name_loc[p.residual][1]}];")
    if p.prelu is not None:
        lines.append(f"    {out_var} = prelu({out_var}, {vec(p.prelu)});")
    return lines


def loads(tex_var, fn, radius):
    """Lit une fois le voisinage (2r+1)^2 x 2 couches dans un tableau local : tex_var[t][couche]."""
    n = (2 * radius + 1) ** 2
    out = [f"    V4 {tex_var}[{n}][2];"]
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            t = (dx + radius) * (2 * radius + 1) + (dy + radius)
            for layer in range(2):
                out.append(f"    {tex_var}[{t}][{layer}] = {fn}(p + ivec2({dx}, {dy}), {layer});")
    return out


def generate(src_path, out_dir, downscale="cr9"):
    passes = fsrcnnx.parse(src_path)
    os.makedirs(out_dir, exist_ok=True)
    plan, shaders = [], []
    name_loc = {}            # nom mpv -> (variable du shader, couche) ; mis à jour après chaque dispatch
    where = {}               # nom mpv -> texture réelle (FEAT, A, B)
    i = 0
    while i < len(passes):
        p = passes[i]
        if p.desc.startswith("sub-pixel"):
            agg = passes[i + 1]
            assert agg.aggregate
            src_tex = where[p.terms[0][0]]
            loc = {n: ("m", name_loc[n][1]) for n in {t[0] for t in p.terms}}
            body = loads("m", "mapAt", 1) + half_body(p, loc, "res")
            code = HEADER.format(src=os.path.basename(src_path)) + """
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, lrSize()))) return;
""" + "\n".join(body) + """
    // agrégation (mpv) : sortie (2x + ix, 2y + iy) = canal ix * 2 + iy ; couleur : bilinéaire + (Y_réseau - Y_bilinéaire)
    vec2 outSize = vec2(lrSize() * 2);
    for (int ix = 0; ix < 2; ++ix) {
        for (int iy = 0; iy < 2; ++iy) {
            ivec2 o = p * 2 + ivec2(ix, iy);
            vec3 bil = textureLod(srcRgb, (vec2(o) + 0.5) / outSize, 0.0).rgb;
            float y = float(res[ix * 2 + iy]);
            imageStore(outRgb, o, vec4(clamp(bil + (y - dot(bil, LUMA709)), 0.0, 1.0), 1.0));
        }
    }
}
"""
            shaders.append(code)
            plan.append((src_tex, "none", "OUT2X", "lr"))
            i += 2
            continue
        q = passes[i + 1]  # la moitié 4-7 de la même couche
        inputs = {t[0] for t in p.terms} | {t[0] for t in q.terms}
        skip = p.residual
        if inputs == {"LUMA"}:
            in_tex, body = "none", ["    SC luma[25];"] + [
                f"    luma[{(dx + 2) * 5 + (dy + 2)}] = SC(lumaAt(p + ivec2({dx}, {dy})));"
                for dx in range(-2, 3) for dy in range(-2, 3)]
            loc = {}
        else:
            in_tex = {where[n] for n in inputs}
            assert len(in_tex) == 1, inputs
            in_tex = in_tex.pop()
            loc = {n: ("m", name_loc[n][1]) for n in inputs}
            radius = max(max(abs(t[1]), abs(t[2])) for t in p.terms)
            body = loads("m", "mapAt", radius)
            if radius == 0:  # 1x1 : l'indice de tap vaut 0
                body = loads("m", "mapAt", 0)
        skip_tex = "none"
        if skip:
            skip_tex = where[skip]
            body += ["    V4 skp[2];", "    skp[0] = skipAt(p, 0);", "    skp[1] = skipAt(p, 1);"]
            loc[skip] = ("skp", name_loc[skip][1])
            loc[q.residual] = ("skp", name_loc[q.residual][1])
        # indices de tap : rayon 1 -> 3x3 ; rayon 0 -> 1x1 (on réécrit l'indice)
        b0, b1 = half_body(p, loc, "r0"), half_body(q, loc, "r1")
        if in_tex != "none" and max(max(abs(t[1]), abs(t[2])) for t in p.terms) == 0:
            b0 = [s.replace("m[4]", "m[0]") for s in b0]
            b1 = [s.replace("m[4]", "m[0]") for s in b1]
        # une texture reste occupée tant qu'un nom qu'elle contient est lu par une passe suivante
        # (un nom réécrit plus loin avant d'être relu ne compte plus : mpv réutilise MODEL21, MODEL1...)
        later, redefined = set(), {p.save, q.save}
        for r in passes[i + 2:]:
            reads = {t[0] for t in r.terms} | ({r.residual} if r.residual else set())
            later |= reads - redefined
            if r.save:
                redefined.add(r.save)
        busy = {in_tex, skip_tex} | {where[n] for n in later if n in where}
        out_tex = next(t for t in ("FEAT", "A", "B") if t not in busy and
                       (in_tex != "none" or t == "FEAT"))
        code = HEADER.format(src=os.path.basename(src_path)) + """
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, lrSize()))) return;
""" + "\n".join(body + b0 + b1) + """
    imageStore(outMap, ivec3(p, 0), vec4(r0));
    imageStore(outMap, ivec3(p, 1), vec4(r1));
}
"""
        shaders.append(code)
        plan.append((in_tex, skip_tex, out_tex, "lr"))
        where[p.save], where[q.save] = out_tex, out_tex
        name_loc[p.save], name_loc[q.save] = (None, 0), (None, 1)
        i += 2

    final = {"cubic": DOWNSCALE, "cr9": DOWNSCALE_CR9, "bilinear": DOWNSCALE_BILINEAR}[downscale]
    shaders.append(HEADER.format(src=os.path.basename(src_path)) + final)
    plan.append(("none", "none", "FINAL", "final"))

    with open(os.path.join(out_dir, "plan.txt"), "w") as fp:
        for n, (code, (a, b, c, g)) in enumerate(zip(shaders, plan)):
            base = os.path.join(out_dir, f"pass_{n:02d}")
            with open(base + ".comp", "w") as sf:
                sf.write(code)
            for suffix, defs in (("", []), ("_fp16", ["-DUSE_FP16"])):
                subprocess.run(["glslangValidator", "-V", "--target-env", "vulkan1.2", *defs,
                                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "gamescope"),
                                base + ".comp",
                                "-o", base + suffix + ".spv", "--quiet"], check=True)
            fp.write(f"pass_{n:02d} {a} {b} {c} {g}\n")
    print(f"{len(shaders)} dispatches -> {out_dir}")


# Variantes de réduction de l'image x2 vers la taille finale, choisies par --downscale.
DOWNSCALE_CR9 = """
// Catmull-Rom 4x4 en 9 lectures bilinéaires (les deux poids centraux de chaque axe regroupés en une lecture filtrée).
// Noyau à l'échelle de l'image x2 : pour une réduction de 0,8, le repliement reste faible.
// SPLIT = 1 (constante de spécialisation) : écran partagé, seule la moitié droite est écrite, avec un trait au milieu.
// strength (constante poussée) : 1 = FSRCNNX seul ; < 1 = FSR 1 (EASU, même code FidelityFX que gamescope, calculé ici
// sur l'image du jeu) + strength x (FSRCNNX - FSR 1), pour garder une partie du détail en atténuant le scintillement.
layout(constant_id = 0) const int SPLIT = 0;
layout(push_constant) uniform PushConstants { float strength; } pc;
#define A_GPU 1
#define A_GLSL 1
#include "ffx_a.h"
#define FSR_EASU_F 1
AF4 FsrEasuRF(AF2 p) { return AF4(textureGather(srcRgb, p, 0)); }
AF4 FsrEasuGF(AF2 p) { return AF4(textureGather(srcRgb, p, 1)); }
AF4 FsrEasuBF(AF2 p) { return AF4(textureGather(srcRgb, p, 2)); }
#include "ffx_fsr1.h"
void main() {
    ivec2 o = ivec2(gl_GlobalInvocationID.xy);
    ivec2 outSize = imageSize(outRgb);
    if (any(greaterThanEqual(o, outSize))) return;
    if (SPLIT != 0) {
        int mid = outSize.x / 2;
        if (o.x < mid) return;
        if (o.x < mid + 2) { imageStore(outRgb, o, vec4(1.0)); return; }
    }
    vec2 inSize = vec2(textureSize(inRgbTex, 0));
    vec2 pos = (vec2(o) + 0.5) * inSize / vec2(outSize);
    vec2 t = floor(pos - 0.5) + 0.5;
    vec2 f = pos - t, f2 = f * f, f3 = f2 * f;
    vec2 w0 = -0.5 * f3 + f2 - 0.5 * f;
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
    vec2 w3 = 0.5 * f3 - 0.5 * f2;
    vec2 w12 = w1 + w2;
    vec2 t0 = (t - 1.0) / inSize, t3 = (t + 2.0) / inSize, t12 = (t + w2 / w12) / inSize;
    vec3 c = (textureLod(inRgbTex, vec2(t0.x, t0.y), 0.0).rgb * w0.x + textureLod(inRgbTex, vec2(t12.x, t0.y), 0.0).rgb * w12.x
            + textureLod(inRgbTex, vec2(t3.x, t0.y), 0.0).rgb * w3.x) * w0.y
           + (textureLod(inRgbTex, vec2(t0.x, t12.y), 0.0).rgb * w0.x + textureLod(inRgbTex, vec2(t12.x, t12.y), 0.0).rgb * w12.x
            + textureLod(inRgbTex, vec2(t3.x, t12.y), 0.0).rgb * w3.x) * w12.y
           + (textureLod(inRgbTex, vec2(t0.x, t3.y), 0.0).rgb * w0.x + textureLod(inRgbTex, vec2(t12.x, t3.y), 0.0).rgb * w12.x
            + textureLod(inRgbTex, vec2(t3.x, t3.y), 0.0).rgb * w3.x) * w3.y;
    if (pc.strength < 0.999) {
        vec2 lr = vec2(textureSize(srcRgb, 0));
        AU4 c1, c2, c3, c4;
        FsrEasuCon(c1, c2, c3, c4, lr.x, lr.y, lr.x, lr.y, float(outSize.x), float(outSize.y));
        AF3 easu;
        FsrEasuF(easu, AU2(o), c1, c2, c3, c4);
        c = mix(vec3(easu), c, clamp(pc.strength, 0.0, 1.0));
    }
    imageStore(outRgb, o, vec4(clamp(c, 0.0, 1.0), 1.0));
}
"""

DOWNSCALE_BILINEAR = """
// Bilinéaire : une lecture filtrée par pixel.
void main() {
    ivec2 o = ivec2(gl_GlobalInvocationID.xy);
    ivec2 outSize = imageSize(outRgb);
    if (any(greaterThanEqual(o, outSize))) return;
    imageStore(outRgb, o, vec4(textureLod(inRgbTex, (vec2(o) + 0.5) / vec2(outSize), 0.0).rgb, 1.0));
}
"""


# Réduction de l'image x2 (inRgb) à la taille de outRgb : Lanczos-2 dont le support suit le facteur de réduction.
DOWNSCALE = """
// Réduction séparable : un noyau cubique de Catmull-Rom (polynomial, sans sinus) dont le support suit le facteur de
// réduction. Mesuré sur le Deck, l'ancienne version (Lanczos-2, deux sinus par échantillon) coûtait 1,8 ms.
float cubic(float x) {
    x = abs(x);
    if (x < 1.0) return (1.5 * x - 2.5) * x * x + 1.0;
    if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    return 0.0;
}
void main() {
    ivec2 o = ivec2(gl_GlobalInvocationID.xy);
    ivec2 outSize = imageSize(outRgb), inSize = imageSize(inRgb);
    if (any(greaterThanEqual(o, outSize))) return;
    vec2 scale = max(vec2(inSize) / vec2(outSize), vec2(1.0));   // > 1 : réduction
    vec2 c = (vec2(o) + 0.5) * vec2(inSize) / vec2(outSize) - 0.5;
    vec2 support = 2.0 * scale;
    ivec2 lo = ivec2(floor(c - support)) + 1, hi = ivec2(floor(c + support));
    float wx[8];
    float wxs = 0.0;
    for (int x = lo.x; x <= min(hi.x, lo.x + 7); ++x) { wx[x - lo.x] = cubic((float(x) - c.x) / scale.x); wxs += wx[x - lo.x]; }
    vec3 sum = vec3(0.0);
    float wys = 0.0;
    for (int y = lo.y; y <= min(hi.y, lo.y + 7); ++y) {
        float wy = cubic((float(y) - c.y) / scale.y);
        vec3 row = vec3(0.0);
        for (int x = lo.x; x <= min(hi.x, lo.x + 7); ++x)
            row += wx[x - lo.x] * imageLoad(inRgb, clamp(ivec2(x, y), ivec2(0), inSize - 1)).rgb;
        sum += wy * row;
        wys += wy;
    }
    imageStore(outRgb, o, vec4(clamp(sum / (wxs * wys), 0.0, 1.0), 1.0));
}
"""

if __name__ == "__main__":
    generate(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cr9")
