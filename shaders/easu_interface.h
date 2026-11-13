// SPDX-License-Identifier: GPL-3.0-or-later
//
// Interface commune : un agrandisseur qui prend la place de la passe EASU (FSR 1) de gamescope.
// Mêmes liaisons, mêmes constantes et même grille que third_party/gamescope/cs_easu.comp. Le shader qui inclut ce
// fichier définit `vec3 upscale(ivec2 pos, vec2 srcSize, vec2 dstSize)` avant de l'inclure.

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

#include "descriptor_set.h"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, scalar)
uniform layers_t {
    uvec4 c1, c2, c3, c4;   // FsrEasuCon
    mat3x4 u_ctm;
};

const float u_linearToNits = 0.0f;
const float u_nitsToLinear = 0.0f;

#include "colorimetry.h"

// FsrEasuCon : con0.xy = taille d'entrée / taille de sortie, con1.xy = 1 / taille d'entrée.
vec2 inputSize()  { return vec2(1.0) / uintBitsToFloat(c2.xy); }
vec2 outputSize() { return floor(inputSize() / uintBitsToFloat(c1.xy) + vec2(0.5)); }

// Comme cs_easu.comp : le scRGB sort en PQ pour RCAS.
vec3 easuOutput(vec3 color)
{
    if (get_layer_colorspace(0) != colorspace_scRGB)
        return color;
    return scRGBEncodingToPQ((vec4(color, 1.0f) * u_ctm).rgb);
}

vec3 upscale(ivec2 pos, vec2 srcSize, vec2 dstSize);

void upscalePass(uvec2 pos, vec2 srcSize, vec2 dstSize)
{
    if (any(greaterThanEqual(pos, uvec2(dstSize))))
        return;
    imageStore(dst, ivec2(pos), vec4(easuOutput(upscale(ivec2(pos), srcSize, dstSize)), 1));
}

void main()
{
    // 64 invocations, 4 pixels chacune, 16x16 pixels par groupe (la grille de gamescope pour EASU).
    vec2 srcSize = inputSize();
    vec2 dstSize = outputSize();
    uvec2 pos = uvec2(gl_LocalInvocationID.x & 7u, gl_LocalInvocationID.x >> 3u)
              + uvec2(gl_WorkGroupID.x << 4u, gl_WorkGroupID.y << 4u);
    upscalePass(pos, srcSize, dstSize);
    pos.x += 8u;
    upscalePass(pos, srcSize, dstSize);
    pos.y += 8u;
    upscalePass(pos, srcSize, dstSize);
    pos.x -= 8u;
    upscalePass(pos, srcSize, dstSize);
}
