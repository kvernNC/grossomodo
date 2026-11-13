// SPDX-License-Identifier: GPL-3.0-or-later
// Traduit un DXBC LS1 en SPIR-V Vulkan avec la convention de bindings de MAKO :
// CBV i -> binding i, sampler i -> 16+i, SRV i -> 32+i, UAV i -> 48+i (set 0).
// usage: translate in.dxbc out.spv nSrv nUav nCbv nSampler
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <vkd3d_shader.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
int main(int c, char** v) {
    void* h = dlopen(getenv("VKD3D_SHADER_LIB"), RTLD_NOW);
    if (!h) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    int (*compile)(const struct vkd3d_shader_compile_info*, struct vkd3d_shader_code*, char**) = dlsym(h, "vkd3d_shader_compile");
    FILE* f = fopen(v[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    void* buf = malloc(n); fread(buf, 1, n, f); fclose(f);
    int counts[4] = {atoi(v[3]), atoi(v[4]), atoi(v[5]), atoi(v[6])};
    struct vkd3d_shader_resource_binding b[16]; unsigned nb = 0;
    const int types[4] = {VKD3D_SHADER_DESCRIPTOR_TYPE_SRV, VKD3D_SHADER_DESCRIPTOR_TYPE_UAV,
                          VKD3D_SHADER_DESCRIPTOR_TYPE_CBV, VKD3D_SHADER_DESCRIPTOR_TYPE_SAMPLER};
    const unsigned base[4] = {32, 48, 0, 16};
    for (int t = 0; t < 4; ++t) for (int i = 0; i < counts[t]; ++i) {
        b[nb].type = types[t]; b[nb].register_space = 0; b[nb].register_index = i;
        b[nb].shader_visibility = VKD3D_SHADER_VISIBILITY_COMPUTE;
        b[nb].flags = t < 2 ? VKD3D_SHADER_BINDING_FLAG_IMAGE : (t == 2 ? VKD3D_SHADER_BINDING_FLAG_BUFFER : 0);
        b[nb].binding.set = 0; b[nb].binding.binding = base[t] + i; b[nb].binding.count = 1; ++nb;
    }
    struct vkd3d_shader_spirv_target_info tgt = {VKD3D_SHADER_STRUCTURE_TYPE_SPIRV_TARGET_INFO};
    tgt.environment = VKD3D_SHADER_SPIRV_ENVIRONMENT_VULKAN_1_0;
    struct vkd3d_shader_interface_info ii = {VKD3D_SHADER_STRUCTURE_TYPE_INTERFACE_INFO, &tgt};
    ii.bindings = b; ii.binding_count = nb;
    struct vkd3d_shader_compile_info ci = {VKD3D_SHADER_STRUCTURE_TYPE_COMPILE_INFO, &ii};
    ci.source.code = buf; ci.source.size = n; ci.source_type = VKD3D_SHADER_SOURCE_DXBC_TPF;
    ci.target_type = VKD3D_SHADER_TARGET_SPIRV_BINARY; ci.log_level = VKD3D_SHADER_LOG_WARNING; ci.source_name = v[1];
    struct vkd3d_shader_code out = {0}; char* msg = NULL;
    int r = compile(&ci, &out, &msg);
    if (msg && *msg) fprintf(stderr, "%s", msg);
    if (r) { fprintf(stderr, "échec %d\n", r); return 1; }
    f = fopen(v[2], "wb"); fwrite(out.code, 1, out.size, f); fclose(f);
    return 0;
}
