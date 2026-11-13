// SPDX-License-Identifier: GPL-3.0-or-later
//
// Banc de test du PoC-A : exécute un shader d'agrandissement avec l'interface de la passe EASU de gamescope
// (liaison 0 : bloc de constantes FsrEasuCon + matrice de couleur, liaison 1 : image de sortie, liaison 3 :
// échantillonneurs, image du jeu dans la case 0 ; groupes de 64 invocations, 16x16 pixels par groupe).
//
//   harness <shader.spv> <entrée.rgba> <sortie.rgba> <largeur_sortie> <hauteur_sortie> [répétitions] [rcas=<stops>]
//
// Avec rcas=<stops>, la liaison 0 reçoit la constante de RCAS (u_c1 = exp2(-stops) en bits de float, comme
// FsrRcasCon) au lieu des constantes d'EASU : pour shaders/cs_rcas.comp.
//
// Fichiers .rgba : en-tête de deux uint32 (largeur, hauteur) puis les pixels RGBA8. Les conversions PNG et les
// mesures sont faites par scripts/poc_a.py.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(VkResult r, const char *what)
{
    if (r != VK_SUCCESS)
        throw std::runtime_error(std::string(what) + " : VkResult " + std::to_string(r));
}

std::vector<char> readFile(const char *path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error(std::string("lecture impossible : ") + path);
    return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

// FsrEasuCon de ffx_fsr1.h (AMD FidelityFX, MIT), tel que gamescope l'appelle : viewport d'entrée = taille d'entrée.
uint32_t asU(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

void fsrEasuCon(uint32_t con[16], float inX, float inY, float outX, float outY)
{
    const float vpX = inX, vpY = inY, szX = inX, szY = inY;
    con[0] = asU(vpX / outX);
    con[1] = asU(vpY / outY);
    con[2] = asU(0.5f * vpX / outX - 0.5f);
    con[3] = asU(0.5f * vpY / outY - 0.5f);
    con[4] = asU(1.0f / szX);
    con[5] = asU(1.0f / szY);
    con[6] = asU(1.0f / szX);
    con[7] = asU(-1.0f / szY);
    con[8] = asU(-1.0f / szX);
    con[9] = asU(2.0f / szY);
    con[10] = asU(1.0f / szX);
    con[11] = asU(2.0f / szY);
    con[12] = asU(0.0f / szX);
    con[13] = asU(4.0f / szY);
    con[14] = 0;
    con[15] = 0;
}

struct Image {
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> px;
};

Image readRgba(const char *path)
{
    std::vector<char> d = readFile(path);
    Image im;
    std::memcpy(&im.w, d.data(), 4);
    std::memcpy(&im.h, d.data() + 4, 4);
    im.px.assign(d.begin() + 8, d.end());
    if (im.px.size() != size_t(im.w) * im.h * 4)
        throw std::runtime_error("taille de .rgba incohérente");
    return im;
}

void writeRgba(const char *path, const Image &im)
{
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(&im.w), 4);
    f.write(reinterpret_cast<const char *>(&im.h), 4);
    f.write(reinterpret_cast<const char *>(im.px.data()), std::streamsize(im.px.size()));
}

struct Ctx {
    VkInstance instance{};
    VkPhysicalDevice phys{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t family = 0;
    VkPhysicalDeviceMemoryProperties mem{};

    uint32_t memType(uint32_t bits, VkMemoryPropertyFlags want) const
    {
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & want) == want)
                return i;
        throw std::runtime_error("pas de type de mémoire");
    }
};

struct Buffer { VkBuffer buf{}; VkDeviceMemory mem{}; void *map = nullptr; };
struct Img { VkImage img{}; VkDeviceMemory mem{}; VkImageView view{}; };

Buffer makeBuffer(Ctx &c, VkDeviceSize size, VkBufferUsageFlags usage)
{
    Buffer b;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    check(vkCreateBuffer(c.device, &bi, nullptr, &b.buf), "vkCreateBuffer");
    VkMemoryRequirements r;
    vkGetBufferMemoryRequirements(c.device, b.buf, &r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = r.size;
    ai.memoryTypeIndex = c.memType(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkAllocateMemory(c.device, &ai, nullptr, &b.mem), "vkAllocateMemory");
    check(vkBindBufferMemory(c.device, b.buf, b.mem, 0), "vkBindBufferMemory");
    check(vkMapMemory(c.device, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map), "vkMapMemory");
    return b;
}

Img makeImage(Ctx &c, uint32_t w, uint32_t h, VkImageUsageFlags usage)
{
    Img i;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    check(vkCreateImage(c.device, &ci, nullptr, &i.img), "vkCreateImage");
    VkMemoryRequirements r;
    vkGetImageMemoryRequirements(c.device, i.img, &r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = r.size;
    ai.memoryTypeIndex = c.memType(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkAllocateMemory(c.device, &ai, nullptr, &i.mem), "vkAllocateMemory");
    check(vkBindImageMemory(c.device, i.img, i.mem, 0), "vkBindImageMemory");
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = i.img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = ci.format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(c.device, &vi, nullptr, &i.view), "vkCreateImageView");
    return i;
}

void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst,
             VkPipelineStageFlags s0, VkPipelineStageFlags s1)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, s0, s1, 0, 0, nullptr, 0, nullptr, 1, &b);
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 6) {
        std::fprintf(stderr, "usage : %s shader.spv entree.rgba sortie.rgba largeur hauteur [repetitions]\n", argv[0]);
        return 2;
    }
    try {
        const std::vector<char> spv = readFile(argv[1]);
        const Image in = readRgba(argv[2]);
        const uint32_t outW = uint32_t(std::atoi(argv[4])), outH = uint32_t(std::atoi(argv[5]));
        const int reps = argc > 6 ? std::atoi(argv[6]) : 1;
        bool rcas = false;
        float rcasStops = 0.0f;
        if (argc > 7 && std::strncmp(argv[7], "rcas=", 5) == 0) {
            rcas = true;
            rcasStops = std::strtof(argv[7] + 5, nullptr);
        }

        Ctx c;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        check(vkCreateInstance(&ici, nullptr, &c.instance), "vkCreateInstance");
        uint32_t n = 1;
        VkResult er = vkEnumeratePhysicalDevices(c.instance, &n, &c.phys);
        if (n == 0 || (er != VK_SUCCESS && er != VK_INCOMPLETE))
            throw std::runtime_error("aucun GPU Vulkan");
        vkGetPhysicalDeviceMemoryProperties(c.phys, &c.mem);
        uint32_t tsBits = 0;
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(c.phys, &props);
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, qf.data());
        for (uint32_t i = 0; i < qn; ++i)
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.family = i; tsBits = qf[i].timestampValidBits; break; }
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = c.family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        // Comme gamescope : écriture d'images de stockage sans format déclaré dans le shader.
        VkPhysicalDeviceFeatures feats{};
        feats.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f12.scalarBlockLayout = VK_TRUE;
        f12.shaderFloat16 = VK_TRUE;
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        f11.storageBuffer16BitAccess = VK_FALSE;
        f11.pNext = &f12;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &f11;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.pEnabledFeatures = &feats;
        check(vkCreateDevice(c.phys, &dci, nullptr, &c.device), "vkCreateDevice");
        vkGetDeviceQueue(c.device, c.family, 0, &c.queue);

        // Ressources : image d'entrée (échantillonnée), image de sortie (stockage), tampons d'envoi et de lecture.
        Img src = makeImage(c, in.w, in.h, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        Img dst = makeImage(c, outW, outH, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        Buffer up = makeBuffer(c, in.px.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memcpy(up.map, in.px.data(), in.px.size());
        Buffer down = makeBuffer(c, VkDeviceSize(outW) * outH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        // Bloc de constantes de la liaison 0 : 4 uvec4 + mat3x4 (disposition scalaire), comme EasuPushData_t.
        struct { uint32_t con[16]; float ctm[12]; } ubo{};
        if (rcas)
            ubo.con[0] = asU(std::exp2(-rcasStops));
        else
            fsrEasuCon(ubo.con, float(in.w), float(in.h), float(outW), float(outH));
        const float ident[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        std::memcpy(ubo.ctm, ident, sizeof ident);
        Buffer uniforms = makeBuffer(c, sizeof ubo, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        std::memcpy(uniforms.map, &ubo, sizeof ubo);

        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VkSampler sampler;
        check(vkCreateSampler(c.device, &si, nullptr, &sampler), "vkCreateSampler");

        // Disposition des liaisons de gamescope utilisées par EASU : 0 (constantes), 1 (sortie), 3 (16 échantillonneurs).
        const uint32_t kSamplerSlots = 16;
        VkDescriptorSetLayoutBinding b[3] = {
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kSamplerSlots, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = 3;
        dli.pBindings = b;
        VkDescriptorSetLayout dsl;
        check(vkCreateDescriptorSetLayout(c.device, &dli, nullptr, &dsl), "vkCreateDescriptorSetLayout");
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
        VkPipelineLayout layout;
        check(vkCreatePipelineLayout(c.device, &pli, nullptr, &layout), "vkCreatePipelineLayout");

        VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                                      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kSamplerSlots}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 3;
        dpi.pPoolSizes = ps;
        VkDescriptorPool pool;
        check(vkCreateDescriptorPool(c.device, &dpi, nullptr, &pool), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &dsl;
        VkDescriptorSet set;
        check(vkAllocateDescriptorSets(c.device, &dai, &set), "vkAllocateDescriptorSets");

        VkDescriptorBufferInfo ubi{uniforms.buf, 0, sizeof ubo};
        VkDescriptorImageInfo oii{VK_NULL_HANDLE, dst.view, VK_IMAGE_LAYOUT_GENERAL};
        std::vector<VkDescriptorImageInfo> sii(kSamplerSlots, {sampler, src.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        VkWriteDescriptorSet w[3]{};
        for (auto &x : w) x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, x.dstSet = set;
        w[0].dstBinding = 0, w[0].descriptorCount = 1, w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, w[0].pBufferInfo = &ubi;
        w[1].dstBinding = 1, w[1].descriptorCount = 1, w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, w[1].pImageInfo = &oii;
        w[2].dstBinding = 3, w[2].descriptorCount = kSamplerSlots, w[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, w[2].pImageInfo = sii.data();
        vkUpdateDescriptorSets(c.device, 3, w, 0, nullptr);

        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = spv.size();
        smi.pCode = reinterpret_cast<const uint32_t *>(spv.data());
        VkShaderModule module;
        check(vkCreateShaderModule(c.device, &smi, nullptr, &module), "vkCreateShaderModule");
        // Constante de spécialisation 4 (c_colorspaceMask) : calque 0 en sRGB, comme un jeu SDR.
        const uint32_t colorspaceMask = 1;
        VkSpecializationMapEntry me{4, 0, 4};
        VkSpecializationInfo spec{1, &me, 4, &colorspaceMask};
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", &spec};
        cpi.layout = layout;
        VkPipeline pipeline;
        check(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline), "vkCreateComputePipelines");

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.queueFamilyIndex = c.family;
        VkCommandPool cmdPool;
        check(vkCreateCommandPool(c.device, &cpci, nullptr, &cmdPool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmdPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb;
        check(vkAllocateCommandBuffers(c.device, &cbai, &cb), "vkAllocateCommandBuffers");

        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 2;
        VkQueryPool queries;
        check(vkCreateQueryPool(c.device, &qpi, nullptr, &queries), "vkCreateQueryPool");

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
        vkCmdResetQueryPool(cb, queries, 0, 2);
        barrier(cb, src.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy reg{};
        reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        reg.imageExtent = {in.w, in.h, 1};
        vkCmdCopyBufferToImage(cb, up.buf, src.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
        barrier(cb, src.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        barrier(cb, dst.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        // Même grille que gamescope : 16x16 pixels par groupe.
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
        vkCmdDispatch(cb, (outW + 15) / 16, (outH + 15) / 16, 1);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
        barrier(cb, dst.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        reg.imageExtent = {outW, outH, 1};
        vkCmdCopyImageToBuffer(cb, dst.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, down.buf, 1, &reg);
        check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");

        VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sub.commandBufferCount = 1;
        sub.pCommandBuffers = &cb;
        double best = 1e30;
        std::vector<double> gpu;
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            check(vkQueueSubmit(c.queue, 1, &sub, VK_NULL_HANDLE), "vkQueueSubmit");
            check(vkQueueWaitIdle(c.queue), "vkQueueWaitIdle");
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            best = ms < best ? ms : best;
            if (tsBits) {
                uint64_t ts[2];
                check(vkGetQueryPoolResults(c.device, queries, 0, 2, sizeof ts, ts, 8,
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults");
                gpu.push_back(double(ts[1] - ts[0]) * props.limits.timestampPeriod * 1e-6);
            }
        }
        double gpuMed = -1;
        if (!gpu.empty()) {
            std::sort(gpu.begin(), gpu.end());
            gpuMed = gpu[gpu.size() / 2];
        }

        Image out;
        out.w = outW;
        out.h = outH;
        out.px.resize(size_t(outW) * outH * 4);
        std::memcpy(out.px.data(), down.map, out.px.size());
        writeRgba(argv[3], out);
        std::printf("%s %ux%u -> %ux%u : GPU %.3f ms (médiane, timestamps), soumission complète %.2f ms (%s)\n", argv[1],
                    in.w, in.h, outW, outH, gpuMed, best, props.deviceName);
        vkDeviceWaitIdle(c.device);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "erreur : %s\n", e.what());
        return 1;
    }
}
