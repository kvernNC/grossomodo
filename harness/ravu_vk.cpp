// SPDX-License-Identifier: GPL-3.0-or-later
//
// Banc RAVU : exécute un shader RAVU-Zoom converti par scripts/gen_ravu_vk.py (une passe, deux tables) et le chronomètre.
//
//   ravu_vk <dossier_ravu> <entrée.rgba> <sortie.rgba> <largeur> <hauteur> [répétitions]

#include <vulkan/vulkan.h>

#include <algorithm>
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

std::vector<char> readFile(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("lecture impossible : " + path);
    return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

struct Image8 {
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> px;
};

Image8 readRgba(const std::string &path)
{
    std::vector<char> d = readFile(path);
    Image8 im;
    std::memcpy(&im.w, d.data(), 4);
    std::memcpy(&im.h, d.data() + 4, 4);
    im.px.assign(d.begin() + 8, d.end());
    if (im.px.size() != size_t(im.w) * im.h * 4)
        throw std::runtime_error("taille de .rgba incohérente");
    return im;
}

void writeRgba(const std::string &path, uint32_t w, uint32_t h, const void *px)
{
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(&w), 4);
    f.write(reinterpret_cast<const char *>(&h), 4);
    f.write(reinterpret_cast<const char *>(px), std::streamsize(size_t(w) * h * 4));
}

struct Ctx {
    VkInstance instance{};
    VkPhysicalDevice phys{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t family = 0;
    VkPhysicalDeviceMemoryProperties mem{};
    VkPhysicalDeviceProperties props{};
    uint32_t timestampBits = 0;

    uint32_t memType(uint32_t bits, VkMemoryPropertyFlags want) const
    {
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & want) == want)
                return i;
        throw std::runtime_error("pas de type de mémoire");
    }
};

struct Buffer { VkBuffer buf{}; VkDeviceMemory mem{}; void *map = nullptr; };

struct Tex {
    VkImage img{};
    VkDeviceMemory mem{};
    VkImageView view{};
    uint32_t w = 0, h = 0;
};

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

Tex makeTex(Ctx &c, uint32_t w, uint32_t h, VkFormat format, uint32_t layers, VkImageUsageFlags usage)
{
    Tex t;
    t.w = w;
    t.h = h;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    check(vkCreateImage(c.device, &ci, nullptr, &t.img), "vkCreateImage");
    VkMemoryRequirements r;
    vkGetImageMemoryRequirements(c.device, t.img, &r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = r.size;
    ai.memoryTypeIndex = c.memType(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkAllocateMemory(c.device, &ai, nullptr, &t.mem), "vkAllocateMemory");
    check(vkBindImageMemory(c.device, t.img, t.mem, 0), "vkBindImageMemory");
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = t.img;
    vi.viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    check(vkCreateImageView(c.device, &vi, nullptr, &t.view), "vkCreateImageView");
    return t;
}

void toGeneral(VkCommandBuffer cb, VkImage img, uint32_t layers)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
}

void memoryBarrier(VkCommandBuffer cb, VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags s0, VkPipelineStageFlags s1)
{
    VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    m.srcAccessMask = src;
    m.dstAccessMask = dst;
    vkCmdPipelineBarrier(cb, s0, s1, 0, 1, &m, 0, nullptr, 0, nullptr);
}



} // namespace

int main(int argc, char **argv)
{
    if (argc < 5) {
        std::fprintf(stderr, "usage : %s dossier_ravu entree.rgba sortie.rgba largeur hauteur [repetitions]\n", argv[0]);
        return 2;
    }
    try {
        const std::string dir = argv[1];
        const Image8 in = readRgba(argv[2]);
        const uint32_t outW = uint32_t(std::atoi(argv[4])), outH = uint32_t(std::atoi(argv[5]));
        const int reps = argc > 6 ? std::max(1, std::atoi(argv[6])) : 1;

        Ctx c;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        check(vkCreateInstance(&ici, nullptr, &c.instance), "vkCreateInstance");
        uint32_t n = 1;
        VkResult er = vkEnumeratePhysicalDevices(c.instance, &n, &c.phys);
        if (n == 0 || (er != VK_SUCCESS && er != VK_INCOMPLETE)) throw std::runtime_error("aucun GPU Vulkan");
        vkGetPhysicalDeviceMemoryProperties(c.phys, &c.mem);
        vkGetPhysicalDeviceProperties(c.phys, &c.props);
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, qf.data());
        for (uint32_t i = 0; i < qn; ++i)
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.family = i; c.timestampBits = qf[i].timestampValidBits; break; }
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = c.family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkPhysicalDeviceFeatures feats{};
        feats.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pEnabledFeatures = &feats;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        check(vkCreateDevice(c.phys, &dci, nullptr, &c.device), "vkCreateDevice");
        vkGetDeviceQueue(c.device, c.family, 0, &c.queue);

        // Tables : luts.txt = "nom largeur hauteur" par ligne, fichiers <nom>.bin en rgba16f.
        struct Lut { std::string name; uint32_t w, h; std::vector<char> data; Tex tex; };
        std::vector<Lut> luts;
        {
            std::vector<char> t = readFile(dir + "/luts.txt");
            std::string s(t.begin(), t.end());
            size_t p = 0;
            while (p < s.size()) {
                size_t e = s.find('\n', p);
                std::string line = s.substr(p, e - p);
                p = e == std::string::npos ? s.size() : e + 1;
                if (line.empty()) continue;
                char nm[128]; unsigned w, h;
                if (std::sscanf(line.c_str(), "%127s %u %u", nm, &w, &h) == 3)
                    luts.push_back({nm, w, h, readFile(dir + "/" + nm + ".bin"), {}});
            }
        }
        // RAVU-Lite : passe x2 à la résolution source (image rgba16f) puis réduction/recombinaison (final.spv).
        const bool lite = std::ifstream(dir + "/final.spv").good();
        Tex src = makeTex(c, in.w, in.h, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        Tex fin = makeTex(c, outW, outH, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        Tex x2 = makeTex(c, in.w * 2, in.h * 2, VK_FORMAT_R16_SFLOAT, 1, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        size_t upSize = in.px.size();
        for (Lut &l : luts) {
            l.tex = makeTex(c, l.w, l.h, VK_FORMAT_R16G16B16A16_SFLOAT, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            upSize += l.data.size();
        }
        Buffer up = makeBuffer(c, upSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        Buffer downFin = makeBuffer(c, VkDeviceSize(outW) * outH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        std::memcpy(up.map, in.px.data(), in.px.size());
        {
            size_t off = in.px.size();
            for (Lut &l : luts) { std::memcpy(static_cast<char *>(up.map) + off, l.data.data(), l.data.size()); off += l.data.size(); }
        }

        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VkSampler sampler, nearest;
        check(vkCreateSampler(c.device, &si, nullptr, &sampler), "vkCreateSampler");
        si.magFilter = si.minFilter = VK_FILTER_NEAREST;
        check(vkCreateSampler(c.device, &si, nullptr, &nearest), "vkCreateSampler");

        VkDescriptorSetLayoutBinding b[4] = {
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = 2 + uint32_t(luts.size());
        dli.pBindings = b;
        VkDescriptorSetLayout dsl;
        check(vkCreateDescriptorSetLayout(c.device, &dli, nullptr, &dsl), "vkCreateDescriptorSetLayout");
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        VkPipelineLayout layout;
        check(vkCreatePipelineLayout(c.device, &pli, nullptr, &layout), "vkCreatePipelineLayout");
        std::vector<char> spv = readFile(dir + "/ravu.spv");
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = spv.size();
        smi.pCode = reinterpret_cast<const uint32_t *>(spv.data());
        VkShaderModule module;
        check(vkCreateShaderModule(c.device, &smi, nullptr, &module), "vkCreateShaderModule");
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        cpi.layout = layout;
        VkPipeline pipe;
        check(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe), "vkCreateComputePipelines");

        VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 2;
        dpi.poolSizeCount = 2;
        dpi.pPoolSizes = ps;
        VkDescriptorPool pool;
        check(vkCreateDescriptorPool(c.device, &dpi, nullptr, &pool), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &dsl;
        VkDescriptorSet set;
        check(vkAllocateDescriptorSets(c.device, &dai, &set), "vkAllocateDescriptorSets");
        VkDescriptorImageInfo ii[4] = {{sampler, src.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, (lite ? x2 : fin).view, VK_IMAGE_LAYOUT_GENERAL}};
        for (size_t i = 0; i < luts.size(); ++i) ii[2 + i] = {lite ? nearest : sampler, luts[i].tex.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w[4]{};
        for (uint32_t k = 0; k < 2 + luts.size(); ++k) {
            w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[k].dstSet = set;
            w[k].dstBinding = k;
            w[k].descriptorCount = 1;
            w[k].descriptorType = b[k].descriptorType;
            w[k].pImageInfo = &ii[k];
        }
        vkUpdateDescriptorSets(c.device, 2 + uint32_t(luts.size()), w, 0, nullptr);

        // Passe de réduction de RAVU-Lite : 0 source, 1 sortie, 2 image x2 ; même mise en page (3 échantillonneurs max).
        VkPipeline pipe2 = VK_NULL_HANDLE;
        VkDescriptorSet set2 = VK_NULL_HANDLE;
        VkPipelineLayout layout2 = VK_NULL_HANDLE;
        if (lite) {
            VkDescriptorSetLayoutCreateInfo dli2{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dli2.bindingCount = 3;
            dli2.pBindings = b;
            VkDescriptorSetLayout dsl2;
            check(vkCreateDescriptorSetLayout(c.device, &dli2, nullptr, &dsl2), "vkCreateDescriptorSetLayout");
            VkPipelineLayoutCreateInfo pli2 = pli;
            pli2.pSetLayouts = &dsl2;
            check(vkCreatePipelineLayout(c.device, &pli2, nullptr, &layout2), "vkCreatePipelineLayout");
            std::vector<char> spv2 = readFile(dir + "/final.spv");
            VkShaderModuleCreateInfo smi2{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smi2.codeSize = spv2.size();
            smi2.pCode = reinterpret_cast<const uint32_t *>(spv2.data());
            VkShaderModule module2;
            check(vkCreateShaderModule(c.device, &smi2, nullptr, &module2), "vkCreateShaderModule");
            VkComputePipelineCreateInfo cpi2 = cpi;
            cpi2.stage.module = module2;
            cpi2.layout = layout2;
            check(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &cpi2, nullptr, &pipe2), "vkCreateComputePipelines");
            VkDescriptorSetAllocateInfo dai2 = dai;
            dai2.pSetLayouts = &dsl2;
            check(vkAllocateDescriptorSets(c.device, &dai2, &set2), "vkAllocateDescriptorSets");
            VkDescriptorImageInfo jj[3] = {{sampler, src.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, fin.view, VK_IMAGE_LAYOUT_GENERAL},
                                           {sampler, x2.view, VK_IMAGE_LAYOUT_GENERAL}};
            VkWriteDescriptorSet w2[3]{};
            for (uint32_t k = 0; k < 3; ++k) {
                w2[k] = w[k];
                w2[k].dstSet = set2;
                w2[k].pImageInfo = &jj[k];
            }
            vkUpdateDescriptorSets(c.device, 3, w2, 0, nullptr);
        }

        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 2;
        VkQueryPool queries;
        check(vkCreateQueryPool(c.device, &qpi, nullptr, &queries), "vkCreateQueryPool");
        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.queueFamilyIndex = c.family;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkCommandPool cmdPool;
        check(vkCreateCommandPool(c.device, &cpci, nullptr, &cmdPool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmdPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb;
        check(vkAllocateCommandBuffers(c.device, &cbai, &cb), "vkAllocateCommandBuffers");
        VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sub.commandBufferCount = 1;
        sub.pCommandBuffers = &cb;
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

        check(vkBeginCommandBuffer(cb, &cbi), "vkBeginCommandBuffer");
        toGeneral(cb, src.img, 1);
        toGeneral(cb, fin.img, 1);
        toGeneral(cb, x2.img, 1);
        for (Lut &l : luts) toGeneral(cb, l.tex.img, 1);
        VkBufferImageCopy reg{};
        reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        reg.imageExtent = {in.w, in.h, 1};
        vkCmdCopyBufferToImage(cb, up.buf, src.img, VK_IMAGE_LAYOUT_GENERAL, 1, &reg);
        {
            VkDeviceSize off = in.px.size();
            for (Lut &l : luts) {
                VkBufferImageCopy r2{};
                r2.bufferOffset = off;
                r2.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                r2.imageExtent = {l.w, l.h, 1};
                vkCmdCopyBufferToImage(cb, up.buf, l.tex.img, VK_IMAGE_LAYOUT_GENERAL, 1, &r2);
                off += l.data.size();
            }
        }
        check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
        check(vkQueueSubmit(c.queue, 1, &sub, VK_NULL_HANDLE), "vkQueueSubmit");
        check(vkQueueWaitIdle(c.queue), "vkQueueWaitIdle");

        const float pc[4] = {float(in.w), float(in.h), float(outW), float(outH)};
        const float pcLite[4] = {float(in.w), float(in.h), float(2 * in.w), float(2 * in.h)};
        std::vector<double> ms;
        for (int r = 0; r < reps; ++r) {
            check(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
            check(vkBeginCommandBuffer(cb, &cbi), "vkBeginCommandBuffer");
            vkCmdResetQueryPool(cb, queries, 0, 2);
            memoryBarrier(cb, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
            if (lite) {
                vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pcLite);
                vkCmdDispatch(cb, (in.w + 31) / 32, (in.h + 7) / 8, 1);
                memoryBarrier(cb, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe2);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout2, 0, 1, &set2, 0, nullptr);
                vkCmdPushConstants(cb, layout2, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
                vkCmdDispatch(cb, (outW + 7) / 8, (outH + 7) / 8, 1);
            } else {
                vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
                vkCmdDispatch(cb, (outW + 31) / 32, (outH + 7) / 8, 1);
            }
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
            memoryBarrier(cb, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            if (r == reps - 1) {
                VkBufferImageCopy rb{};
                rb.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                rb.imageExtent = {outW, outH, 1};
                vkCmdCopyImageToBuffer(cb, fin.img, VK_IMAGE_LAYOUT_GENERAL, downFin.buf, 1, &rb);
            }
            check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
            check(vkQueueSubmit(c.queue, 1, &sub, VK_NULL_HANDLE), "vkQueueSubmit");
            check(vkQueueWaitIdle(c.queue), "vkQueueWaitIdle");
            if (c.timestampBits) {
                uint64_t ts[2];
                check(vkGetQueryPoolResults(c.device, queries, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                      "vkGetQueryPoolResults");
                ms.push_back(double(ts[1] - ts[0]) * c.props.limits.timestampPeriod * 1e-6);
            }
        }
        writeRgba(argv[3], outW, outH, downFin.map);
        std::sort(ms.begin(), ms.end());
        std::printf("GPU : %s, RAVU (%s) %ux%u -> %ux%u, %d répétitions : %.3f ms (médiane)\n", c.props.deviceName, dir.c_str(),
                    in.w, in.h, outW, outH, reps, ms.empty() ? 0.0 : ms[ms.size() / 2]);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "erreur : %s\n", e.what());
        return 1;
    }
}
