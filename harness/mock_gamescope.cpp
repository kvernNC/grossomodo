// SPDX-License-Identifier: GPL-3.0-or-later
//
// Imitation de la séquence Vulkan de gamescope autour de FSR 1 (src/rendervulkan.cpp, CVulkanCmdBuffer::dispatch),
// pour tester VK_LAYER_GROSSOMODO_upscale sans gamescope :
//   - disposition de descripteurs de gamescope (0 : constantes, 1-2 : sorties, 3-4 : 16 échantillonneurs, 5-6 : LUT) ;
//   - image de sortie temporaire en B8G8R8A8 (DRM_FORMAT_ARGB8888) écrite par EASU, puis lue par RCAS ;
//   - pour chaque dispatch : vkCmdBindPipeline, barrières, vkUpdateDescriptorSets sur un descripteur neuf,
//     vkCmdBindDescriptorSets, vkCmdDispatch (grille 16x16 comme gamescope).
//
//   mock_gamescope <easu.spv> <rcas.spv> <entrée.rgba> <largeur> <hauteur> <sortie_easu.rgba> <sortie_rcas.rgba> [images]
//
// Avec la couche : VK_LAYER_PATH=<dossier du manifeste> VK_INSTANCE_LAYERS=VK_LAYER_GROSSOMODO_upscale GROSSOMODO_FORCE=1.

#include <vulkan/vulkan.h>

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
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + " : VkResult " + std::to_string(r));
}

std::vector<char> readFile(const char *path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("lecture impossible : ") + path);
    return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

uint32_t asU(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct Ctx {
    VkDevice device{};
    VkPhysicalDeviceMemoryProperties mem{};
    uint32_t memType(uint32_t bits, VkMemoryPropertyFlags want) const
    {
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & want) == want) return i;
        throw std::runtime_error("pas de type de mémoire");
    }
};

struct Buf { VkBuffer buf{}; VkDeviceMemory mem{}; void *map = nullptr; };
struct Img { VkImage img{}; VkDeviceMemory mem{}; VkImageView view{}; };

Buf makeBuf(Ctx &c, VkDeviceSize size, VkBufferUsageFlags usage)
{
    Buf b;
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

Img makeImg(Ctx &c, uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags usage)
{
    Img i;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = f;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
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
    vi.format = f;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(c.device, &vi, nullptr, &i.view), "vkCreateImageView");
    return i;
}

void imgBarrier(VkCommandBuffer cb, VkImage img, VkImageLayout from)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout = from;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 8) {
        std::fprintf(stderr, "usage : %s easu.spv rcas.spv entree.rgba largeur hauteur sortie_easu.rgba sortie_rcas.rgba [images]\n", argv[0]);
        return 2;
    }
    try {
        std::vector<char> easuSpv = readFile(argv[1]), rcasSpv = readFile(argv[2]), in = readFile(argv[3]);
        uint32_t inW, inH;
        std::memcpy(&inW, in.data(), 4);
        std::memcpy(&inH, in.data() + 4, 4);
        const uint32_t outW = uint32_t(std::atoi(argv[4])), outH = uint32_t(std::atoi(argv[5]));
        const int frames = argc > 8 ? std::atoi(argv[8]) : 2;

        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        VkInstance instance;
        check(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");
        uint32_t n = 1;
        VkPhysicalDevice phys;
        vkEnumeratePhysicalDevices(instance, &n, &phys);
        Ctx c;
        vkGetPhysicalDeviceMemoryProperties(phys, &c.mem);
        float prio = 1;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f12.scalarBlockLayout = VK_TRUE;
        f12.shaderFloat16 = VK_TRUE;
        VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f.pNext = &f12;
        f.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &f;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        check(vkCreateDevice(phys, &dci, nullptr, &c.device), "vkCreateDevice");
        VkQueue queue;
        vkGetDeviceQueue(c.device, 0, 0, &queue);

        // Disposition de gamescope : 7 liaisons.
        VkDescriptorSetLayoutBinding b[7] = {
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = 7;
        dli.pBindings = b;
        VkDescriptorSetLayout dsl;
        check(vkCreateDescriptorSetLayout(c.device, &dli, nullptr, &dsl), "vkCreateDescriptorSetLayout");
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
        VkPipelineLayout layout;
        check(vkCreatePipelineLayout(c.device, &pli, nullptr, &layout), "vkCreatePipelineLayout");

        auto pipeline = [&](const std::vector<char> &spv, uint32_t colorspaceMask) {
            VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smi.codeSize = spv.size();
            smi.pCode = reinterpret_cast<const uint32_t *>(spv.data());
            VkShaderModule m;
            check(vkCreateShaderModule(c.device, &smi, nullptr, &m), "vkCreateShaderModule");
            VkSpecializationMapEntry e{4, 0, 4};
            VkSpecializationInfo spec{1, &e, 4, &colorspaceMask};
            VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", &spec};
            cpi.layout = layout;
            VkPipeline p;
            check(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &p), "vkCreateComputePipelines");
            return p;
        };
        VkPipeline easu = pipeline(easuSpv, 1 /* calque 0 en sRGB */), rcas = pipeline(rcasSpv, 1);

        Img src = makeImg(c, inW, inH, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        Img tmp = makeImg(c, outW, outH, VK_FORMAT_B8G8R8A8_UNORM,
                          VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        Img fin = makeImg(c, outW, outH, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        Buf up = makeBuf(c, in.size() - 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memcpy(up.map, in.data() + 8, in.size() - 8);
        Buf downTmp = makeBuf(c, VkDeviceSize(outW) * outH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        Buf downFin = makeBuf(c, VkDeviceSize(outW) * outH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        // Tampon de constantes : EASU (FsrEasuCon + matrice) à l'offset 0, RCAS (u_c1) à l'offset 256.
        Buf ubo = makeBuf(c, 512, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        uint32_t con[16] = {asU(float(inW) / outW), asU(float(inH) / outH), asU(0.5f * inW / outW - 0.5f),
                            asU(0.5f * inH / outH - 0.5f), asU(1.0f / inW), asU(1.0f / inH), asU(1.0f / inW),
                            asU(-1.0f / inH), asU(-1.0f / inW), asU(2.0f / inH), asU(1.0f / inW), asU(2.0f / inH),
                            asU(0.0f), asU(4.0f / inH), 0, 0};
        const float ctm[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        std::memcpy(ubo.map, con, sizeof con);
        std::memcpy(static_cast<char *>(ubo.map) + 64, ctm, sizeof ctm);
        const uint32_t c1 = asU(std::exp2(-0.2f));  // netteté 2, le défaut de gamescope
        std::memcpy(static_cast<char *>(ubo.map) + 256, &c1, 4);

        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VkSampler sampler;
        check(vkCreateSampler(c.device, &si, nullptr, &sampler), "vkCreateSampler");

        VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 128},
                                      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64 * 36}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 64;
        dpi.poolSizeCount = 3;
        dpi.pPoolSizes = ps;
        VkDescriptorPool pool;
        check(vkCreateDescriptorPool(c.device, &dpi, nullptr, &pool), "vkCreateDescriptorPool");

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkCommandPool cmdPool;
        check(vkCreateCommandPool(c.device, &cpci, nullptr, &cmdPool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmdPool;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb;
        check(vkAllocateCommandBuffers(c.device, &cbai, &cb), "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sub.commandBufferCount = 1;
        sub.pCommandBuffers = &cb;

        // Une « image » de gamescope, comme CVulkanCmdBuffer::dispatch.
        auto gsDispatch = [&](VkPipeline p, VkImageView target, VkImageView tex0, VkDeviceSize uboOffset, uint32_t gw, uint32_t gh) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
            VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dai.descriptorPool = pool;
            dai.descriptorSetCount = 1;
            dai.pSetLayouts = &dsl;
            VkDescriptorSet set;
            check(vkAllocateDescriptorSets(c.device, &dai, &set), "vkAllocateDescriptorSets");
            VkDescriptorBufferInfo bufInfo{ubo.buf, uboOffset, 256};
            VkDescriptorImageInfo targetInfo{VK_NULL_HANDLE, target, VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo samplers[16];
            for (auto &s : samplers) s = {sampler, tex0, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet w[3]{};
            for (auto &x : w) x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, x.dstSet = set;
            w[0].dstBinding = 0, w[0].descriptorCount = 1, w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, w[0].pBufferInfo = &bufInfo;
            w[1].dstBinding = 1, w[1].descriptorCount = 1, w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, w[1].pImageInfo = &targetInfo;
            w[2].dstBinding = 3, w[2].descriptorCount = 16, w[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, w[2].pImageInfo = samplers;
            vkUpdateDescriptorSets(c.device, 3, w, 0, nullptr);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
            vkCmdDispatch(cb, (gw + 15) / 16, (gh + 15) / 16, 1);
        };

        for (int frame = 0; frame < frames; ++frame) {
            check(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
            check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
            if (frame == 0) {
                imgBarrier(cb, src.img, VK_IMAGE_LAYOUT_UNDEFINED);
                VkBufferImageCopy reg{};
                reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                reg.imageExtent = {inW, inH, 1};
                vkCmdCopyBufferToImage(cb, up.buf, src.img, VK_IMAGE_LAYOUT_GENERAL, 1, &reg);
                imgBarrier(cb, src.img, VK_IMAGE_LAYOUT_GENERAL);
            }
            imgBarrier(cb, tmp.img, VK_IMAGE_LAYOUT_UNDEFINED);   // gamescope : « discarded »
            gsDispatch(easu, tmp.view, src.view, 0, outW, outH);  // EASU (remplacé par la couche)
            imgBarrier(cb, tmp.img, VK_IMAGE_LAYOUT_GENERAL);     // gamescope : « dirty »
            imgBarrier(cb, fin.img, VK_IMAGE_LAYOUT_UNDEFINED);
            gsDispatch(rcas, fin.view, tmp.view, 256, outW, outH); // RCAS : doit relire le résultat
            imgBarrier(cb, fin.img, VK_IMAGE_LAYOUT_GENERAL);
            if (frame == frames - 1) {
                VkBufferImageCopy rb{};
                rb.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                rb.imageExtent = {outW, outH, 1};
                vkCmdCopyImageToBuffer(cb, tmp.img, VK_IMAGE_LAYOUT_GENERAL, downTmp.buf, 1, &rb);
                vkCmdCopyImageToBuffer(cb, fin.img, VK_IMAGE_LAYOUT_GENERAL, downFin.buf, 1, &rb);
            }
            check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
            check(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE), "vkQueueSubmit");
            check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
        }

        // BGRA -> RGBA pour les fichiers de sortie
        auto save = [&](const char *path, const Buf &buf) {
            std::vector<uint8_t> px(size_t(outW) * outH * 4);
            const uint8_t *s = static_cast<const uint8_t *>(buf.map);
            for (size_t i = 0; i < px.size(); i += 4) px[i] = s[i + 2], px[i + 1] = s[i + 1], px[i + 2] = s[i], px[i + 3] = s[i + 3];
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char *>(&outW), 4);
            f.write(reinterpret_cast<const char *>(&outH), 4);
            f.write(reinterpret_cast<const char *>(px.data()), std::streamsize(px.size()));
        };
        save(argv[6], downTmp);
        save(argv[7], downFin);
        vkDeviceWaitIdle(c.device);
        vkDestroyDevice(c.device, nullptr);
        vkDestroyInstance(instance, nullptr);
        std::printf("ok : %d images, %ux%u -> %ux%u\n", frames, inW, inH, outW, outH);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "erreur : %s\n", e.what());
        return 1;
    }
}
