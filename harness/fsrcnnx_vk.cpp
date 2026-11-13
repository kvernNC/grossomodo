// SPDX-License-Identifier: GPL-3.0-or-later
//
// Exécute FSRCNNX en passes Vulkan (shaders générés par scripts/gen_fsrcnnx_vk.py) et mesure le temps GPU de chaque
// passe avec des timestamps. Pensé pour tourner aussi sur le Steam Deck (binaire Linux, Vulkan de RADV).
//
//   fsrcnnx_vk <dossier_du_plan> <entrée.rgba> <sortie.rgba> <largeur> <hauteur> [répétitions] [fp16] [x2=<fichier.rgba>]
//
// La sortie est l'image finale (x2 puis réduction à largeur x hauteur). x2=... écrit aussi l'image x2 avant réduction.
// Les temps affichés sont les médianes sur les répétitions.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
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

struct PassDesc {
    std::string name, in, skip, out, grid;
};

} // namespace

int main(int argc, char **argv)
{
    if (argc < 6) {
        std::fprintf(stderr, "usage : %s plan_dir entree.rgba sortie.rgba largeur hauteur [repetitions] [fp16] [x2=fichier]\n", argv[0]);
        return 2;
    }
    try {
        const std::string dir = argv[1];
        const Image8 in = readRgba(argv[2]);
        const uint32_t outW = uint32_t(std::atoi(argv[4])), outH = uint32_t(std::atoi(argv[5]));
        int reps = 1;
        bool fp16 = false;
        std::string x2Path;
        float strength = 1.0f;
        for (int i = 6; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "fp16") fp16 = true;
            else if (a.rfind("x2=", 0) == 0) x2Path = a.substr(3);
            else if (a.rfind("strength=", 0) == 0) strength = std::strtof(a.c_str() + 9, nullptr);
            else reps = std::max(1, std::atoi(a.c_str()));
        }

        std::vector<PassDesc> plan;
        {
            std::istringstream ps(std::string(readFile(dir + "/plan.txt").data(), readFile(dir + "/plan.txt").size()));
            PassDesc p;
            while (ps >> p.name >> p.in >> p.skip >> p.out >> p.grid)
                plan.push_back(p);
        }

        Ctx c;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        check(vkCreateInstance(&ici, nullptr, &c.instance), "vkCreateInstance");
        uint32_t n = 1;
        VkResult er = vkEnumeratePhysicalDevices(c.instance, &n, &c.phys);
        if (n == 0 || (er != VK_SUCCESS && er != VK_INCOMPLETE))
            throw std::runtime_error("aucun GPU Vulkan");
        vkGetPhysicalDeviceMemoryProperties(c.phys, &c.mem);
        vkGetPhysicalDeviceProperties(c.phys, &c.props);
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, qf.data());
        for (uint32_t i = 0; i < qn; ++i)
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.family = i; c.timestampBits = qf[i].timestampValidBits; break; }

        VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features have11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        have11.pNext = &have12;
        VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        have.pNext = &have11;
        vkGetPhysicalDeviceFeatures2(c.phys, &have);
        if (fp16 && !have12.shaderFloat16)
            throw std::runtime_error("le GPU n'a pas shaderFloat16");

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = c.family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f12.shaderFloat16 = fp16 ? VK_TRUE : VK_FALSE;
        VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        feats.pNext = &f12;
        feats.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;  // comme gamescope
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &feats;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        check(vkCreateDevice(c.phys, &dci, nullptr, &c.device), "vkCreateDevice");
        vkGetDeviceQueue(c.device, c.family, 0, &c.queue);

        // Ressources : image du jeu (échantillonnée), trois cartes 8 canaux (RGBA16F, 2 couches), image x2, image finale.
        const VkImageUsageFlags storage = VK_IMAGE_USAGE_STORAGE_BIT;
        Tex src = makeTex(c, in.w, in.h, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        std::map<std::string, Tex> maps;
        for (const char *m : {"FEAT", "A", "B"})
            maps[m] = makeTex(c, in.w, in.h, VK_FORMAT_R16G16B16A16_SFLOAT, 2, storage);
        Tex out2x = makeTex(c, in.w * 2, in.h * 2, VK_FORMAT_R8G8B8A8_UNORM, 1,
                            storage | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        Tex fin = makeTex(c, outW, outH, VK_FORMAT_R8G8B8A8_UNORM, 1, storage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        Buffer up = makeBuffer(c, in.px.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memcpy(up.map, in.px.data(), in.px.size());
        Buffer down2x = makeBuffer(c, VkDeviceSize(out2x.w) * out2x.h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        Buffer downFin = makeBuffer(c, VkDeviceSize(outW) * outH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VkSampler sampler;
        check(vkCreateSampler(c.device, &si, nullptr, &sampler), "vkCreateSampler");

        // 0 et 6 : échantillonneurs (image du jeu, image x2) ; 1 à 5 : images de stockage.
        VkDescriptorSetLayoutBinding b[7];
        for (uint32_t i = 0; i < 7; ++i)
            b[i] = {i, (i == 0 || i == 6) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                    VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = 7;
        dli.pBindings = b;
        VkDescriptorSetLayout dsl;
        check(vkCreateDescriptorSetLayout(c.device, &dli, nullptr, &dsl), "vkCreateDescriptorSetLayout");
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};  // intensité du mélange (dernière passe)
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        VkPipelineLayout layout;
        check(vkCreatePipelineLayout(c.device, &pli, nullptr, &layout), "vkCreatePipelineLayout");

        const uint32_t np = uint32_t(plan.size());
        VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * np}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5 * np}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = np;
        dpi.poolSizeCount = 2;
        dpi.pPoolSizes = ps;
        VkDescriptorPool pool;
        check(vkCreateDescriptorPool(c.device, &dpi, nullptr, &pool), "vkCreateDescriptorPool");

        std::vector<VkPipeline> pipes(np);
        std::vector<VkDescriptorSet> sets(np);
        auto mapView = [&](const std::string &name, const char *fallback) {
            return maps.count(name) ? maps[name].view : maps[fallback].view;
        };
        for (uint32_t i = 0; i < np; ++i) {
            const PassDesc &p = plan[i];
            std::vector<char> spv = readFile(dir + "/" + p.name + (fp16 ? "_fp16.spv" : ".spv"));
            VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smi.codeSize = spv.size();
            smi.pCode = reinterpret_cast<const uint32_t *>(spv.data());
            VkShaderModule module;
            check(vkCreateShaderModule(c.device, &smi, nullptr, &module), "vkCreateShaderModule");
            VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
            cpi.layout = layout;
            check(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipes[i]), "vkCreateComputePipelines");

            VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dai.descriptorPool = pool;
            dai.descriptorSetCount = 1;
            dai.pSetLayouts = &dsl;
            check(vkAllocateDescriptorSets(c.device, &dai, &sets[i]), "vkAllocateDescriptorSets");
            // Liaisons inutilisées par une passe : n'importe quelle image du bon type (le shader ne les lit pas).
            VkDescriptorImageInfo ii[7] = {
                {sampler, src.view, VK_IMAGE_LAYOUT_GENERAL},
                {VK_NULL_HANDLE, mapView(p.in, "A"), VK_IMAGE_LAYOUT_GENERAL},
                {VK_NULL_HANDLE, mapView(p.skip, "A"), VK_IMAGE_LAYOUT_GENERAL},
                {VK_NULL_HANDLE, mapView(p.out, "B"), VK_IMAGE_LAYOUT_GENERAL},
                {VK_NULL_HANDLE, p.out == "FINAL" ? fin.view : out2x.view, VK_IMAGE_LAYOUT_GENERAL},
                {VK_NULL_HANDLE, out2x.view, VK_IMAGE_LAYOUT_GENERAL},
                {sampler, out2x.view, VK_IMAGE_LAYOUT_GENERAL},
            };
            VkWriteDescriptorSet w[7]{};
            for (uint32_t k = 0; k < 7; ++k) {
                w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[k].dstSet = sets[i];
                w[k].dstBinding = k;
                w[k].descriptorCount = 1;
                w[k].descriptorType = b[k].descriptorType;
                w[k].pImageInfo = &ii[k];
            }
            vkUpdateDescriptorSets(c.device, 7, w, 0, nullptr);
        }

        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = np + 1;
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
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

        // Préparation : mise en place des dispositions et envoi de l'image du jeu.
        check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
        toGeneral(cb, src.img, 1);
        for (auto &m : maps) toGeneral(cb, m.second.img, 2);
        toGeneral(cb, out2x.img, 1);
        toGeneral(cb, fin.img, 1);
        VkBufferImageCopy reg{};
        reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        reg.imageExtent = {in.w, in.h, 1};
        vkCmdCopyBufferToImage(cb, up.buf, src.img, VK_IMAGE_LAYOUT_GENERAL, 1, &reg);
        check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
        check(vkQueueSubmit(c.queue, 1, &sub, VK_NULL_HANDLE), "vkQueueSubmit");
        check(vkQueueWaitIdle(c.queue), "vkQueueWaitIdle");

        std::vector<std::vector<double>> passMs(np);
        for (int r = 0; r < reps; ++r) {
            check(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
            check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
            vkCmdResetQueryPool(cb, queries, 0, np + 1);
            memoryBarrier(cb, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
            for (uint32_t i = 0; i < np; ++i) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[i]);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[i], 0, nullptr);
                vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &strength);
                const bool final = plan[i].grid == "final";
                const uint32_t gw = final ? outW : in.w, gh = final ? outH : in.h;
                vkCmdDispatch(cb, (gw + 7) / 8, (gh + 7) / 8, 1);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, i + 1);
                memoryBarrier(cb, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
            }
            if (r == reps - 1) {
                VkBufferImageCopy rb{};
                rb.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                rb.imageExtent = {out2x.w, out2x.h, 1};
                vkCmdCopyImageToBuffer(cb, out2x.img, VK_IMAGE_LAYOUT_GENERAL, down2x.buf, 1, &rb);
                rb.imageExtent = {outW, outH, 1};
                vkCmdCopyImageToBuffer(cb, fin.img, VK_IMAGE_LAYOUT_GENERAL, downFin.buf, 1, &rb);
            }
            check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
            check(vkQueueSubmit(c.queue, 1, &sub, VK_NULL_HANDLE), "vkQueueSubmit");
            check(vkQueueWaitIdle(c.queue), "vkQueueWaitIdle");
            if (c.timestampBits) {
                std::vector<uint64_t> ts(np + 1);
                check(vkGetQueryPoolResults(c.device, queries, 0, np + 1, ts.size() * 8, ts.data(), 8,
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults");
                for (uint32_t i = 0; i < np; ++i)
                    passMs[i].push_back(double(ts[i + 1] - ts[i]) * c.props.limits.timestampPeriod * 1e-6);
            }
        }

        writeRgba(argv[3], outW, outH, downFin.map);
        if (!x2Path.empty())
            writeRgba(x2Path, out2x.w, out2x.h, down2x.map);

        std::printf("GPU : %s, %s, %ux%u -> %ux%u -> %ux%u, %d répétitions\n", c.props.deviceName, fp16 ? "FP16" : "FP32",
                    in.w, in.h, out2x.w, out2x.h, outW, outH, reps);
        if (!c.timestampBits) {
            std::printf("  (pas de timestamps sur cette file)\n");
            return 0;
        }
        double total = 0;
        for (uint32_t i = 0; i < np; ++i) {
            std::vector<double> v = passMs[i];
            std::sort(v.begin(), v.end());
            const double med = v[v.size() / 2];
            total += med;
            std::printf("  %s  %-5s -> %-6s %8.3f ms\n", plan[i].name.c_str(), plan[i].in.c_str(), plan[i].out.c_str(), med);
        }
        std::printf("  total (médianes)                %8.3f ms\n", total);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "erreur : %s\n", e.what());
        return 1;
    }
}
