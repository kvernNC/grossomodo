// SPDX-License-Identifier: GPL-3.0-or-later
//
// Banc de comparaison : exécute le scaler spatial LS1 de Lossless Scaling à partir de SPIR-V que l'utilisateur a
// traduit lui-même depuis SON Lossless.dll (rien de LS1 n'est distribué ici). Même chaîne que MAKO : bindings
// CBV i -> i, sampler -> 16+i, SRV -> 32+i, UAV -> 48+i ; groupes 16x16 ; paramètres de 48 octets.
//
//   ls1_vk <dossier_spv> quality|performance <variante 0-4> <entrée.rgba> <sortie.rgba> <largeur> <hauteur> [répétitions]

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


struct Stage {
    std::string file;
    uint32_t srv, uav, cbv, smp;
    std::vector<Tex *> in;
    Tex *out;
    bool full;  // grille de sortie (sinon grille de l'image source)
    VkDescriptorSetLayout dsl{};
    VkPipelineLayout layout{};
    VkPipeline pipe{};
    VkDescriptorSet set{};
};

} // namespace

int main(int argc, char **argv)
{
    if (argc < 8) {
        std::fprintf(stderr, "usage : %s spv_dir quality|performance variante entree.rgba sortie.rgba largeur hauteur [repetitions]\n", argv[0]);
        return 2;
    }
    try {
        const std::string dir = argv[1];
        const bool perf = std::string(argv[2]) == "performance";
        const uint32_t variant = uint32_t(std::atoi(argv[3]));
        if (variant > 4) throw std::runtime_error("variante 0 a 4");
        const Image8 in = readRgba(argv[4]);
        const uint32_t outW = uint32_t(std::atoi(argv[6])), outH = uint32_t(std::atoi(argv[7]));
        const int reps = argc > 8 ? std::max(1, std::atoi(argv[8])) : 1;

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

        // Images comme MAKO : source RGBA8, intermédiaires RGBA8 (qualité), caractéristiques R8_SNORM en 2x, sortie RGBA8.
        const VkImageUsageFlags rw = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        Tex src = makeTex(c, in.w, in.h, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        Tex ia = makeTex(c, in.w, in.h, VK_FORMAT_R8G8B8A8_UNORM, 1, rw);
        Tex ib = makeTex(c, in.w, in.h, VK_FORMAT_R8G8B8A8_UNORM, 1, rw);
        Tex feat = makeTex(c, in.w * 2, in.h * 2, VK_FORMAT_R8_SNORM, 1, rw);
        Tex fin = makeTex(c, outW, outH, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        struct Params { uint32_t sw, sh, cw, ch, ox, oy, gamma, res, ow, oh, oox, ooy; } prm{
            in.w, in.h, in.w, in.h, 0, 0, 0, 0, outW, outH, 0, 0};
        Buffer ubo = makeBuffer(c, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        std::memcpy(ubo.map, &prm, sizeof prm);
        Buffer up = makeBuffer(c, in.px.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memcpy(up.map, in.px.data(), in.px.size());
        Buffer downFin = makeBuffer(c, VkDeviceSize(outW) * outH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VkSampler sampler;
        check(vkCreateSampler(c.device, &si, nullptr, &sampler), "vkCreateSampler");

        // Ressources du DLL (numérotation de MAKO) : performance 141+v ; qualité 147+3v, +1, +2 ; reconstruction 146.
        std::vector<Stage> stages;
        auto id = [](uint32_t i) { return std::to_string(i) + ".spv"; };
        if (perf) {
            stages.push_back({id(141 + variant), 1, 1, 1, 0, {&src}, &feat, false});
        } else {
            const uint32_t f = 147 + 3 * variant;
            stages.push_back({id(f), 1, 1, 1, 0, {&src}, &ia, false});
            stages.push_back({id(f + 1), 1, 1, 0, 0, {&ia}, &ib, false});
            stages.push_back({id(f + 2), 1, 1, 1, 0, {&ib}, &feat, false});
        }
        stages.push_back({id(146), 2, 1, 1, 1, {&feat, &src}, &fin, true});

        VkDescriptorPoolSize ps[4] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8}, {VK_DESCRIPTOR_TYPE_SAMPLER, 8},
                                      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 8;
        dpi.poolSizeCount = 4;
        dpi.pPoolSizes = ps;
        VkDescriptorPool pool;
        check(vkCreateDescriptorPool(c.device, &dpi, nullptr, &pool), "vkCreateDescriptorPool");

        for (Stage &s : stages) {
            std::vector<VkDescriptorSetLayoutBinding> b;
            auto add = [&](uint32_t base, uint32_t count, VkDescriptorType t) {
                for (uint32_t i = 0; i < count; ++i) b.push_back({base + i, t, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
            };
            add(0, s.cbv, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
            add(16, s.smp, VK_DESCRIPTOR_TYPE_SAMPLER);
            add(32, s.srv, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
            add(48, s.uav, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dli.bindingCount = uint32_t(b.size());
            dli.pBindings = b.data();
            check(vkCreateDescriptorSetLayout(c.device, &dli, nullptr, &s.dsl), "vkCreateDescriptorSetLayout");
            VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pli.setLayoutCount = 1;
            pli.pSetLayouts = &s.dsl;
            check(vkCreatePipelineLayout(c.device, &pli, nullptr, &s.layout), "vkCreatePipelineLayout");
            std::vector<char> spv = readFile(dir + "/" + s.file);
            VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smi.codeSize = spv.size();
            smi.pCode = reinterpret_cast<const uint32_t *>(spv.data());
            VkShaderModule module;
            check(vkCreateShaderModule(c.device, &smi, nullptr, &module), "vkCreateShaderModule");
            VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
            cpi.layout = s.layout;
            check(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &s.pipe), "vkCreateComputePipelines");
            VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dai.descriptorPool = pool;
            dai.descriptorSetCount = 1;
            dai.pSetLayouts = &s.dsl;
            check(vkAllocateDescriptorSets(c.device, &dai, &s.set), "vkAllocateDescriptorSets");
            std::vector<VkWriteDescriptorSet> w;
            std::vector<VkDescriptorImageInfo> ii(8);
            VkDescriptorBufferInfo bi{ubo.buf, 0, sizeof prm};
            size_t k = 0;
            auto wr = [&](uint32_t binding, VkDescriptorType t) {
                VkWriteDescriptorSet x{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                x.dstSet = s.set;
                x.dstBinding = binding;
                x.descriptorCount = 1;
                x.descriptorType = t;
                if (t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) x.pBufferInfo = &bi;
                else x.pImageInfo = &ii[k++];
                w.push_back(x);
            };
            if (s.cbv) wr(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
            if (s.smp) { ii[k] = {sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL}; wr(16, VK_DESCRIPTOR_TYPE_SAMPLER); }
            for (uint32_t i = 0; i < s.srv; ++i) { ii[k] = {VK_NULL_HANDLE, s.in[i]->view, VK_IMAGE_LAYOUT_GENERAL}; wr(32 + i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE); }
            ii[k] = {VK_NULL_HANDLE, s.out->view, VK_IMAGE_LAYOUT_GENERAL};
            wr(48, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            vkUpdateDescriptorSets(c.device, uint32_t(w.size()), w.data(), 0, nullptr);
        }
        const uint32_t np = uint32_t(stages.size());

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
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

        check(vkBeginCommandBuffer(cb, &cbi), "vkBeginCommandBuffer");
        for (Tex *t : {&src, &ia, &ib, &feat, &fin}) toGeneral(cb, t->img, 1);
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
            check(vkBeginCommandBuffer(cb, &cbi), "vkBeginCommandBuffer");
            vkCmdResetQueryPool(cb, queries, 0, np + 1);
            memoryBarrier(cb, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
            for (uint32_t i = 0; i < np; ++i) {
                const Stage &s = stages[i];
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.layout, 0, 1, &s.set, 0, nullptr);
                const uint32_t gw = s.full ? outW : in.w, gh = s.full ? outH : in.h;
                vkCmdDispatch(cb, (gw + 15) / 16, (gh + 15) / 16, 1);  // comme MAKO
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, i + 1);
                memoryBarrier(cb, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
            }
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
                std::vector<uint64_t> ts(np + 1);
                check(vkGetQueryPoolResults(c.device, queries, 0, np + 1, ts.size() * 8, ts.data(), 8,
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults");
                for (uint32_t i = 0; i < np; ++i)
                    passMs[i].push_back(double(ts[i + 1] - ts[i]) * c.props.limits.timestampPeriod * 1e-6);
            }
        }
        writeRgba(argv[5], outW, outH, downFin.map);
        std::printf("GPU : %s, LS1 %s variante %u, %ux%u -> %ux%u, %d répétitions\n", c.props.deviceName,
                    perf ? "performance" : "qualité", variant, in.w, in.h, outW, outH, reps);
        if (!c.timestampBits) return 0;
        double total = 0;
        for (uint32_t i = 0; i < np; ++i) {
            std::vector<double> v = passMs[i];
            std::sort(v.begin(), v.end());
            total += v[v.size() / 2];
            std::printf("  %-8s %8.3f ms\n", stages[i].file.c_str(), v[v.size() / 2]);
        }
        std::printf("  total    %8.3f ms\n", total);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "erreur : %s\n", e.what());
        return 1;
    }
}
