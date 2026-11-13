// SPDX-License-Identifier: GPL-3.0-or-later
//
// GrossoModo Bench : comparer des upscalers spatiaux en direct, sur le Steam Deck (mode Bureau, ou mode Jeu en jeu
// non-Steam) ou sur PC. Une grande image défile lentement ; elle est « rendue » à basse résolution (avec ou sans
// anti-crénelage), agrandie vers 1280x800 par deux méthodes au choix (écran partagé), puis affinée par RCAS comme dans
// gamescope. Le temps GPU de chaque méthode est mesuré et affiché.
//
//   grossomodo_bench [dossier_images]              fenêtre (plein écran sous gamescope)
//   grossomodo_bench --headless sortie.rgba [...]  sans écran (tests) : --frames N --a I --b J --res WxH --aa 0|1
//
// Commandes (clavier / manette) :
//   2 / A          méthode de droite suivante        1 / X       méthode de gauche suivante
//   S / Y          écran partagé oui/non             Espace / B  pause du défilement
//   Haut, Bas      résolution de rendu               [ ] / L1 R1 netteté RCAS (curseur du Deck, 0 = max ; -1 = off)
//   M / Vue        anti-crénelage de la scène         N / Menu    image suivante
//   Z / R3         loupe x1, x2, x4                   H / L3      aide        Échap : quitter

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "../third_party/stb/stb_image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t OUT_W = 1280, OUT_H = 800;

void check(VkResult r, const char *what)
{
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + " : VkResult " + std::to_string(r));
}

std::vector<char> readFile(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("lecture impossible : " + path);
    return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

uint32_t asU(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

uint16_t toHalf(float f)
{
    uint32_t x = asU(f);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = int32_t((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return uint16_t(sign);
    if (e >= 31) return uint16_t(sign | 0x7c00u);
    return uint16_t(sign | (uint32_t(e) << 10) | (m >> 13));
}

// ------------------------------------------------------------------------------------------------------- Vulkan

struct Img {
    VkImage img{};
    VkDeviceMemory mem{};
    VkImageView view{};
    uint32_t w = 0, h = 0;
};

struct Buf {
    VkBuffer buf{};
    VkDeviceMemory mem{};
    void *map = nullptr;
};

struct Pipe {
    VkDescriptorSetLayout dsl{};
    VkPipelineLayout layout{};
    VkPipeline pipe{};
    std::vector<VkDescriptorType> types;
};

struct Gpu {
    VkInstance instance{};
    VkPhysicalDevice phys{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t family = 0;
    VkPhysicalDeviceMemoryProperties mem{};
    VkPhysicalDeviceProperties props{};
    VkDescriptorPool pool{};
    VkSampler linear{}, nearest{};

    uint32_t memType(uint32_t bits, VkMemoryPropertyFlags want) const
    {
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & want) == want) return i;
        throw std::runtime_error("pas de type de mémoire");
    }

    Img image(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage)
    {
        Img t;
        t.w = w;
        t.h = h;
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = fmt;
        ci.extent = {w, h, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = usage;
        check(vkCreateImage(device, &ci, nullptr, &t.img), "vkCreateImage");
        VkMemoryRequirements r;
        vkGetImageMemoryRequirements(device, t.img, &r);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = r.size;
        ai.memoryTypeIndex = memType(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(device, &ai, nullptr, &t.mem), "vkAllocateMemory");
        check(vkBindImageMemory(device, t.img, t.mem, 0), "vkBindImageMemory");
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = t.img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = fmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(device, &vi, nullptr, &t.view), "vkCreateImageView");
        return t;
    }

    void free(Img &t)
    {
        if (t.view) vkDestroyImageView(device, t.view, nullptr);
        if (t.img) vkDestroyImage(device, t.img, nullptr);
        if (t.mem) vkFreeMemory(device, t.mem, nullptr);
        t = Img{};
    }

    Buf buffer(VkDeviceSize size, VkBufferUsageFlags usage)
    {
        Buf b;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = usage;
        check(vkCreateBuffer(device, &bi, nullptr, &b.buf), "vkCreateBuffer");
        VkMemoryRequirements r;
        vkGetBufferMemoryRequirements(device, b.buf, &r);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = r.size;
        ai.memoryTypeIndex = memType(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device, &ai, nullptr, &b.mem), "vkAllocateMemory");
        check(vkBindBufferMemory(device, b.buf, b.mem, 0), "vkBindBufferMemory");
        check(vkMapMemory(device, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map), "vkMapMemory");
        return b;
    }

    void free(Buf &b)
    {
        if (b.buf) vkDestroyBuffer(device, b.buf, nullptr);
        if (b.mem) vkFreeMemory(device, b.mem, nullptr);
        b = Buf{};
    }

    // Pipeline de calcul : liaisons 0..n-1 des types donnés, constantes poussées de pcSize octets.
    Pipe pipeline(const std::string &spvPath, std::vector<VkDescriptorType> types, uint32_t pcSize)
    {
        Pipe p;
        p.types = types;
        std::vector<VkDescriptorSetLayoutBinding> b;
        for (uint32_t i = 0; i < types.size(); ++i) b.push_back({i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = uint32_t(b.size());
        dli.pBindings = b.data();
        check(vkCreateDescriptorSetLayout(device, &dli, nullptr, &p.dsl), "vkCreateDescriptorSetLayout");
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &p.dsl;
        pli.pushConstantRangeCount = pcSize ? 1 : 0;
        pli.pPushConstantRanges = &pcr;
        check(vkCreatePipelineLayout(device, &pli, nullptr, &p.layout), "vkCreatePipelineLayout");
        std::vector<char> spv = readFile(spvPath);
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = spv.size();
        smi.pCode = reinterpret_cast<const uint32_t *>(spv.data());
        VkShaderModule module;
        check(vkCreateShaderModule(device, &smi, nullptr, &module), "vkCreateShaderModule");
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        cpi.layout = p.layout;
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &p.pipe), "vkCreateComputePipelines");
        vkDestroyShaderModule(device, module, nullptr);
        return p;
    }

    // Une ressource par liaison : image (vue + échantillonneur éventuel) ou tampon de stockage.
    struct Res {
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
    };

    void dispatch(VkCommandBuffer cb, const Pipe &p, std::vector<Res> res, const void *pc, uint32_t pcSize, uint32_t gx,
                  uint32_t gy)
    {
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &p.dsl;
        VkDescriptorSet set;
        check(vkAllocateDescriptorSets(device, &dai, &set), "vkAllocateDescriptorSets");
        std::vector<VkWriteDescriptorSet> w(res.size());
        std::vector<VkDescriptorImageInfo> ii(res.size());
        std::vector<VkDescriptorBufferInfo> bi(res.size());
        for (size_t i = 0; i < res.size(); ++i) {
            w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[i].dstSet = set;
            w[i].dstBinding = uint32_t(i);
            w[i].descriptorCount = 1;
            w[i].descriptorType = p.types[i];
            if (p.types[i] == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                bi[i] = {res[i].buffer, 0, VK_WHOLE_SIZE};
                w[i].pBufferInfo = &bi[i];
            } else {
                ii[i] = {res[i].sampler, res[i].view, VK_IMAGE_LAYOUT_GENERAL};
                w[i].pImageInfo = &ii[i];
            }
        }
        vkUpdateDescriptorSets(device, uint32_t(w.size()), w.data(), 0, nullptr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
        if (pcSize) vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
        vkCmdDispatch(cb, gx, gy, 1);
        VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        m.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
    }
};

void toGeneral(VkCommandBuffer cb, VkImage img)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
}

// ------------------------------------------------------------------------------------------------------ images

struct Picture {
    std::string name;
    int w = 0, h = 0;
    std::vector<uint8_t> rgba;
};

// Mire intégrée (pire cas pour le scintillement) : anneaux de fréquence croissante, traits fins à plusieurs angles,
// damiers, texte simulé, dégradés.
Picture testPattern()
{
    Picture p;
    p.name = "mire";
    p.w = 3200;
    p.h = 2000;
    p.rgba.resize(size_t(p.w) * p.h * 4);
    for (int y = 0; y < p.h; ++y)
        for (int x = 0; x < p.w; ++x) {
            float v = 0.5f, r, g, b;
            float fx = float(x), fy = float(y);
            if (x < 1600 && y < 1000) {  // plaque de zones
                float dx = fx - 800, dy = fy - 500;
                v = 0.5f + 0.5f * std::cos(0.0012f * (dx * dx + dy * dy));
            } else if (x >= 1600 && y < 1000) {  // traits fins à plusieurs angles
                int sector = int((fx - 1600) / 400);
                float a = 0.25f + 0.35f * float(sector);
                float d = std::fabs(std::fmod((fx - 1600) * std::cos(a) + fy * std::sin(a), 12.0f));
                v = d < 1.5f ? 0.05f : 0.92f;
            } else if (x < 1600) {  // damiers de tailles décroissantes
                int s = 2 + (x / 400) * 2;
                v = ((x / s + y / s) & 1) ? 0.9f : 0.1f;
            } else {  // « texte » : blocs de traits courts
                int cx = (x - 1600) % 22, cy = (y - 1000) % 30;
                int glyph = ((x - 1600) / 22 * 7 + (y - 1000) / 30 * 13) % 5;
                bool ink = cy > 6 && cy < 24 && cx > 3 && cx < 18 &&
                           ((glyph == 0 && (cx < 6 || cy > 21)) || (glyph == 1 && (cx > 15 || cy < 9)) ||
                            (glyph == 2 && std::abs(cx - cy + 5) < 2) || (glyph == 3 && (cy > 13 && cy < 16)) ||
                            (glyph == 4 && (cx < 6 || cx > 15)));
                v = ink ? 0.08f : 0.95f;
            }
            r = g = b = v;
            if (y >= 1000 && x >= 1600 && (y - 1000) > 800) {  // dégradé coloré en bas à droite
                r = fx / p.w;
                g = 1.0f - fy / p.h;
                b = 0.5f;
            }
            uint8_t *o = &p.rgba[(size_t(y) * p.w + x) * 4];
            o[0] = uint8_t(r * 255);
            o[1] = uint8_t(g * 255);
            o[2] = uint8_t(b * 255);
            o[3] = 255;
        }
    return p;
}

std::vector<Picture> loadPictures(const std::string &dir)
{
    std::vector<Picture> out;
    out.push_back(testPattern());
    std::vector<std::string> names;
    if (DIR *d = opendir(dir.c_str())) {
        while (dirent *e = readdir(d)) {
            std::string n = e->d_name;
            std::string l = n;
            std::transform(l.begin(), l.end(), l.begin(), ::tolower);
            if (l.size() > 4 && (l.rfind(".png") == l.size() - 4 || l.rfind(".jpg") == l.size() - 4 || l.rfind(".jpeg") == l.size() - 5))
                names.push_back(n);
        }
        closedir(d);
    }
    std::sort(names.begin(), names.end());
    for (const std::string &n : names) {
        Picture p;
        int c;
        stbi_uc *px = stbi_load((dir + "/" + n).c_str(), &p.w, &p.h, &c, 4);
        if (!px) {
            std::fprintf(stderr, "image illisible : %s\n", n.c_str());
            continue;
        }
        p.name = n;
        p.rgba.assign(px, px + size_t(p.w) * p.h * 4);
        stbi_image_free(px);
        out.push_back(std::move(p));
    }
    return out;
}

// ------------------------------------------------------------------------------------------------------ méthodes

struct Ravu {
    std::string name;
    bool lite = false;
    uint32_t gw = 32, gh = 8;
    Pipe main, final;
    Img lut;
    bool lutNearest = false;
};

struct Side {
    Img tmp, out;  // sortie de la méthode, puis après RCAS
    Img x2;        // luminance x2 pour RAVU-Lite (résolution de rendu x2)
};

const std::vector<std::pair<uint32_t, uint32_t>> RES = {{640, 400}, {720, 450}, {800, 500}, {922, 576}, {960, 600}, {1024, 640}, {1280, 800}};

struct App {
    Gpu g;
    std::string root;  // dossier des shaders et tables
    Pipe scene, bilinear, easu, rcas, compose, text;
    std::vector<Ravu> ravus;
    std::vector<std::string> methods;  // noms affichés
    std::vector<Picture> pics;
    Img picture, render, native, finalImg;
    Side sides[2];
    Buf textBuf, upload;
    VkQueryPool queries{};
    VkCommandPool cmdPool{};
    VkCommandBuffer cb{};
    VkFence fence{};

    int methodA = 1, methodB = 2, resIndex = 2, sharp = 2, zoom = 1, picIndex = 0;
    bool split = true, paused = false, aa = false, help = true;
    double panPos = 0.0;  // en pixels de sortie
    int panDir = 1;
    float msA = 0, msB = 0;
    bool needUpload = true, needResize = true;

    std::string spv(const std::string &n) const { return root + "/shaders/" + n + ".spv"; }

    void init(bool needsSwapchainExt, const std::vector<const char *> &instExts)
    {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "GrossoModo Bench";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        ici.enabledExtensionCount = uint32_t(instExts.size());
        ici.ppEnabledExtensionNames = instExts.data();
        check(vkCreateInstance(&ici, nullptr, &g.instance), "vkCreateInstance");
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(g.instance, &n, nullptr);
        if (!n) throw std::runtime_error("aucun GPU Vulkan");
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(g.instance, &n, devs.data());
        g.phys = devs[0];
        for (VkPhysicalDevice d : devs) {  // préférer un vrai GPU
            VkPhysicalDeviceProperties pr;
            vkGetPhysicalDeviceProperties(d, &pr);
            if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                g.phys = d;
                break;
            }
        }
        vkGetPhysicalDeviceMemoryProperties(g.phys, &g.mem);
        vkGetPhysicalDeviceProperties(g.phys, &g.props);
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qn, qf.data());
        g.family = UINT32_MAX;
        for (uint32_t i = 0; i < qn; ++i)
            if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { g.family = i; break; }
        if (g.family == UINT32_MAX) throw std::runtime_error("pas de file graphique + calcul");
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = g.family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkPhysicalDeviceFeatures feats{};
        feats.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        const char *devExt[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pEnabledFeatures = &feats;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = needsSwapchainExt ? 1 : 0;
        dci.ppEnabledExtensionNames = devExt;
        check(vkCreateDevice(g.phys, &dci, nullptr, &g.device), "vkCreateDevice");
        vkGetDeviceQueue(g.device, g.family, 0, &g.queue);

        VkDescriptorPoolSize ps[4] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 128},
                                      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16}, {VK_DESCRIPTOR_TYPE_SAMPLER, 4}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 128;
        dpi.poolSizeCount = 4;
        dpi.pPoolSizes = ps;
        check(vkCreateDescriptorPool(g.device, &dpi, nullptr, &g.pool), "vkCreateDescriptorPool");
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        check(vkCreateSampler(g.device, &si, nullptr, &g.linear), "vkCreateSampler");
        si.magFilter = si.minFilter = VK_FILTER_NEAREST;
        check(vkCreateSampler(g.device, &si, nullptr, &g.nearest), "vkCreateSampler");

        const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        scene = g.pipeline(spv("scene"), {S, I}, 32);
        bilinear = g.pipeline(spv("bilinear"), {S, I}, 16);
        easu = g.pipeline(spv("easu"), {S, I}, 64);
        rcas = g.pipeline(spv("rcas"), {S, I}, 16);
        compose = g.pipeline(spv("compose"), {S, S, I}, 24);
        text = g.pipeline(spv("text"), {I, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, 16);
        methods = {"Bilineaire", "FSR 1 (EASU)"};
        for (const char *r : {"ravu-lite-ar-r3", "ravu-lite-r3", "ravu-zoom-r2"}) {
            try {
                ravus.push_back(loadRavu(r));
                methods.push_back(std::string("RAVU ") + (r + 5));
            } catch (const std::exception &e) {
                std::fprintf(stderr, "%s absent : %s\n", r, e.what());
            }
        }
        methods.push_back("Natif 1280x800");
        methodB = std::min<int>(2, int(methods.size()) - 1);

        textBuf = g.buffer(4 * 4 * 80, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        finalImg = g.image(OUT_W, OUT_H, VK_FORMAT_R8G8B8A8_UNORM,
                           VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        native = g.image(OUT_W, OUT_H, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        for (Side &s : sides) {
            s.tmp = g.image(OUT_W, OUT_H, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            s.out = g.image(OUT_W, OUT_H, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 4;
        check(vkCreateQueryPool(g.device, &qpi, nullptr, &queries), "vkCreateQueryPool");
        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.queueFamilyIndex = g.family;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        check(vkCreateCommandPool(g.device, &cpci, nullptr, &cmdPool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmdPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(g.device, &cbai, &cb), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        check(vkCreateFence(g.device, &fci, nullptr, &fence), "vkCreateFence");
        // Les images de sortie (fixes) passent en GENERAL une fois pour toutes.
        oneShot([&](VkCommandBuffer c) {
            for (Img *t : {&finalImg, &native, &sides[0].tmp, &sides[0].out, &sides[1].tmp, &sides[1].out}) toGeneral(c, t->img);
            for (Ravu &r : ravus) uploadLut(c, r);
        });
    }

    std::vector<Buf> pendingUploads;

    template <typename F>
    void oneShot(F f)
    {
        check(vkWaitForFences(g.device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        check(vkQueueWaitIdle(g.queue), "vkQueueWaitIdle");
        check(vkResetDescriptorPool(g.device, g.pool, 0), "vkResetDescriptorPool");
        check(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
        f(cb);
        check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
        VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sub.commandBufferCount = 1;
        sub.pCommandBuffers = &cb;
        check(vkQueueSubmit(g.queue, 1, &sub, VK_NULL_HANDLE), "vkQueueSubmit");
        check(vkQueueWaitIdle(g.queue), "vkQueueWaitIdle");
        for (Buf &b : pendingUploads) g.free(b);
        pendingUploads.clear();
    }

    void copyToImage(VkCommandBuffer c, const void *data, size_t size, Img &dst)
    {
        Buf b = g.buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memcpy(b.map, data, size);
        toGeneral(c, dst.img);
        VkBufferImageCopy reg{};
        reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        reg.imageExtent = {dst.w, dst.h, 1};
        vkCmdCopyBufferToImage(c, b.buf, dst.img, VK_IMAGE_LAYOUT_GENERAL, 1, &reg);
        VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        m.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
        pendingUploads.push_back(b);
    }

    std::vector<char> lutData;
    Ravu loadRavu(const std::string &name)
    {
        Ravu r;
        r.name = name;
        const std::string dir = root + "/ravu/" + name;
        std::vector<char> t = readFile(dir + "/luts.txt");
        char nm[128];
        unsigned w, h;
        if (std::sscanf(std::string(t.begin(), t.end()).c_str(), "%127s %u %u", nm, &w, &h) != 3) throw std::runtime_error("luts.txt");
        r.lite = std::ifstream(dir + "/final.spv").good();
        r.lutNearest = r.lite;
        const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        r.main = g.pipeline(dir + "/ravu.spv", {S, I, S}, 16);
        if (r.lite) r.final = g.pipeline(dir + "/final.spv", {S, I, S}, 16);
        r.lut = g.image(w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        r.name = name;
        return r;
    }

    void uploadLut(VkCommandBuffer c, Ravu &r)
    {
        const std::string dir = root + "/ravu/" + r.name;
        std::vector<char> t = readFile(dir + "/luts.txt");
        char nm[128];
        unsigned w, h;
        std::sscanf(std::string(t.begin(), t.end()).c_str(), "%127s %u %u", nm, &w, &h);
        std::vector<char> data = readFile(dir + "/" + nm + ".bin");
        copyToImage(c, data.data(), data.size(), r.lut);
    }

    void resize()
    {
        vkDeviceWaitIdle(g.device);
        g.free(render);
        for (Side &s : sides) g.free(s.x2);
        const auto [w, h] = RES[resIndex];
        render = g.image(w, h, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        for (Side &s : sides)
            s.x2 = g.image(2 * w, 2 * h, VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        oneShot([&](VkCommandBuffer c) {
            toGeneral(c, render.img);
            for (Side &s : sides) toGeneral(c, s.x2.img);
        });
        needResize = false;
    }

    void uploadPicture()
    {
        vkDeviceWaitIdle(g.device);
        g.free(picture);
        const Picture &p = pics[picIndex];
        picture = g.image(uint32_t(p.w), uint32_t(p.h), VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        oneShot([&](VkCommandBuffer c) { copyToImage(c, p.rgba.data(), p.rgba.size(), picture); });
        needUpload = false;
        panPos = 0;
    }

    // Fenêtre de l'image source (coordonnées normalisées) : 16:10, 75 % de l'image, défilement horizontal.
    void sceneRect(float rect[4], double pos) const
    {
        const Picture &p = pics[picIndex];
        float cw = 0.75f * float(p.w), ch = cw * 10.0f / 16.0f;
        if (ch > 0.95f * float(p.h)) {
            ch = 0.95f * float(p.h);
            cw = ch * 16.0f / 10.0f;
        }
        float maxShift = float(p.w) - cw;
        // pos est en pixels de sortie : 1 pixel de sortie = cw / OUT_W pixels source
        float x = float(pos) * cw / float(OUT_W);
        x = std::min(std::max(x, 0.0f), maxShift);
        float y = 0.5f * (float(p.h) - ch);
        rect[0] = x / float(p.w);
        rect[1] = y / float(p.h);
        rect[2] = cw / float(p.w);
        rect[3] = ch / float(p.h);
    }

    double maxPan() const
    {
        const Picture &p = pics[picIndex];
        float cw = 0.75f * float(p.w), ch = cw * 10.0f / 16.0f;
        if (ch > 0.95f * float(p.h)) cw = 0.95f * float(p.h) * 16.0f / 10.0f;
        return double(float(p.w) - cw) * OUT_W / cw;
    }

    void runMethod(VkCommandBuffer c, int m, Side &s)
    {
        const auto [w, h] = RES[resIndex];
        const float sizes[4] = {float(w), float(h), float(OUT_W), float(OUT_H)};
        const std::string &name = methods[m];
        if (name == "Bilineaire") {
            g.dispatch(c, bilinear, {{render.view, g.linear}, {s.tmp.view}}, sizes, 16, (OUT_W + 7) / 8, (OUT_H + 7) / 8);
        } else if (name == "FSR 1 (EASU)") {
            uint32_t con[16] = {};
            float iw = float(w), ih = float(h), ow = float(OUT_W), oh = float(OUT_H);
            con[0] = asU(iw / ow); con[1] = asU(ih / oh); con[2] = asU(0.5f * iw / ow - 0.5f); con[3] = asU(0.5f * ih / oh - 0.5f);
            con[4] = asU(1.0f / iw); con[5] = asU(1.0f / ih); con[6] = asU(1.0f / iw); con[7] = asU(-1.0f / ih);
            con[8] = asU(-1.0f / iw); con[9] = asU(2.0f / ih); con[10] = asU(1.0f / iw); con[11] = asU(2.0f / ih);
            con[12] = asU(0.0f / iw); con[13] = asU(4.0f / ih);
            g.dispatch(c, easu, {{render.view, g.linear}, {s.tmp.view}}, con, 64, (OUT_W + 15) / 16, (OUT_H + 15) / 16);
        } else if (name == "Natif 1280x800") {
            float rect[4];
            sceneRect(rect, panPos);
            struct { float r[4]; int32_t sz[2]; int32_t aa, pad; } pc{{rect[0], rect[1], rect[2], rect[3]}, {int32_t(OUT_W), int32_t(OUT_H)}, aa ? 1 : 0, 0};
            g.dispatch(c, scene, {{picture.view, g.linear}, {s.tmp.view}}, &pc, 32, (OUT_W + 7) / 8, (OUT_H + 7) / 8);
        } else {
            Ravu &r = ravus[size_t(m - 2)];
            if (r.lite) {
                const float pcLite[4] = {float(w), float(h), float(2 * w), float(2 * h)};
                g.dispatch(c, r.main, {{render.view, g.linear}, {s.x2.view}, {r.lut.view, g.nearest}}, pcLite, 16,
                           (w + r.gw - 1) / r.gw, (h + r.gh - 1) / r.gh);
                g.dispatch(c, r.final, {{render.view, g.linear}, {s.tmp.view}, {s.x2.view, g.linear}}, sizes, 16,
                           (OUT_W + 7) / 8, (OUT_H + 7) / 8);
            } else {
                g.dispatch(c, r.main, {{render.view, g.linear}, {s.tmp.view}, {r.lut.view, g.linear}}, sizes, 16,
                           (OUT_W + 31) / 32, (OUT_H + 7) / 8);
            }
        }
        // Netteté RCAS comme dans gamescope (curseur 0..20 -> sharpness = valeur / 10 « stops », 0 = maximum).
        uint32_t con[4] = {0, 0, 0, 0};
        if (sharp >= 0 && name != "Natif 1280x800") {
            float s = std::exp2(-float(sharp) / 10.0f);
            con[0] = asU(s);
            uint16_t hs = toHalf(s);
            con[1] = uint32_t(hs) | (uint32_t(hs) << 16);
        }
        g.dispatch(c, rcas, {{s.tmp.view, g.nearest}, {s.out.view}}, con, 16, (OUT_W + 15) / 16, (OUT_H + 15) / 16);
    }

    void writeText()
    {
        char lines[4][81] = {};
        const auto [w, h] = RES[resIndex];
        char ta[48], tb[48];
        std::snprintf(ta, sizeof ta, "%s %.2f ms", methods[methodA].c_str(), msA);
        std::snprintf(tb, sizeof tb, "%s %.2f ms", methods[methodB].c_str(), msB);
        if (split)
            std::snprintf(lines[0], 81, "G: %s | D: %s", ta, tb);
        else
            std::snprintf(lines[0], 81, "%s", tb);
        char sh[16];
        if (sharp < 0) std::snprintf(sh, sizeof sh, "off");
        else std::snprintf(sh, sizeof sh, "%d", sharp);
        std::snprintf(lines[1], 81, "Rendu %ux%u x%.2f | AA %s | Nettete %s | Zoom x%d | %s%s", w, h, double(OUT_W) / w,
                      aa ? "oui" : "non", sh, zoom, pics[picIndex].name.c_str(), paused ? " | pause" : "");
        if (help) {
            std::snprintf(lines[2], 81, "A/X: methode D/G  Y: partage  Haut/Bas: rendu  L1/R1: nettete");
            std::snprintf(lines[3], 81, "B: pause  Vue: AA  Menu: image  R3: loupe  L3: aide");
        }
        // La police n'a que l'ASCII : lettres accentuées UTF-8 (U+00C0-U+00FF) ramenées à leur lettre de base.
        static const char latin[] = "AAAAAAACEEEEIIIIDNOOOOOxOUUUUYTsaaaaaaaceeeeiiiidnooooo/ouuuuyty";
        uint32_t *t = static_cast<uint32_t *>(textBuf.map);
        for (int l = 0; l < 4; ++l) {
            int c = 0;
            for (const unsigned char *q = reinterpret_cast<const unsigned char *>(lines[l]); *q && c < 80; ++q) {
                unsigned char ch = *q;
                if (ch == 0xC3 && q[1] >= 0x80 && q[1] <= 0xBF) ch = uint8_t(latin[(*++q) - 0x80]);
                else if (ch >= 0x80) { while ((q[1] & 0xC0) == 0x80) ++q; ch = '?'; }
                t[l * 80 + c++] = ch;
            }
            for (; c < 80; ++c) t[l * 80 + c] = 0;
        }
    }

    // Enregistre une image complète dans finalImg. Renvoie après soumission (fence signalée à la fin).
    void recordFrame(VkCommandBuffer c)
    {
        const auto [w, h] = RES[resIndex];
        vkCmdResetQueryPool(c, queries, 0, 4);
        float rect[4];
        sceneRect(rect, panPos);
        struct { float r[4]; int32_t sz[2]; int32_t aa, pad; } pc{{rect[0], rect[1], rect[2], rect[3]}, {int32_t(w), int32_t(h)}, aa ? 1 : 0, 0};
        g.dispatch(c, scene, {{picture.view, g.linear}, {render.view}}, &pc, 32, (w + 7) / 8, (h + 7) / 8);
        if (split) {
            vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
            runMethod(c, methodA, sides[0]);
            vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
        }
        vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 2);
        runMethod(c, methodB, sides[1]);
        vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 3);
        struct { int32_t sz[2]; int32_t split, zoom; float center[2]; } cp{{int32_t(OUT_W), int32_t(OUT_H)}, split ? int32_t(OUT_W / 2) : 0,
                                                                            zoom, {OUT_W * 0.5f, OUT_H * 0.5f}};
        g.dispatch(c, compose, {{sides[0].out.view, g.nearest}, {sides[1].out.view, g.nearest}, {finalImg.view}}, &cp, 24,
                   (OUT_W + 7) / 8, (OUT_H + 7) / 8);
        writeText();
        int32_t tp[4] = {12, 10, 4, 80};
        g.dispatch(c, text, {{finalImg.view}, {VK_NULL_HANDLE, VK_NULL_HANDLE, textBuf.buf}}, tp, 16, (OUT_W + 7) / 8, (4 * 18 + 10 + 7) / 8);
    }

    void readTimings()
    {
        uint64_t ts[4];
        if (vkGetQueryPoolResults(g.device, queries, 0, 4, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return;
        const double k = g.props.limits.timestampPeriod * 1e-6;
        auto smooth = [](float &v, double x) { v = v == 0 ? float(x) : float(0.9 * v + 0.1 * x); };
        if (split) smooth(msA, double(ts[1] - ts[0]) * k);
        smooth(msB, double(ts[3] - ts[2]) * k);
    }

    void step(double dt)
    {
        if (paused) return;
        panPos += panDir * dt * 20.0;  // 20 pixels de sortie par seconde : lent, comme un panoramique de caméra
        double m = maxPan();
        if (panPos > m) { panPos = m; panDir = -1; }
        if (panPos < 0) { panPos = 0; panDir = 1; }
    }
};

}  // namespace

int main(int argc, char **argv)
{
    try {
        App app;
        std::string exe = argv[0];
        app.root = exe.find('/') != std::string::npos ? exe.substr(0, exe.rfind('/')) : ".";
        std::string images = app.root + "/images", headlessOut;
        int frames = 30;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
            if (a == "--headless") headlessOut = next();
            else if (a == "--frames") frames = std::atoi(next().c_str());
            else if (a == "--a") app.methodA = std::atoi(next().c_str());
            else if (a == "--b") app.methodB = std::atoi(next().c_str());
            else if (a == "--aa") app.aa = next() == "1";
            else if (a == "--image") app.picIndex = std::atoi(next().c_str());
            else if (a == "--zoom") app.zoom = std::atoi(next().c_str());
            else if (a == "--res") {
                std::string r = next();
                for (size_t k = 0; k < RES.size(); ++k)
                    if (r == std::to_string(RES[k].first) + "x" + std::to_string(RES[k].second)) app.resIndex = int(k);
            } else images = a;
        }
        app.pics = loadPictures(images);
        app.picIndex = std::min<int>(app.picIndex, int(app.pics.size()) - 1);

        if (!headlessOut.empty()) {
            int ma = app.methodA, mb = app.methodB;
            app.init(false, {});
            app.methodA = std::min<int>(ma, int(app.methods.size()) - 1);
            app.methodB = std::min<int>(mb, int(app.methods.size()) - 1);
            app.help = false;
            app.resize();
            app.uploadPicture();
            Buf down = app.g.buffer(OUT_W * OUT_H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            for (int f = 0; f < frames; ++f) {
                app.step(1.0 / 60.0);
                app.oneShot([&](VkCommandBuffer c) {
                    app.recordFrame(c);
                    if (f == frames - 1) {
                        VkBufferImageCopy rb{};
                        rb.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                        rb.imageExtent = {OUT_W, OUT_H, 1};
                        vkCmdCopyImageToBuffer(c, app.finalImg.img, VK_IMAGE_LAYOUT_GENERAL, down.buf, 1, &rb);
                    }
                });
                app.readTimings();
            }
            std::ofstream o(headlessOut, std::ios::binary);
            uint32_t wh[2] = {OUT_W, OUT_H};
            o.write(reinterpret_cast<const char *>(wh), 8);
            o.write(static_cast<const char *>(down.map), OUT_W * OUT_H * 4);
            std::printf("%s | G: %s %.3f ms | D: %s %.3f ms\n", app.g.props.deviceName, app.methods[app.methodA].c_str(), app.msA,
                        app.methods[app.methodB].c_str(), app.msB);
            return 0;
        }

        // ------------------------------------------------------------------------------------------ fenêtre
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) != 0) throw std::runtime_error(SDL_GetError());
        bool fullscreen = std::getenv("SteamDeck") || std::getenv("GAMESCOPE_WAYLAND_DISPLAY");
        SDL_Window *win = SDL_CreateWindow("GrossoModo Bench", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, OUT_W, OUT_H,
                                           SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | (fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0));
        if (!win) throw std::runtime_error(SDL_GetError());
        unsigned ne = 0;
        SDL_Vulkan_GetInstanceExtensions(win, &ne, nullptr);
        std::vector<const char *> exts(ne);
        SDL_Vulkan_GetInstanceExtensions(win, &ne, exts.data());
        app.init(true, exts);
        VkSurfaceKHR surface;
        if (!SDL_Vulkan_CreateSurface(win, app.g.instance, &surface)) throw std::runtime_error(SDL_GetError());
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(app.g.phys, app.g.family, surface, &present);
        if (!present) throw std::runtime_error("la file ne peut pas présenter");

        VkSwapchainKHR swap = VK_NULL_HANDLE;
        std::vector<VkImage> swapImages;
        VkExtent2D extent{};
        auto makeSwapchain = [&]() {
            vkDeviceWaitIdle(app.g.device);
            VkSurfaceCapabilitiesKHR caps;
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(app.g.phys, surface, &caps);
            uint32_t nf = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(app.g.phys, surface, &nf, nullptr);
            std::vector<VkSurfaceFormatKHR> fmts(nf);
            vkGetPhysicalDeviceSurfaceFormatsKHR(app.g.phys, surface, &nf, fmts.data());
            VkSurfaceFormatKHR fmt = fmts[0];
            for (auto f : fmts)  // UNORM : les valeurs sont déjà encodées sRGB, pas de seconde conversion
                if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) { fmt = f; break; }
            extent = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent : VkExtent2D{OUT_W, OUT_H};
            VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
            sci.surface = surface;
            sci.minImageCount = std::max(caps.minImageCount, 2u);
            if (caps.maxImageCount) sci.minImageCount = std::min(sci.minImageCount, caps.maxImageCount);
            sci.imageFormat = fmt.format;
            sci.imageColorSpace = fmt.colorSpace;
            sci.imageExtent = extent;
            sci.imageArrayLayers = 1;
            sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            sci.preTransform = caps.currentTransform;
            sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
            sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
            sci.clipped = VK_TRUE;
            sci.oldSwapchain = swap;
            VkSwapchainKHR next;
            check(vkCreateSwapchainKHR(app.g.device, &sci, nullptr, &next), "vkCreateSwapchainKHR");
            if (swap) vkDestroySwapchainKHR(app.g.device, swap, nullptr);
            swap = next;
            uint32_t n = 0;
            vkGetSwapchainImagesKHR(app.g.device, swap, &n, nullptr);
            swapImages.resize(n);
            vkGetSwapchainImagesKHR(app.g.device, swap, &n, swapImages.data());
        };
        makeSwapchain();
        VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkSemaphore acquired, rendered;
        check(vkCreateSemaphore(app.g.device, &semi, nullptr, &acquired), "vkCreateSemaphore");
        check(vkCreateSemaphore(app.g.device, &semi, nullptr, &rendered), "vkCreateSemaphore");
        for (int i = 0; i < SDL_NumJoysticks(); ++i)
            if (SDL_IsGameController(i)) SDL_GameControllerOpen(i);

        auto last = std::chrono::steady_clock::now();
        bool running = true, swapDirty = false;
        const int nm = int(app.methods.size());
        while (running) {
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                int action = 0;  // 1 droite, 2 gauche, 3 partage, 4 pause, 5 rendu+, 6 rendu-, 7 netteté-, 8 netteté+, 9 AA, 10 image, 11 loupe, 12 aide, 13 quitter
                if (e.type == SDL_QUIT) running = false;
                else if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) swapDirty = true;
                else if (e.type == SDL_CONTROLLERDEVICEADDED) SDL_GameControllerOpen(e.cdevice.which);
                else if (e.type == SDL_KEYDOWN) {
                    switch (e.key.keysym.sym) {
                    case SDLK_2: case SDLK_TAB: action = 1; break;
                    case SDLK_1: action = 2; break;
                    case SDLK_s: action = 3; break;
                    case SDLK_SPACE: action = 4; break;
                    case SDLK_UP: action = 5; break;
                    case SDLK_DOWN: action = 6; break;
                    case SDLK_LEFTBRACKET: action = 7; break;
                    case SDLK_RIGHTBRACKET: action = 8; break;
                    case SDLK_m: action = 9; break;
                    case SDLK_n: action = 10; break;
                    case SDLK_z: action = 11; break;
                    case SDLK_h: action = 12; break;
                    case SDLK_ESCAPE: case SDLK_q: action = 13; break;
                    }
                } else if (e.type == SDL_CONTROLLERBUTTONDOWN) {
                    switch (e.cbutton.button) {
                    case SDL_CONTROLLER_BUTTON_A: action = 1; break;
                    case SDL_CONTROLLER_BUTTON_X: action = 2; break;
                    case SDL_CONTROLLER_BUTTON_Y: action = 3; break;
                    case SDL_CONTROLLER_BUTTON_B: action = 4; break;
                    case SDL_CONTROLLER_BUTTON_DPAD_UP: action = 5; break;
                    case SDL_CONTROLLER_BUTTON_DPAD_DOWN: action = 6; break;
                    case SDL_CONTROLLER_BUTTON_LEFTSHOULDER: action = 7; break;
                    case SDL_CONTROLLER_BUTTON_RIGHTSHOULDER: action = 8; break;
                    case SDL_CONTROLLER_BUTTON_BACK: action = 9; break;
                    case SDL_CONTROLLER_BUTTON_START: action = 10; break;
                    case SDL_CONTROLLER_BUTTON_RIGHTSTICK: action = 11; break;
                    case SDL_CONTROLLER_BUTTON_LEFTSTICK: action = 12; break;
                    }
                }
                switch (action) {
                case 1: app.methodB = (app.methodB + 1) % nm; app.msB = 0; break;
                case 2: app.methodA = (app.methodA + 1) % nm; app.msA = 0; break;
                case 3: app.split = !app.split; break;
                case 4: app.paused = !app.paused; break;
                case 5: if (app.resIndex + 1 < int(RES.size())) { app.resIndex++; app.needResize = true; app.msA = app.msB = 0; } break;
                case 6: if (app.resIndex > 0) { app.resIndex--; app.needResize = true; app.msA = app.msB = 0; } break;
                case 7: app.sharp = std::max(-1, app.sharp - 1); break;  // vers -1 = netteté désactivée
                case 8: app.sharp = std::min(20, app.sharp + 1); break;
                case 9: app.aa = !app.aa; break;
                case 10: app.picIndex = (app.picIndex + 1) % int(app.pics.size()); app.needUpload = true; break;
                case 11: app.zoom = app.zoom >= 4 ? 1 : app.zoom * 2; break;
                case 12: app.help = !app.help; break;
                case 13: running = false; break;
                }
            }
            if (app.needResize) app.resize();
            if (app.needUpload) app.uploadPicture();
            if (swapDirty) { makeSwapchain(); swapDirty = false; }
            auto now = std::chrono::steady_clock::now();
            app.step(std::min(0.1, std::chrono::duration<double>(now - last).count()));
            last = now;

            check(vkWaitForFences(app.g.device, 1, &app.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
            app.readTimings();
            uint32_t idx;
            VkResult ar = vkAcquireNextImageKHR(app.g.device, swap, UINT64_MAX, acquired, VK_NULL_HANDLE, &idx);
            if (ar == VK_ERROR_OUT_OF_DATE_KHR) { swapDirty = true; continue; }
            if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) check(ar, "vkAcquireNextImageKHR");
            check(vkResetFences(app.g.device, 1, &app.fence), "vkResetFences");
            check(vkResetDescriptorPool(app.g.device, app.g.pool, 0), "vkResetDescriptorPool");
            check(vkResetCommandBuffer(app.cb, 0), "vkResetCommandBuffer");
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            check(vkBeginCommandBuffer(app.cb, &bi), "vkBeginCommandBuffer");
            app.recordFrame(app.cb);
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = swapImages[idx];
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(app.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
            VkImageBlit blit{};
            blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[1] = {int32_t(OUT_W), int32_t(OUT_H), 1};
            // garder les proportions 16:10 (bandes noires sinon)
            float s = std::min(float(extent.width) / OUT_W, float(extent.height) / OUT_H);
            int32_t bw = int32_t(OUT_W * s), bh = int32_t(OUT_H * s);
            int32_t ox = (int32_t(extent.width) - bw) / 2, oy = (int32_t(extent.height) - bh) / 2;
            VkClearColorValue black{};
            VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(app.cb, swapImages[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rng);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(app.cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            blit.dstOffsets[0] = {ox, oy, 0};
            blit.dstOffsets[1] = {ox + bw, oy + bh, 1};
            vkCmdBlitImage(app.cb, app.finalImg.img, VK_IMAGE_LAYOUT_GENERAL, swapImages[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &blit, (bw == int32_t(OUT_W) && bh == int32_t(OUT_H)) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask = 0;
            b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            vkCmdPipelineBarrier(app.cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
            check(vkEndCommandBuffer(app.cb), "vkEndCommandBuffer");
            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            sub.waitSemaphoreCount = 1;
            sub.pWaitSemaphores = &acquired;
            sub.pWaitDstStageMask = &waitStage;
            sub.commandBufferCount = 1;
            sub.pCommandBuffers = &app.cb;
            sub.signalSemaphoreCount = 1;
            sub.pSignalSemaphores = &rendered;
            check(vkQueueSubmit(app.g.queue, 1, &sub, app.fence), "vkQueueSubmit");
            VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &rendered;
            pi.swapchainCount = 1;
            pi.pSwapchains = &swap;
            pi.pImageIndices = &idx;
            VkResult pr = vkQueuePresentKHR(app.g.queue, &pi);
            if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) swapDirty = true;
        }
        vkDeviceWaitIdle(app.g.device);
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "erreur : %s\n", e.what());
        return 1;
    }
}
