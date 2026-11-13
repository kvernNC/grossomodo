// SPDX-License-Identifier: GPL-3.0-or-later
//
// VK_LAYER_GROSSOMODO_upscale : couche Vulkan chargée dans gamescope, qui remplace la passe EASU de FSR 1 par RAVU-Lite
// (bjin, mpv-prescalers ; shaders et table embarqués, voir layer/ravu_spv.h). Rien n'est modifié dans gamescope ni dans
// le jeu. Deux dispatches : RAVU-Lite fait un x2 de la luminance à la résolution du jeu (image R16F), puis une passe à
// la taille de sortie réduit ce x2 (Catmull-Rom ; lecture directe pour un x2 exact) et recompose la couleur
// (bilinéaire + écart de luminance). Gamescope applique ensuite RCAS comme d'habitude.
//
// Principe (gamescope, src/rendervulkan.cpp, CVulkanCmdBuffer::dispatch) : pour chaque dispatch, gamescope lie son
// pipeline, met à jour un descripteur neuf (liaison 1 = image de sortie, liaison 3 = échantillonneurs, image du jeu
// en case 0), le lie, puis appelle vkCmdDispatch. La couche :
//   - reconnaît les modules SPIR-V d'EASU (nom de fonction « easuPass( » dans le SPIR-V ; « FsrEasuH » = FP16) ;
//   - suit les pipelines EASU, le pipeline et le descripteur liés à chaque command buffer ;
//   - au vkCmdDispatch d'un pipeline EASU, enregistre ses 2 passes (image du jeu -> image de sortie de gamescope) au
//     lieu du dispatch d'origine. Gamescope relie tout au dispatch suivant (RCAS) et pose lui-même la barrière sur
//     l'image de sortie, qu'il croit avoir écrite.
// En cas de doute (format HDR, image inconnue, couche désactivée), le dispatch d'origine passe : c'est FSR 1.
//
// Configuration : /etc/grossomodo/grossomodo.conf (ou $GROSSOMODO_CONFIG), relue toutes les 2 s :
//   enabled=1     0 : FSR 1 d'origine
//   compare=0     1 : écran partagé, FSR 1 à gauche, RAVU à droite
// Journal : /tmp/grossomodo-<uid>.log. La couche ne s'active que dans gamescope (ou si GROSSOMODO_FORCE=1, tests).

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ravu_spv.h"

#if defined(__GNUC__)
#define FSUP_EXPORT extern "C" __attribute__((visibility("default")))
#else
#define FSUP_EXPORT extern "C"
#endif

namespace fsup {
namespace {

constexpr uint32_t kPasses = 2;          // x2 RAVU-Lite, puis réduction + couleur
constexpr uint32_t kRing = 8;            // jeux de descripteurs par passe, en rotation (images en vol de gamescope : 2-3)
constexpr uint32_t kBindings = 3;        // 0 : image du jeu ; 1 : image de stockage écrite ; 2 : table ou image x2

std::mutex g_lock;

template <typename T>
void *key(T handle) { return *reinterpret_cast<void **>(handle); }

// ---------------------------------------------------------------------------------------------------- journal, config

FILE *logFile()
{
    static FILE *f = [] {
        char path[64];
        std::snprintf(path, sizeof path, "/tmp/grossomodo-%u.log", unsigned(getuid()));
        FILE *file = std::fopen(path, "a");
        if (file) setvbuf(file, nullptr, _IOLBF, 0);
        return file;
    }();
    return f;
}

void logf(const char *fmt, ...)
{
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "[grossomodo] %s\n", msg);
    if (FILE *f = logFile()) {
        time_t t = time(nullptr);
        char ts[32];
        strftime(ts, sizeof ts, "%F %T", localtime(&t));
        std::fprintf(f, "%s [%d] %s\n", ts, int(getpid()), msg);
    }
}

bool isGamescope()
{
    static const bool yes = [] {
        if (const char *force = std::getenv("GROSSOMODO_FORCE"); force && std::strcmp(force, "1") == 0)
            return true;
        char exe[512] = {};
        ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        if (n <= 0) return false;
        const char *base = std::strrchr(exe, '/');
        base = base ? base + 1 : exe;
        return std::strncmp(base, "gamescope", 9) == 0;
    }();
    return yes;
}

struct Config {
    bool enabled = true;
    bool compare = false;  // écran partagé : FSR 1 à gauche, RAVU à droite
    time_t lastCheck = 0;
    time_t mtime = -1;
};
Config g_config;

void refreshConfig()
{
    time_t now = time(nullptr);
    if (now - g_config.lastCheck < 2) return;
    g_config.lastCheck = now;
    const char *path = std::getenv("GROSSOMODO_CONFIG");
    if (!path) path = "/etc/grossomodo/grossomodo.conf";
    struct stat st{};
    if (stat(path, &st) != 0) {
        if (g_config.mtime != 0) logf("config %s absente : RAVU actif par défaut", path);
        g_config.mtime = 0;
        g_config.enabled = true;
        return;
    }
    if (st.st_mtime == g_config.mtime) return;
    g_config.mtime = st.st_mtime;
    if (FILE *f = std::fopen(path, "r")) {
        char line[128];
        g_config.enabled = true;
        g_config.compare = false;
        while (std::fgets(line, sizeof line, f)) {
            int v;
            if (std::sscanf(line, " enabled = %d", &v) == 1) g_config.enabled = v != 0;
            if (std::sscanf(line, " compare = %d", &v) == 1) g_config.compare = v != 0;
        }
        std::fclose(f);
    }
    logf("config %s : enabled=%d compare=%d", path, int(g_config.enabled), int(g_config.compare));
}

// --------------------------------------------------------------------------------------------------- objets suivis

struct InstanceData {
    VkInstance instance{};
    PFN_vkGetInstanceProcAddr gipa{};
    PFN_vkDestroyInstance DestroyInstance{};
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties{};
};

struct Tex {
    VkImage img{};
    VkDeviceMemory mem{};
    VkImageView view{};
};

struct EasuPipe {
    bool fp16 = false;
    uint32_t colorspaceMask = 0;
};

struct SetInfo {
    VkImageView dst{}, src{};
    VkSampler srcSampler{};
};

struct CbState {
    const EasuPipe *easu = nullptr;
    VkDescriptorSet set{};
};

struct DeviceData {
    VkDevice device{};
    PFN_vkGetDeviceProcAddr gdpa{};
    VkPhysicalDeviceMemoryProperties mem{};

#define FSUP_FN(name) PFN_vk##name name{};
    FSUP_FN(DestroyDevice) FSUP_FN(CreateShaderModule) FSUP_FN(DestroyShaderModule) FSUP_FN(CreateComputePipelines)
    FSUP_FN(DestroyPipeline) FSUP_FN(CmdBindPipeline) FSUP_FN(UpdateDescriptorSets) FSUP_FN(CmdBindDescriptorSets)
    FSUP_FN(CmdDispatch) FSUP_FN(CreateImage) FSUP_FN(DestroyImage) FSUP_FN(CreateImageView) FSUP_FN(DestroyImageView)
    FSUP_FN(AllocateMemory) FSUP_FN(FreeMemory) FSUP_FN(BindImageMemory) FSUP_FN(GetImageMemoryRequirements)
    FSUP_FN(CreateDescriptorSetLayout) FSUP_FN(DestroyDescriptorSetLayout) FSUP_FN(CreatePipelineLayout)
    FSUP_FN(DestroyPipelineLayout) FSUP_FN(CreateDescriptorPool) FSUP_FN(DestroyDescriptorPool)
    FSUP_FN(AllocateDescriptorSets) FSUP_FN(CreateSampler) FSUP_FN(DestroySampler) FSUP_FN(CmdPipelineBarrier)
    FSUP_FN(DeviceWaitIdle) FSUP_FN(CmdPushConstants) FSUP_FN(CreateBuffer) FSUP_FN(DestroyBuffer)
    FSUP_FN(GetBufferMemoryRequirements) FSUP_FN(BindBufferMemory) FSUP_FN(MapMemory) FSUP_FN(CmdCopyBufferToImage)
#undef FSUP_FN

    std::unordered_map<VkShaderModule, bool> easuModules;        // module EASU -> FP16 ?
    std::unordered_map<VkPipeline, EasuPipe> easuPipes;
    std::unordered_map<VkCommandBuffer, CbState> cbs;
    std::unordered_map<VkDescriptorSet, SetInfo> sets;
    std::unordered_map<VkImageView, VkImage> views;
    std::unordered_map<VkImage, VkExtent3D> images;

    // ressources de l'agrandisseur
    bool ready = false, failed = false;
    VkDescriptorSetLayout dsl{};
    VkPipelineLayout layout{};
    VkPipeline lite{}, final{}, finalSplit{};
    VkSampler linear{}, nearest{};
    VkDescriptorPool pool{};
    VkDescriptorSet ring[kRing][kPasses]{};
    uint32_t ringPos = 0;
    Tex lut;                          // table de RAVU-Lite (rgba16f)
    VkBuffer lutBuf{};                // copie de la table en mémoire visible de l'hôte, envoyée au premier usage
    VkDeviceMemory lutMem{};
    bool lutUploaded = false;
    uint32_t srcW = 0, srcH = 0;
    Tex x2;                           // luminance x2 (R16F)
    bool needsInit = false;
    uint64_t frames = 0;
    bool loggedSkip = false;
};

std::unordered_map<void *, InstanceData> g_instances;
std::unordered_map<void *, DeviceData *> g_devices;

DeviceData *dev(void *k)
{
    auto it = g_devices.find(k);
    return it == g_devices.end() ? nullptr : it->second;
}

// -------------------------------------------------------------------------------------------------- agrandisseur

uint32_t memType(DeviceData &d, uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < d.mem.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (d.mem.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

bool makeTex(DeviceData &d, Tex &t, uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage)
{
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    if (d.CreateImage(d.device, &ci, nullptr, &t.img) != VK_SUCCESS) return false;
    VkMemoryRequirements r;
    d.GetImageMemoryRequirements(d.device, t.img, &r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = r.size;
    ai.memoryTypeIndex = memType(d, r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX || d.AllocateMemory(d.device, &ai, nullptr, &t.mem) != VK_SUCCESS) return false;
    if (d.BindImageMemory(d.device, t.img, t.mem, 0) != VK_SUCCESS) return false;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = t.img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return d.CreateImageView(d.device, &vi, nullptr, &t.view) == VK_SUCCESS;
}

void freeTex(DeviceData &d, Tex &t)
{
    if (t.view) d.DestroyImageView(d.device, t.view, nullptr);
    if (t.img) d.DestroyImage(d.device, t.img, nullptr);
    if (t.mem) d.FreeMemory(d.device, t.mem, nullptr);
    t = Tex{};
}

bool makePipeline(DeviceData &d, const uint32_t *code, size_t size, int32_t split, VkPipeline &out)
{
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = size;
    smi.pCode = code;
    VkShaderModule module;
    // appel direct au suivant : nos propres modules ne doivent pas passer par la détection d'EASU
    if (d.CreateShaderModule(d.device, &smi, nullptr, &module) != VK_SUCCESS) return false;
    VkSpecializationMapEntry e{0, 0, 4};
    VkSpecializationInfo spec{1, &e, 4, &split};
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main",
                 split ? &spec : nullptr};
    cpi.layout = d.layout;
    VkResult r = d.CreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &out);
    d.DestroyShaderModule(d.device, module, nullptr);
    return r == VK_SUCCESS;
}

bool setupPipelines(DeviceData &d)
{
    VkDescriptorSetLayoutBinding b[kBindings] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = kBindings;
    dli.pBindings = b;
    if (d.CreateDescriptorSetLayout(d.device, &dli, nullptr, &d.dsl) != VK_SUCCESS) return false;
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};  // tailles : source (xy), sortie de la passe (zw)
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &d.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (d.CreatePipelineLayout(d.device, &pli, nullptr, &d.layout) != VK_SUCCESS) return false;
    if (!makePipeline(d, k_ravu_lite, sizeof k_ravu_lite, 0, d.lite) ||
        !makePipeline(d, k_ravu_final, sizeof k_ravu_final, 0, d.final) ||
        !makePipeline(d, k_ravu_final, sizeof k_ravu_final, 1, d.finalSplit))
        return false;

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (d.CreateSampler(d.device, &si, nullptr, &d.linear) != VK_SUCCESS) return false;
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    if (d.CreateSampler(d.device, &si, nullptr, &d.nearest) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * kRing * kPasses},
                                  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kRing * kPasses}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = kRing * kPasses;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = ps;
    if (d.CreateDescriptorPool(d.device, &dpi, nullptr, &d.pool) != VK_SUCCESS) return false;
    std::vector<VkDescriptorSetLayout> layouts(kRing * kPasses, d.dsl);
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = d.pool;
    dai.descriptorSetCount = kRing * kPasses;
    dai.pSetLayouts = layouts.data();
    if (d.AllocateDescriptorSets(d.device, &dai, &d.ring[0][0]) != VK_SUCCESS) return false;

    // Table : image rgba16f + tampon visible de l'hôte, copié dans le premier command buffer qui l'utilise.
    if (!makeTex(d, d.lut, kRavuLutW, kRavuLutH, VK_FORMAT_R16G16B16A16_SFLOAT,
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        return false;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = sizeof k_ravu_lut;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (d.CreateBuffer(d.device, &bi, nullptr, &d.lutBuf) != VK_SUCCESS) return false;
    VkMemoryRequirements r;
    d.GetBufferMemoryRequirements(d.device, d.lutBuf, &r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = r.size;
    ai.memoryTypeIndex = memType(d, r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX || d.AllocateMemory(d.device, &ai, nullptr, &d.lutMem) != VK_SUCCESS) return false;
    if (d.BindBufferMemory(d.device, d.lutBuf, d.lutMem, 0) != VK_SUCCESS) return false;
    void *map = nullptr;
    if (d.MapMemory(d.device, d.lutMem, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS) return false;
    std::memcpy(map, k_ravu_lut, sizeof k_ravu_lut);  // reste mappée : libérée avec la mémoire
    return true;
}

bool ensureTextures(DeviceData &d, uint32_t w, uint32_t h)
{
    if (d.srcW == w && d.srcH == h && d.x2.img) return true;
    if (d.x2.img) {
        // changement de résolution du jeu : rare, on attend que le GPU n'utilise plus l'ancienne image
        d.DeviceWaitIdle(d.device);
        freeTex(d, d.x2);
    }
    if (!makeTex(d, d.x2, 2 * w, 2 * h, VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))
        return false;
    d.srcW = w;
    d.srcH = h;
    d.needsInit = true;
    logf("image x2 %ux%u", 2 * w, 2 * h);
    return true;
}

void barrier(DeviceData &d, VkCommandBuffer cb)
{
    VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    m.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    m.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &m, 0,
                         nullptr, 0, nullptr);
}

VkImageMemoryBarrier toGeneral(VkImage img, VkImageLayout from, VkAccessFlags src, VkAccessFlags dst)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.oldLayout = from;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return b;
}

// Journalise une raison d'abandon (une seule fois par raison) et renvoie false.
bool skip(const char *why)
{
    static const char *seen[16];  // littéraux : une comparaison d'adresse suffit
    static int n = 0;
    for (int i = 0; i < n; ++i)
        if (seen[i] == why) return false;
    if (n < 16) seen[n++] = why;
    logf("dispatch EASU laissé au FSR 1 d'origine : %s", why);
    return false;
}

// Enregistre RAVU-Lite à la place du dispatch EASU. Renvoie false pour laisser passer le dispatch d'origine.
bool recordUpscale(DeviceData &d, VkCommandBuffer cb, const CbState &st, uint32_t gx, uint32_t gy, uint32_t gz)
{
    auto si = d.sets.find(st.set);
    if (!st.set) return skip("aucun descripteur lié avant le dispatch");
    if (si == d.sets.end()) return skip("descripteur lié jamais mis à jour par gamescope");
    if (!si->second.src) return skip("pas d'image du jeu en liaison 3 case 0");
    if (!si->second.dst) return skip("pas d'image de sortie en liaison 1");
    const SetInfo &set = si->second;
    auto vs = d.views.find(set.src), vd = d.views.find(set.dst);
    if (vs == d.views.end()) return skip("vue de l'image du jeu inconnue");
    if (vd == d.views.end()) return skip("vue de l'image de sortie inconnue");
    auto is = d.images.find(vs->second), id = d.images.find(vd->second);
    if (is == d.images.end()) return skip("image du jeu inconnue (créée hors de la couche ?)");
    if (id == d.images.end()) return skip("image de sortie inconnue");
    const uint32_t sw = is->second.width, sh = is->second.height, dw = id->second.width, dh = id->second.height;
    if (dw < sw || dh < sh) return skip("réduction (sortie plus petite que l'image du jeu)");

    // HDR (scRGB, PQ) : EASU encode la sortie ; on laisse faire FSR 1.
    const uint32_t cs0 = st.easu->colorspaceMask & 7u;
    if (cs0 == 2u || cs0 == 3u) {
        if (!d.loggedSkip) logf("calque HDR (espace %u) : FSR 1 d'origine", cs0);
        d.loggedSkip = true;
        return false;
    }

    if (!d.ready) {
        if (d.failed) return false;
        if (!setupPipelines(d)) {
            logf("échec de création des pipelines RAVU : FSR 1 d'origine");
            d.failed = true;
            return false;
        }
        d.ready = true;
        logf("%s prêt (table %ux%u)", kRavuName, kRavuLutW, kRavuLutH);
    }
    if (!ensureTextures(d, sw, sh)) {
        logf("échec d'allocation de l'image x2 : FSR 1 d'origine");
        d.failed = true;
        return false;
    }

    if (!d.lutUploaded) {
        VkImageMemoryBarrier b = toGeneral(d.lut.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkBufferImageCopy reg{};
        reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        reg.imageExtent = {kRavuLutW, kRavuLutH, 1};
        d.CmdCopyBufferToImage(cb, d.lutBuf, d.lut.img, VK_IMAGE_LAYOUT_GENERAL, 1, &reg);
        VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        m.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
        d.lutUploaded = true;
    }
    if (d.needsInit) {
        VkImageMemoryBarrier b = toGeneral(d.x2.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        d.needsInit = false;
    }

    VkDescriptorSet *sets = d.ring[d.ringPos];
    d.ringPos = (d.ringPos + 1) % kRing;
    // passe 0 : image du jeu -> x2 (table) ; passe 1 : image du jeu + x2 -> image de sortie de gamescope
    VkDescriptorImageInfo ii[kPasses][kBindings] = {
        {{d.linear, set.src, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, d.x2.view, VK_IMAGE_LAYOUT_GENERAL},
         {d.nearest, d.lut.view, VK_IMAGE_LAYOUT_GENERAL}},
        {{d.linear, set.src, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, set.dst, VK_IMAGE_LAYOUT_GENERAL},
         {d.linear, d.x2.view, VK_IMAGE_LAYOUT_GENERAL}}};
    VkWriteDescriptorSet writes[kPasses * kBindings];
    for (uint32_t p = 0; p < kPasses; ++p)
        for (uint32_t k = 0; k < kBindings; ++k) {
            VkWriteDescriptorSet &w = writes[p * kBindings + k];
            w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = sets[p];
            w.dstBinding = k;
            w.descriptorCount = 1;
            w.descriptorType = k == 1 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &ii[p][k];
        }
    d.UpdateDescriptorSets(d.device, kPasses * kBindings, writes, 0, nullptr);

    if (g_config.compare) {
        // écran partagé : le FSR 1 d'origine écrit toute l'image (pipeline et descripteur de gamescope encore liés),
        // RAVU réécrit ensuite la moitié droite.
        d.CmdDispatch(cb, gx, gy, gz);
    }
    barrier(d, cb);  // l'image x2 de l'image précédente n'est plus lue ; la sortie d'EASU (écran partagé) est écrite
    const float pcLite[4] = {float(sw), float(sh), float(2 * sw), float(2 * sh)};
    const float pcFinal[4] = {float(sw), float(sh), float(dw), float(dh)};
    d.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.lite);
    d.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.layout, 0, 1, &sets[0], 0, nullptr);
    d.CmdPushConstants(cb, d.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pcLite);
    d.CmdDispatch(cb, (sw + kRavuGroupW - 1) / kRavuGroupW, (sh + kRavuGroupH - 1) / kRavuGroupH, 1);
    barrier(d, cb);
    d.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_config.compare ? d.finalSplit : d.final);
    d.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.layout, 0, 1, &sets[1], 0, nullptr);
    d.CmdPushConstants(cb, d.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pcFinal);
    d.CmdDispatch(cb, (dw + 7) / 8, (dh + 7) / 8, 1);
    if (d.frames++ == 0)
        logf("première image : %ux%u -> %ux%u par %s à la place d'EASU%s", sw, sh, dw, dh, kRavuName,
             dw == 2 * sw && dh == 2 * sh ? " (x2 exact : sans réduction)" : "");
    return true;
}

// --------------------------------------------------------------------------------------------- fonctions interceptées

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *alloc, VkInstance *out)
{
    auto *link = reinterpret_cast<const VkLayerInstanceCreateInfo *>(ci->pNext);
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && link->function == VK_LAYER_LINK_INFO))
        link = reinterpret_cast<const VkLayerInstanceCreateInfo *>(link->pNext);
    if (!link) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    const_cast<VkLayerInstanceCreateInfo *>(link)->u.pLayerInfo = link->u.pLayerInfo->pNext;
    auto create = reinterpret_cast<PFN_vkCreateInstance>(gipa(VK_NULL_HANDLE, "vkCreateInstance"));
    VkResult r = create(ci, alloc, out);
    if (r != VK_SUCCESS) return r;
    std::lock_guard<std::mutex> g(g_lock);
    InstanceData &id = g_instances[key(*out)];
    id.instance = *out;
    id.gipa = gipa;
    id.DestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(gipa(*out, "vkDestroyInstance"));
    id.GetPhysicalDeviceMemoryProperties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(gipa(*out, "vkGetPhysicalDeviceMemoryProperties"));
    if (isGamescope()) logf("chargée dans gamescope (pid %d)", int(getpid()));
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance instance, const VkAllocationCallbacks *alloc)
{
    PFN_vkDestroyInstance next;
    {
        std::lock_guard<std::mutex> g(g_lock);
        auto it = g_instances.find(key(instance));
        if (it == g_instances.end()) return;
        next = it->second.DestroyInstance;
        g_instances.erase(it);
    }
    next(instance, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice phys, const VkDeviceCreateInfo *ci,
                                            const VkAllocationCallbacks *alloc, VkDevice *out)
{
    auto *link = reinterpret_cast<const VkLayerDeviceCreateInfo *>(ci->pNext);
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && link->function == VK_LAYER_LINK_INFO))
        link = reinterpret_cast<const VkLayerDeviceCreateInfo *>(link->pNext);
    if (!link) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    const_cast<VkLayerDeviceCreateInfo *>(link)->u.pLayerInfo = link->u.pLayerInfo->pNext;
    auto create = reinterpret_cast<PFN_vkCreateDevice>(gipa(VK_NULL_HANDLE, "vkCreateDevice"));
    VkResult r = create(phys, ci, alloc, out);
    if (r != VK_SUCCESS) return r;

    auto *d = new DeviceData;
    d->device = *out;
    d->gdpa = gdpa;
#define FSUP_LOAD(name) d->name = reinterpret_cast<PFN_vk##name>(gdpa(*out, "vk" #name));
    FSUP_LOAD(DestroyDevice) FSUP_LOAD(CreateShaderModule) FSUP_LOAD(DestroyShaderModule) FSUP_LOAD(CreateComputePipelines)
    FSUP_LOAD(DestroyPipeline) FSUP_LOAD(CmdBindPipeline) FSUP_LOAD(UpdateDescriptorSets) FSUP_LOAD(CmdBindDescriptorSets)
    FSUP_LOAD(CmdDispatch) FSUP_LOAD(CreateImage) FSUP_LOAD(DestroyImage) FSUP_LOAD(CreateImageView) FSUP_LOAD(DestroyImageView)
    FSUP_LOAD(AllocateMemory) FSUP_LOAD(FreeMemory) FSUP_LOAD(BindImageMemory) FSUP_LOAD(GetImageMemoryRequirements)
    FSUP_LOAD(CreateDescriptorSetLayout) FSUP_LOAD(DestroyDescriptorSetLayout) FSUP_LOAD(CreatePipelineLayout)
    FSUP_LOAD(DestroyPipelineLayout) FSUP_LOAD(CreateDescriptorPool) FSUP_LOAD(DestroyDescriptorPool)
    FSUP_LOAD(AllocateDescriptorSets) FSUP_LOAD(CreateSampler) FSUP_LOAD(DestroySampler) FSUP_LOAD(CmdPipelineBarrier)
    FSUP_LOAD(DeviceWaitIdle) FSUP_LOAD(CmdPushConstants) FSUP_LOAD(CreateBuffer) FSUP_LOAD(DestroyBuffer)
    FSUP_LOAD(GetBufferMemoryRequirements) FSUP_LOAD(BindBufferMemory) FSUP_LOAD(MapMemory) FSUP_LOAD(CmdCopyBufferToImage)
#undef FSUP_LOAD
    std::lock_guard<std::mutex> g(g_lock);
    auto ii = g_instances.find(key(phys));
    if (ii != g_instances.end()) ii->second.GetPhysicalDeviceMemoryProperties(phys, &d->mem);
    g_devices[key(*out)] = d;
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device, const VkAllocationCallbacks *alloc)
{
    DeviceData *d;
    {
        std::lock_guard<std::mutex> g(g_lock);
        auto it = g_devices.find(key(device));
        if (it == g_devices.end()) return;
        d = it->second;
        g_devices.erase(it);
    }
    if (d->ready || d->failed) {
        d->DeviceWaitIdle(device);
        freeTex(*d, d->x2);
        freeTex(*d, d->lut);
        if (d->lutBuf) d->DestroyBuffer(device, d->lutBuf, nullptr);
        if (d->lutMem) d->FreeMemory(device, d->lutMem, nullptr);
        for (VkPipeline p : {d->lite, d->final, d->finalSplit}) if (p) d->DestroyPipeline(device, p, nullptr);
        if (d->pool) d->DestroyDescriptorPool(device, d->pool, nullptr);
        if (d->linear) d->DestroySampler(device, d->linear, nullptr);
        if (d->nearest) d->DestroySampler(device, d->nearest, nullptr);
        if (d->layout) d->DestroyPipelineLayout(device, d->layout, nullptr);
        if (d->dsl) d->DestroyDescriptorSetLayout(device, d->dsl, nullptr);
    }
    PFN_vkDestroyDevice next = d->DestroyDevice;
    delete d;
    next(device, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *ci,
                                                  const VkAllocationCallbacks *alloc, VkShaderModule *out)
{
    DeviceData *d;
    {
        std::lock_guard<std::mutex> g(g_lock);
        d = dev(key(device));
    }
    VkResult r = d->CreateShaderModule(device, ci, alloc, out);
    if (r != VK_SUCCESS || !isGamescope()) return r;
    // les noms de fonctions sont conservés par glslangValidator -V (OpName)
    const char *bytes = reinterpret_cast<const char *>(ci->pCode);
    const std::string_view code(bytes, ci->codeSize);
    if (code.find("easuPass(") != std::string_view::npos) {
        const bool fp16 = code.find("FsrEasuH") != std::string_view::npos;
        std::lock_guard<std::mutex> g(g_lock);
        d->easuModules[*out] = fp16;
        logf("module EASU reconnu (%s)", fp16 ? "FP16" : "FP32");
    }
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyShaderModule(VkDevice device, VkShaderModule module, const VkAllocationCallbacks *alloc)
{
    DeviceData *d;
    {
        std::lock_guard<std::mutex> g(g_lock);
        d = dev(key(device));
        d->easuModules.erase(module);
    }
    d->DestroyShaderModule(device, module, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(VkDevice device, VkPipelineCache cache, uint32_t n,
                                                      const VkComputePipelineCreateInfo *cis,
                                                      const VkAllocationCallbacks *alloc, VkPipeline *out)
{
    DeviceData *d;
    {
        std::lock_guard<std::mutex> g(g_lock);
        d = dev(key(device));
    }
    VkResult r = d->CreateComputePipelines(device, cache, n, cis, alloc, out);
    if (r != VK_SUCCESS) return r;
    std::lock_guard<std::mutex> g(g_lock);
    for (uint32_t i = 0; i < n; ++i) {
        auto m = d->easuModules.find(cis[i].stage.module);
        if (m == d->easuModules.end()) continue;
        EasuPipe ep;
        ep.fp16 = m->second;
        if (const VkSpecializationInfo *spec = cis[i].stage.pSpecializationInfo) {
            for (uint32_t k = 0; k < spec->mapEntryCount; ++k) {
                const VkSpecializationMapEntry &e = spec->pMapEntries[k];
                if (e.constantID == 4 && e.size == 4)  // c_colorspaceMask (descriptor_set.h de gamescope)
                    std::memcpy(&ep.colorspaceMask, static_cast<const char *>(spec->pData) + e.offset, 4);
            }
        }
        d->easuPipes[out[i]] = ep;
        logf("pipeline EASU créé (%s, espace de couleur du calque 0 : %u)", ep.fp16 ? "FP16" : "FP32", ep.colorspaceMask & 7u);
    }
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks *alloc)
{
    DeviceData *d;
    {
        std::lock_guard<std::mutex> g(g_lock);
        d = dev(key(device));
        d->easuPipes.erase(pipeline);
    }
    d->DestroyPipeline(device, pipeline, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImage(VkDevice device, const VkImageCreateInfo *ci, const VkAllocationCallbacks *alloc, VkImage *out)
{
    DeviceData *d;
    {
        std::lock_guard<std::mutex> g(g_lock);
        d = dev(key(device));
    }
    VkResult r = d->CreateImage(device, ci, alloc, out);
    if (r == VK_SUCCESS) {
        std::lock_guard<std::mutex> g(g_lock);
        d->images[*out] = ci->extent;
    }
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks *alloc)
{
    DeviceData *d;
    {
        std::lock_guard<std::mutex> g(g_lock);
        d = dev(key(device));
        d->images.erase(image);
    }
    d->DestroyImage(device, image, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImageView(VkDevice device, const VkImageViewCreateInfo *ci,
                                               const VkAllocationCallbacks *alloc, VkImageView *out)
{
    DeviceData *d;
    {
        std::lock_guard<std::mutex> g(g_lock);
        d = dev(key(device));
    }
    VkResult r = d->CreateImageView(device, ci, alloc, out);
    if (r == VK_SUCCESS) {
        std::lock_guard<std::mutex> g(g_lock);
        d->views[*out] = ci->image;
    }
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks *alloc)
{
    DeviceData *d;
    {
        std::lock_guard<std::mutex> g(g_lock);
        d = dev(key(device));
        d->views.erase(view);
    }
    d->DestroyImageView(device, view, alloc);
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(VkDevice device, uint32_t nw, const VkWriteDescriptorSet *w, uint32_t nc,
                                                const VkCopyDescriptorSet *c)
{
    DeviceData *d;
    {
        std::lock_guard<std::mutex> g(g_lock);
        d = dev(key(device));
        if (isGamescope() && !d->easuPipes.empty()) {
            for (uint32_t i = 0; i < nw; ++i) {
                if (w[i].dstBinding == 1 && w[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && w[i].pImageInfo)
                    d->sets[w[i].dstSet].dst = w[i].pImageInfo[0].imageView;
                else if (w[i].dstBinding == 3 && w[i].dstArrayElement == 0 &&
                         w[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && w[i].pImageInfo) {
                    d->sets[w[i].dstSet].src = w[i].pImageInfo[0].imageView;
                    d->sets[w[i].dstSet].srcSampler = w[i].pImageInfo[0].sampler;
                }
            }
        }
    }
    d->UpdateDescriptorSets(device, nw, w, nc, c);
}

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline pipeline)
{
    DeviceData *d;
    {
        std::lock_guard<std::mutex> g(g_lock);
        d = dev(key(cb));
        if (bp == VK_PIPELINE_BIND_POINT_COMPUTE && !d->easuPipes.empty()) {
            auto it = d->easuPipes.find(pipeline);
            d->cbs[cb].easu = it == d->easuPipes.end() ? nullptr : &it->second;
        }
    }
    d->CmdBindPipeline(cb, bp, pipeline);
}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout,
                                                 uint32_t first, uint32_t n, const VkDescriptorSet *sets, uint32_t nd,
                                                 const uint32_t *offsets)
{
    DeviceData *d;
    {
        std::lock_guard<std::mutex> g(g_lock);
        d = dev(key(cb));
        if (bp == VK_PIPELINE_BIND_POINT_COMPUTE && first == 0 && n >= 1 && !d->easuPipes.empty())
            d->cbs[cb].set = sets[0];
    }
    d->CmdBindDescriptorSets(cb, bp, layout, first, n, sets, nd, offsets);
}

VKAPI_ATTR void VKAPI_CALL CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z)
{
    DeviceData *d;
    {
        std::lock_guard<std::mutex> g(g_lock);
        d = dev(key(cb));
        auto it = d->cbs.find(cb);
        if (it != d->cbs.end() && it->second.easu) {
            static bool seen = false;
            if (!seen) logf("premier dispatch EASU vu (grille %ux%u)", x, y);
            seen = true;
            refreshConfig();
            const CbState st = it->second;
            it->second.easu = nullptr;  // un seul remplacement par liaison du pipeline EASU
            if (g_config.enabled && recordUpscale(*d, cb, st, x, y, z))
                return;
        }
    }
    d->CmdDispatch(cb, x, y, z);
}

PFN_vkVoidFunction intercepted(const char *name)
{
#define FSUP_HOOK(fn) if (!std::strcmp(name, "vk" #fn)) return reinterpret_cast<PFN_vkVoidFunction>(fn);
    FSUP_HOOK(CreateDevice) FSUP_HOOK(DestroyDevice) FSUP_HOOK(CreateShaderModule) FSUP_HOOK(DestroyShaderModule)
    FSUP_HOOK(CreateComputePipelines) FSUP_HOOK(DestroyPipeline) FSUP_HOOK(CreateImage) FSUP_HOOK(DestroyImage)
    FSUP_HOOK(CreateImageView) FSUP_HOOK(DestroyImageView) FSUP_HOOK(UpdateDescriptorSets) FSUP_HOOK(CmdBindPipeline)
    FSUP_HOOK(CmdBindDescriptorSets) FSUP_HOOK(CmdDispatch)
#undef FSUP_HOOK
    return nullptr;
}

} // namespace
} // namespace fsup

using namespace fsup;

FSUP_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fsup_GetDeviceProcAddr(VkDevice device, const char *name)
{
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(fsup_GetDeviceProcAddr);
    if (isGamescope())
        if (PFN_vkVoidFunction f = intercepted(name)) return f;
    PFN_vkGetDeviceProcAddr next;
    {
        std::lock_guard<std::mutex> g(g_lock);
        DeviceData *d = dev(key(device));
        if (!d) return nullptr;
        next = d->gdpa;
    }
    return next(device, name);
}

FSUP_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fsup_GetInstanceProcAddr(VkInstance instance, const char *name)
{
    if (!std::strcmp(name, "vkGetInstanceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(fsup_GetInstanceProcAddr);
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(fsup_GetDeviceProcAddr);
    if (!std::strcmp(name, "vkCreateInstance")) return reinterpret_cast<PFN_vkVoidFunction>(CreateInstance);
    if (!std::strcmp(name, "vkDestroyInstance")) return reinterpret_cast<PFN_vkVoidFunction>(DestroyInstance);
    if (isGamescope())
        if (PFN_vkVoidFunction f = intercepted(name)) return f;
    if (!instance) return nullptr;
    PFN_vkGetInstanceProcAddr next;
    {
        std::lock_guard<std::mutex> g(g_lock);
        auto it = g_instances.find(key(instance));
        if (it == g_instances.end()) return nullptr;
        next = it->second.gipa;
    }
    return next(instance, name);
}

FSUP_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *v)
{
    if (v->loaderLayerInterfaceVersion < 2) return VK_ERROR_INITIALIZATION_FAILED;
    v->loaderLayerInterfaceVersion = 2;
    v->pfnGetInstanceProcAddr = fsup_GetInstanceProcAddr;
    v->pfnGetDeviceProcAddr = fsup_GetDeviceProcAddr;
    v->pfnGetPhysicalDeviceProcAddr = nullptr;
    return VK_SUCCESS;
}
