# GrossoModo

**A neural upscaler for gamescope, the Steam Deck's compositor: FSRCNNX in place of FSR 1, without forking gamescope
and without touching the game.**

*Grosso modo* means "roughly" in French, and *gros* means "big": it makes the image bigger, a bit hunchbacked like
its cousin Quasimodo, but with a big heart.

> Personal, experimental project. Tested on a Steam Deck (SteamOS, RADV Van Gogh) in Game Mode and Desktop Mode.
> Developer notes and every measurement, in French: [docs/JOURNAL.md](docs/JOURNAL.md).

## What it does

In Game Mode, a game running below the screen resolution with the **"Sharp" (FSR) scaling filter** is upscaled by
gamescope with AMD FSR 1: an EASU pass (upscale) followed by RCAS (sharpening, the "Sharpness" slider).

GrossoModo is a **Vulkan layer loaded into gamescope only**. When gamescope records its EASU pass, the layer records
instead an **FSRCNNX 8-0-4-1** network (igv's small CNN, the one mpv users know) as 8 Vulkan compute passes, blended
with FSR 1 by a `strength` setting. Gamescope then carries on as usual with RCAS and composition.

- **No gamescope fork**: Valve's binary is untouched and keeps its updates.
- **Nothing in the game process**: the layer is enabled through a Vulkan loader settings file that targets
  `/usr/bin/gamescope` only. Games never load it. (No anti-cheat vendor has reviewed this; see *Online games*.)
- **Same menu**: you keep using the Quick Access Menu's "Sharp" filter and "Sharpness" slider.
- **Safe fallback**: HDR layers, unknown images, any error, or `enabled=0` give back gamescope's own FSR 1.

As far as we know, upscaling inside gamescope with a neural network, through a layer instead of a fork, had not been
done before.

```
game (any API) ──► gamescope composites ──► "Sharp" filter
                                             ├─ original: EASU (FSR 1) ─► RCAS ─► screen
                                             └─ GrossoModo: FSRCNNX x2 ─► downscale ─► blend with EASU ─► RCAS ─► screen
```

## Results

### Cost on the Steam Deck (GPU timestamps, median of 200 runs, `results/`)

| 800x450 -> 1280x720 | GPU time |
|---|---|
| EASU FP16 (gamescope's FSR 1) | 0.47 ms |
| RCAS 1280x720 (both cases) | 0.16 ms |
| **FSRCNNX 8 FP16 (7 passes) + Catmull-Rom downscale** | **2.21 ms** (network 1.92 + downscale 0.29) |
| FSRCNNX at 1024x576 input | ~3 ms |

### Quality (5 4K Elden Ring screenshots downscaled to 1280x720, before RCAS)

| 800x450 -> 1280x720 | PSNR | fine detail kept |
|---|---|---|
| EASU (FSR 1) | 35.54 dB | 59 % |
| Lanczos-3 + anti-ringing | 35.93 dB | 62 % |
| **FSRCNNX 8** | **35.99 dB** | **86 %** |

At equal perceived sharpness (RCAS tuned for each), FSRCNNX beats FSR 1 by +0.8 to +0.9 dB. Spatial filters without
a learned model (Lanczos, EWA Lanczos, SGSR 1, the MAKO Scaler) all stay within a few tenths of a dB of EASU.

### Detail versus shimmer: the `strength` blend

FSRCNNX alone shows more detail but also more shimmer on aliased games. The layer outputs
`FSR1 + strength x (FSRCNNX - FSR1)`. Simulated slow pan (1/3 pixel per frame) on the same screenshots, x1.6:

| strength | PSNR | fine detail | temporal error (shimmer, lower is better) |
|---|---|---|---|
| 0 (FSR 1) | 35.33 dB | 59 % | 1.463 |
| **0.5 (default)** | **35.88 dB** | 70 % | **1.429** |
| 1 (FSRCNNX) | 35.58 dB | 84 % | 1.486 |

## Install on a Steam Deck

Requires SteamOS 3.x (gamescope with `cap_sys_nice`, Vulkan loader 1.4). Have a way back (Desktop Mode or SSH).

```sh
tar xzf grossomodo-layer.tar.gz
sudo ./grossomodo-layer/install.sh
```

It installs `/etc/grossomodo/` (layer, manifest, settings) and `/etc/vulkan/loader_settings.d/vk_loader_settings.json`
(refuses to overwrite a loader settings file that is not its own). Restart gamescope (switch to Desktop Mode and
back, or reboot). Uninstall: `sudo /etc/grossomodo/uninstall.sh`.

### Use

1. In the game, pick a resolution **below the screen**, ideally with the screen's shape: **800x500** for 16:10 games
   (x1.6 to 1280x800), **800x450** for 16:9 games such as Elden Ring.
2. Quick Access Menu: scaling filter **Sharp**, and **lower the sharpness** (FSRCNNX needs less).
3. Settings in `/etc/grossomodo/grossomodo.conf`, re-read every 2 seconds, no restart needed:

```ini
enabled=1      # 0: gamescope's own FSR 1
strength=0.5   # 0 = FSR 1, 1 = FSRCNNX alone
compare=0      # 1: split screen, FSR 1 on the left, GrossoModo on the right (white line in the middle)
```

Log: `/tmp/grossomodo-1000.log` ("première image ... par FSRCNNX 8" means it is working).

### Online games

GrossoModo never enters the game's process, the same position as gamescope's own FSR 1. That is the reason for the
design. Still, no anti-cheat vendor has reviewed it: use it online at your own risk.

## Build

From Linux, with `g++`, `glslangValidator` (glslang-tools), Python 3 with numpy and Pillow, and the Vulkan headers:

```sh
scripts/package_layer.sh      # build/grossomodo-layer.tar.gz (layer + install files)
scripts/package_deck.sh       # build/deck-bench.tar.gz (GPU timing bench to run on the Deck)
```

The layer only exports its three entry points and depends on libc only (libstdc++ linked statically and hidden).

### Benches and tests

- `harness/mock_gamescope.cpp`: reproduces gamescope's Vulkan sequence around FSR 1 (descriptor layout, EASU then
  RCAS) to test the layer without gamescope. With the layer, the image is bit-identical to the standalone FSRCNNX
  bench; with `enabled=0`, bit-identical to FSR 1.
- `harness/fsrcnnx_vk.cpp`: runs the FSRCNNX passes and times each one; `harness/harness.cpp`: single-pass shaders.
- `scripts/fsrcnnx.py`: runs an FSRCNNX mpv shader file with numpy (reference); `scripts/gen_fsrcnnx_vk.py` generates
  the Vulkan shaders from it (weights inlined, 14 mpv passes merged into 7 dispatches, FP32 and FP16 variants).
- `scripts/poc_*.py`: the quality comparisons above (`data/er_4k/` holds the screenshots, not distributed).

## Layout

| Path | Content |
|---|---|
| `layer/` | the Vulkan layer (`grossomodo_layer.cpp`) and its install files (`layer/deck/`) |
| `shaders/` | candidate upscalers with EASU's interface, RCAS (benches) |
| `harness/` | Vulkan benches and the gamescope mock |
| `scripts/` | shader generator, packaging, measurements |
| `third_party/` | gamescope shader headers (BSD-2), FidelityFX FSR 1 (MIT), FSRCNNX (LGPL-3) |
| `results/` | raw reports from the Steam Deck |
| `docs/JOURNAL.md` | development notes and all measurements (French) |

## Limits and next steps

- Spatial upscaling: no history, no motion vectors. Aliasing in the source is upscaled too; an in-game temporal
  AA/upscaler upstream (DittoSR, FSR 2+, TAA) gives the best results.
- FSRCNNX is trained for x2: other ratios go through x2 then a downscale. A network trained for x1.6 would be cheaper
  and sharper.
- Planned: skip FSRCNNX below a minimum ratio (`min_ratio`), better handling of text and color (FSRCNNX processes
  luma only), a blind temporal anti-shimmer pass.

## Credits

FSRCNNX by igv; FSRCNN by Dong, Loy and Tang; FidelityFX FSR 1 by AMD; gamescope by Valve; the gamescope-on-Deck
facts this relies on were checked on the device. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

License: **GPL-3.0-or-later** ([LICENSE](LICENSE)), third-party parts under their own licenses.
