# GrossoModo

**A learned upscaler for gamescope, the Steam Deck's compositor: RAVU in place of FSR 1, without forking gamescope
and without touching the game.**

*Grosso modo* means "roughly" in French, and *gros* means "big": it makes the image bigger, a bit hunchbacked like
its cousin Quasimodo, but with a big heart.

> Personal, experimental project. Tested on a Steam Deck (SteamOS, RADV Van Gogh) in Game Mode and Desktop Mode.
> Developer notes and every measurement, in French: [docs/JOURNAL.md](docs/JOURNAL.md).

## What it does

In Game Mode, a game running below the screen resolution with the **"Sharp" (FSR) scaling filter** is upscaled by
gamescope with AMD FSR 1: an EASU pass (upscale) followed by RCAS (sharpening, the "Sharpness" slider).

GrossoModo is a **Vulkan layer loaded into gamescope only**. When gamescope records its EASU pass, the layer records
instead **RAVU-Lite** (bjin's learned prescaler for mpv, anti-ringing, radius 3) in two Vulkan compute passes: a x2 of
the luma at the game's resolution, then a pass at the output size that resamples it (Catmull-Rom, skipped for an exact
x2) and restores the color. Gamescope then carries on as usual with RCAS and composition.

RAVU is not a neural network: it picks, for each pixel, a filter learned offline for the local edge direction,
strength and coherence. Earlier versions ran the FSRCNNX 8 network; RAVU-Lite gives the same image quality for a third
of the cost (see *Results*).

- **No gamescope fork**: Valve's binary is untouched and keeps its updates.
- **Nothing in the game process**: the layer is enabled through a Vulkan loader settings file that targets
  `/usr/bin/gamescope` only. Games never load it. (No anti-cheat vendor has reviewed this; see *Online games*.)
- **Same menu**: you keep using the Quick Access Menu's "Sharp" filter and "Sharpness" slider.
- **Safe fallback**: HDR layers, unknown images, any error, or `enabled=0` give back gamescope's own FSR 1.

As far as we know, upscaling inside gamescope with a learned upscaler, through a layer instead of a fork, had not been
done before.

```
game (any API) ──► gamescope composites ──► "Sharp" filter
                                             ├─ original: EASU (FSR 1) ─► RCAS ─► screen
                                             └─ GrossoModo: RAVU-Lite x2 (luma) ─► resample + color ─► RCAS ─► screen
```

## Results

### RAVU-Lite (current) on the Steam Deck

800x500 -> 1280x800, before RCAS, 5 4K screenshots with a simulated slow pan (`results/compare-ravu-lite-2026-11-13.txt`,
GPU time from `results/deck-2026-11-13-rapport7.txt`):

| Method | Deck GPU | PSNR | detail | shimmer |
|---|---|---|---|---|
| FSR 1 (EASU) | 0.52 ms | 35.43 dB | 59 % | 1.534 |
| **RAVU-Lite AR r3 (GrossoModo)** | **1.13 ms** | **35.93 dB** | 72 % | **1.460** |
| LS1 Quality (Lossless Scaling, sharpness 0) | 1.28 ms | 35.64 dB | 75 % | 1.571 |
| FSRCNNX 8 blended 0.5 (previous GrossoModo) | 3.19 ms | 35.98 dB | 70 % | 1.494 |

Exact x2 (640x400 -> 1280x800): 0.70 ms. 922x576 -> 1280x800: 36.93 dB, 81 % detail, shimmer 1.372 (FSR 1: 36.49 dB,
68 %, 1.443). The layer replaces EASU, so the extra cost over FSR 1 is about 0.6 ms at 800x500.

### History: FSRCNNX 8 (previous versions)

### Cost on the Steam Deck (GPU timestamps, median of 200 runs, `results/`)

| 800x450 -> 1280x720 | GPU time |
|---|---|
| EASU FP16 (gamescope's FSR 1) | 0.47 ms |
| RCAS 1280x720 (both cases) | 0.16 ms |
| **FSRCNNX 8 FP16 (7 passes) + Catmull-Rom downscale** | **2.21 ms** (network 1.92 + downscale 0.29) |
| FSRCNNX at 1024x576 input | 3.39 ms |
| FSRCNNX + blend, 640x400 -> 1280x800, direct x2 path (measured, report 4) | 2.11 ms |
| FSRCNNX + blend, 800x500 -> 1280x800 (measured, report 4) | 3.21 ms |
| EASU FP16 at 1280x800 output, any input (replaced by GrossoModo) | 0.52 ms |

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

## Which render resolution for a 1280x800 Steam Deck?

*This section was measured with the previous FSRCNNX version ("GrossoModo" below = FSRCNNX blended 0.5). With
RAVU-Lite the recommendations hold (800x500 recommended, 640x400 for performance, 576p renders benefit too) and the
cost is about a third; see the RAVU table above.*

Measured with `scripts/sweep_800p.py` (raw numbers in `results/sweep-800p-2026-11-13.txt`): 16:10 renders upscaled
to 1280x800, 5 4K screenshots cropped to 16:10, simulated slow pan, before RCAS. "Detail" is the fine detail kept
(100 % = reference), "shimmer" the temporal error (lower is better). GrossoModo cost on the Deck is extrapolated from
the measured passes (network proportional to render pixels; x2 exact skips the downscale pass). **These estimates
were too low**: measured on the Deck (`results/deck-2026-11-13-rapport4.txt`, strength 0.5), the full chain costs
2.11 ms at 640x400, 2.85 ms at 720x450, 3.21 ms at 800x500, 4.15 ms at 960x600 and 4.57 ms at 1024x640, because
the last pass (downscale + EASU computed for the blend) costs ~1.07 ms at 1280x800. GrossoModo replaces gamescope's
EASU (0.52 ms), so the extra cost over FSR 1 is about 0.5 ms less.

| Render | Scale | Pixels | GrossoModo cost | FSR 1: PSNR / detail / shimmer | GrossoModo (strength 0.5): PSNR / detail / shimmer |
|---|---|---|---|---|---|
| **640x400** | x2.00 | 25 % | **1.37 ms** (direct x2) | 33.95 dB / 46 % / 1.647 | 34.47 dB / 57 % / 1.610 |
| 720x450 | x1.78 | 32 % | 2.05 ms | 34.61 dB / 53 % / 1.604 | 35.11 dB / 63 % / 1.570 |
| **800x500** | x1.60 | 39 % | **2.45 ms** | 35.43 dB / 59 % / 1.534 | **35.98 dB / 70 % / 1.494** |
| 854x534 | x1.50 | 45 % | 2.75 ms | 35.83 dB / 63 % / 1.497 | 36.35 dB / 74 % / 1.456 |
| 960x600 | x1.33 | 56 % | 3.39 ms | 37.08 dB / 71 % / 1.392 | 37.67 dB / 82 % / 1.347 |
| 1024x640 | x1.25 | 64 % | 3.82 ms | 37.44 dB / 74 % / 1.344 | 37.99 dB / 85 % / 1.302 |

At every resolution GrossoModo gains about **+0.5 dB, +11 points of detail and a little less shimmer** over FSR 1.
Its cost grows with the render resolution, so the choice is about where those ~2 ms are best spent:

- **800x500 (x1.6): the recommended setting.** GrossoModo at 800x500 beats FSR 1 at 854x534 on every measure (fidelity,
  detail and shimmer) while the game renders 13 % fewer pixels. It is also where FSRCNNX was measured most useful.
- **640x400 (x2): the performance setting.** The exact x2 runs the network's native scale with a dedicated shader
  path (the last pass writes the final image: no x2 buffer, no downscale), so GrossoModo costs only ~1.4 ms. It
  reaches the fidelity of FSR 1 at 720x450 with 22 % fewer pixels, with more detail and less shimmer. Image quality
  is lower than at 800x500: for games that are heavy on the GPU.
- Above 960x600 (x1.33 and below), GrossoModo costs over 3 ms for a gain that FSR 1's sharpening partly hides:
  keep FSR 1 there (`enabled=0`).
- 16:9-only games: **800x450** (x1.6 to 1280x720) and **640x360** (exact x2) are the equivalents.

To get 800x500 or 640x400 in games that do not list them, set Steam's per-game **"Game resolution"** (game
properties, Steam Deck section): gamescope then presents a display of that size and most games render at it.

### Compared with LS1 (Lossless Scaling's spatial scaler)

Same sweep, LS1 run from the user's own `Lossless.dll` (`scripts/compare_ls1.py`, nothing from Lossless Scaling is
distributed; raw numbers in `results/compare-ls1-2026-11-13.txt`). LS1 applies its own sharpening (variants 0 to 4);
FSR 1 and GrossoModo are measured before RCAS. 800x500 -> 1280x800:

| Method | PSNR | detail | shimmer |
|---|---|---|---|
| FSR 1 | 35.43 dB | 59 % | 1.534 |
| **GrossoModo (strength 0.5)** | **35.98 dB** | 70 % | **1.494** |
| LS1 Quality, sharpness 0 | 35.64 dB | 75 % | 1.571 |
| LS1 Quality, sharpness 2 | 35.11 dB | 93 % | 1.646 |
| LS1 Performance, sharpness 0 | 35.45 dB | 73 % | 1.557 |

At every render resolution GrossoModo has the best fidelity and the least shimmer; LS1 shows more detail, much of it
from sharpening (above 100 % at high sharpness, i.e. over-sharpened), and shimmers more than FSR 1 too. On the Deck GPU (same report, sharpness 2) LS1 is much cheaper:

| 1280x800 output | 640x400 | 720x450 | 800x500 | 960x600 | 1024x640 |
|---|---|---|---|---|---|
| EASU (FSR 1) | 0.52 ms | 0.53 ms | 0.52 ms | 0.52 ms | 0.53 ms |
| GrossoModo (strength 0.5) | 2.11 ms | 2.85 ms | 3.21 ms | 4.15 ms | 4.57 ms |
| LS1 Quality | 0.86 ms | 1.07 ms | 1.29 ms | 1.78 ms | 1.99 ms |
| LS1 Performance | 0.45 ms | 0.54 ms | 0.64 ms | 0.88 ms | 0.98 ms |

So GrossoModo buys the best fidelity and the least shimmer for about 2 ms more than LS1 Quality; LS1 Performance
costs about as much as FSR 1, with more detail but more shimmer.

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

1. In the game (or in Steam's per-game "Game resolution"), pick **800x500** (recommended) or **640x400**
   (performance); **800x450** / **640x360** for 16:9-only games. See the section above.
2. Quick Access Menu: scaling filter **Sharp**; adjust the sharpness to taste (RAVU keeps more detail than FSR 1, so
   less sharpening is usually needed).
3. Settings in `/etc/grossomodo/grossomodo.conf`, re-read every 2 seconds, no restart needed:

```ini
enabled=1      # 0: gamescope's own FSR 1
compare=0      # 1: split screen, FSR 1 on the left, GrossoModo on the right (white line in the middle)
```

Log: `/tmp/grossomodo-1000.log` ("première image ... par ravu_lite_ar_r3" means it is working).

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

- **`bench/grossomodo_bench.cpp` — interactive comparison** (`scripts/package_bench.sh` -> `build/grossomodo-bench.tar.gz`):
  a Vulkan + SDL2 app for the Deck (Desktop Mode, or Game Mode as a non-Steam game) or a PC. A large picture (a
  built-in test pattern, plus your own PNG/JPG in `images/`) pans slowly, is "rendered" at 640x400 to 1024x640 with or
  without anti-aliasing, and is upscaled to 1280x800 by two methods side by side (bilinear, FSR 1 EASU, RAVU-Lite AR
  r3, RAVU-Lite r3, RAVU-Zoom r2, native), then sharpened by RCAS like gamescope. GPU time of each side is shown.
  Controls: A/X next method right/left, Y split screen, D-pad up/down render resolution, L1/R1 sharpness, B pause,
  View anti-aliasing, Menu next picture, R3 magnifier (keyboard: 2/1, S, arrows, [ ], Space, M, N, Z).

- `harness/mock_gamescope.cpp`: reproduces gamescope's Vulkan sequence around FSR 1 (descriptor layout, EASU then
  RCAS) to test the layer without gamescope. With the layer, the image is bit-identical to the standalone RAVU bench
  (`harness/ravu_vk.cpp`), also for an exact x2 and in split screen; with `enabled=0`, bit-identical to FSR 1.
- `scripts/gen_ravu_vk.py`: converts a RAVU mpv shader (`third_party/mpv-prescalers/`) into Vulkan shaders and tables;
  `scripts/embed_ravu_layer.py` embeds RAVU-Lite AR r3 in the layer; `harness/ravu_vk.cpp` runs and times them.
- `scripts/compare_ls1.py`, `scripts/compare_mpv.py`: comparisons with LS1 (from the user's own `Lossless.dll`, never
  distributed), RAVU, CuNNy and others.
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
| `third_party/` | gamescope shader headers (BSD-2), FidelityFX FSR 1 (MIT), FSRCNNX and RAVU (LGPL-3) |
| `results/` | raw reports from the Steam Deck |
| `docs/JOURNAL.md` | development notes and all measurements (French) |

## Limits and next steps

- Spatial upscaling: no history, no motion vectors. Aliasing in the source is upscaled too; an in-game temporal
  AA/upscaler upstream (DittoSR, FSR 2+, TAA) gives the best results.
- RAVU-Lite works on luma at x2; color is bilinear plus the luma difference, and other ratios go through a downscale.
- Next: RAVU tables (or a small network, `train/`) trained on game images with a shimmer penalty, FP16 shaders,
  skipping the upscaler below a minimum ratio.

## Credits

RAVU by bjin (mpv-prescalers); FSRCNNX by igv; FSRCNN by Dong, Loy and Tang; FidelityFX FSR 1 by AMD; gamescope by Valve; the gamescope-on-Deck
facts this relies on were checked on the device. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

License: **GPL-3.0-or-later** ([LICENSE](LICENSE)), third-party parts under their own licenses.
