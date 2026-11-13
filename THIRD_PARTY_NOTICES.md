# Third-party notices

GrossoModo's own code (layer, harnesses, scripts, shaders written for this project) is licensed under the
**GNU General Public License v3.0 or later** (`LICENSE`). It includes or embeds the following third-party material,
which keeps its own license:

| Material | Where | License |
|---|---|---|
| **FSRCNNX 8-0-4-1** network and weights, by igv (mpv user shader `FSRCNNX_x2_8-0-4-1.glsl`, release 1.1 of https://github.com/igv/FSRCNN-TensorFlow) | `third_party/fsrcnnx/`; translated to Vulkan compute shaders and embedded in `libVkLayer_GROSSOMODO_upscale.so` | **LGPL-3.0-or-later** (`third_party/fsrcnnx/LICENSE-LGPL-3.0.txt`, header of each file) |
| FSRCNN architecture | Dong, Loy, Tang, *Accelerating the Super-Resolution Convolutional Neural Network*, ECCV 2016 | (paper) |
| **AMD FidelityFX Super Resolution 1** (`ffx_a.h`, `ffx_fsr1.h`): EASU, used for the FSR 1 / FSRCNNX blend and the benches; RCAS, used in the benches | `third_party/gamescope/` | **MIT** (header of each file) |
| **gamescope** shader headers (`descriptor_set.h`, `colorimetry.h`, ...) and `cs_easu*.comp`, commit 865a71c of https://github.com/ValveSoftware/gamescope | `third_party/gamescope/` | **BSD-2-Clause** (`third_party/gamescope/LICENSE`) |
| **Snapdragon Game Super Resolution 1** (`sgsr1.h`, Qualcomm), benches only | `third_party/gamescope/` | **BSD-3-Clause** (header of the file) |
| MAKO Scaler reconstruction (`shaders/cs_mako_easu.comp`), benches only, from https://github.com/eugeniosegala/MAKO | `shaders/` | GPL-3.0-or-later |

Not included: no Lossless Scaling, AMD driver, game or other proprietary file. The Elden Ring screenshots used by
the benches (`data/er_4k/`) belong to their authors and are not distributed.
