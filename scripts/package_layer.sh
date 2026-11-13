#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Construit build/grossomodo-layer.tar.gz : la couche (shaders FSRCNNX embarqués) et ses fichiers d'installation.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf build/fsrcnnx8 && python3 -I scripts/gen_fsrcnnx_vk.py third_party/fsrcnnx/FSRCNNX_x2_8-0-4-1.glsl build/fsrcnnx8 >/dev/null
python3 -I scripts/embed_layer_shaders.py build/fsrcnnx8 layer/fsrcnnx8_spv.h >/dev/null
OUT=build/grossomodo-layer
rm -rf "$OUT" && mkdir -p "$OUT"
g++ -O2 -std=c++17 -Wall -Wno-missing-field-initializers -fPIC -shared -fvisibility=hidden -static-libstdc++ \
    -static-libgcc -Wl,--exclude-libs,ALL -Wl,-z,defs -Ilayer layer/grossomodo_layer.cpp -o "$OUT/libVkLayer_GROSSOMODO_upscale.so"
cp layer/deck/* "$OUT/"
cp third_party/fsrcnnx/SOURCE.md "$OUT/FSRCNNX-SOURCE.md"
cp third_party/fsrcnnx/LICENSE-LGPL-3.0.txt LICENSE THIRD_PARTY_NOTICES.md "$OUT/"
tar -C build -czf build/grossomodo-layer.tar.gz grossomodo-layer
ls -la build/grossomodo-layer.tar.gz
