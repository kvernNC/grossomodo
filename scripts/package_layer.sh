#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Construit build/grossomodo-layer.tar.gz : la couche (RAVU-Lite AR r3 embarqué) et ses fichiers d'installation.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf build/ravu_lite_ar_r3 && python3 -I scripts/gen_ravu_vk.py third_party/mpv-prescalers/ravu-lite-ar-r3.hook build/ravu_lite_ar_r3 >/dev/null
python3 -I scripts/embed_ravu_layer.py build/ravu_lite_ar_r3 layer/ravu_spv.h >/dev/null
OUT=build/grossomodo-layer
rm -rf "$OUT" && mkdir -p "$OUT"
g++ -O2 -std=c++17 -Wall -Wno-missing-field-initializers -fPIC -shared -fvisibility=hidden -fvisibility-inlines-hidden -static-libstdc++ \
    -static-libgcc -Wl,--exclude-libs,ALL -Wl,-z,defs -Ilayer layer/grossomodo_layer.cpp -o "$OUT/libVkLayer_GROSSOMODO_upscale.so"
cp layer/deck/* "$OUT/"
cp third_party/mpv-prescalers/SOURCE.md "$OUT/RAVU-SOURCE.md"
cp third_party/mpv-prescalers/LICENSE-LGPL-3.0.txt LICENSE THIRD_PARTY_NOTICES.md "$OUT/"
tar -C build -czf build/grossomodo-layer.tar.gz grossomodo-layer
ls -la build/grossomodo-layer.tar.gz
