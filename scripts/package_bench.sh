#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Construit build/grossomodo-bench.tar.gz : le comparateur interactif (Vulkan + SDL2), ses shaders et les tables RAVU.
# Sur le Deck : extraire, puis ./grossomodo-bench/grossomodo_bench (mode Bureau) ou l'ajouter en jeu non-Steam.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/grossomodo-bench
rm -rf "$OUT" && mkdir -p "$OUT/shaders" "$OUT/ravu" "$OUT/images" "$OUT/licences"
for s in bench/shaders/*.comp; do
  glslangValidator -V --quiet --target-env vulkan1.1 -Ithird_party/gamescope "$s" -o "$OUT/shaders/$(basename "$s" .comp).spv"
done
for r in ravu-lite-ar-r3 ravu-lite-r3 ravu-zoom-r2; do
  python3 -I scripts/gen_ravu_vk.py third_party/mpv-prescalers/$r.hook "$OUT/ravu/$r" >/dev/null
  rm -f "$OUT/ravu/$r/"*.comp
done
# libstdc++ statique ; SDL2 et libvulkan du système (présentes sur SteamOS).
g++ -O2 -std=c++17 -Wall -Wno-missing-field-initializers -static-libstdc++ -static-libgcc $(sdl2-config --cflags) \
    bench/grossomodo_bench.cpp -o "$OUT/grossomodo_bench" -lSDL2 -lvulkan
cp third_party/mpv-prescalers/LICENSE-LGPL-3.0.txt third_party/mpv-prescalers/SOURCE.md "$OUT/licences/"
cp LICENSE THIRD_PARTY_NOTICES.md "$OUT/licences/"
cat > "$OUT/images/LISEZMOI.txt" <<'TXT'
Mettez ici vos captures (PNG ou JPG, idéalement 4K ou 1440p) : elles s'ajoutent à la mire intégrée (bouton Menu / touche N).
TXT
tar -C build -czf build/grossomodo-bench.tar.gz grossomodo-bench
ls -la build/grossomodo-bench.tar.gz
