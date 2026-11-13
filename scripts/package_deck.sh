#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Construit build/deck-bench/ (et build/deck-bench.tar.gz) : de quoi mesurer sur le Steam Deck le temps GPU d'EASU,
# de RCAS, de Lanczos-3 et de FSRCNNX 8 (FP32 / FP16), sans rien installer. Images d'entrée synthétiques (libres).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/deck-bench
rm -rf "$OUT" && mkdir -p "$OUT/shaders"

for s in third_party/gamescope/cs_easu.comp third_party/gamescope/cs_easu_fp16.comp shaders/cs_lanczos3_easu.comp shaders/cs_rcas.comp; do
  glslangValidator -V --target-env vulkan1.3 -Ithird_party/gamescope "$s" -o "$OUT/shaders/$(basename "$s" .comp).spv" --quiet
done
python3 -I scripts/gen_fsrcnnx_vk.py third_party/fsrcnnx/FSRCNNX_x2_8-0-4-1.glsl "$OUT/fsrcnnx8" >/dev/null
rm -f "$OUT"/fsrcnnx8/*.comp

# libstdc++ et libgcc liées en statique : seuls la glibc et libvulkan.so.1 du système sont nécessaires.
g++ -O2 -std=c++17 -static-libstdc++ -static-libgcc harness/harness.cpp -lvulkan -o "$OUT/harness"
g++ -O2 -std=c++17 -static-libstdc++ -static-libgcc harness/fsrcnnx_vk.cpp -lvulkan -o "$OUT/fsrcnnx_vk"

python3 -I - "$OUT" <<'PY'
import sys; sys.path.insert(0, "scripts")
import poc_a
from PIL import Image
img = poc_a.synthetic_images()["texture_feuillage"]
for w, h in ((640, 400), (720, 450), (800, 500), (960, 600), (1024, 640), (1280, 800)):
    poc_a.write_rgba(f"{sys.argv[1]}/in_{w}x{h}.rgba", img.resize((w, h), Image.BOX))
PY

cat > "$OUT/run.sh" <<'RUN'
#!/usr/bin/env bash
# Mesures GPU sur le Steam Deck (mode Bureau ou SSH, jeu fermé), sortie 1280x800. Résultat dans rapport.txt.
cd "$(dirname "$0")"
R=rapport.txt
{
  echo "== $(date -Iseconds) $(uname -r)"
  grep -m1 -E "^PRETTY_NAME" /etc/os-release || true
  for src in 640x400 720x450 800x500 960x600 1024x640; do
    echo; echo "## $src -> 1280x800"
    ./harness shaders/cs_easu_fp16.spv in_$src.rgba /tmp/out.rgba 1280 800 200
    ./fsrcnnx_vk fsrcnnx8 in_$src.rgba /tmp/out.rgba 1280 800 200 fp16 strength=0.5
  done
  echo; echo "## RCAS 1280x800 (netteté 2)"
  ./harness shaders/cs_rcas.spv in_1280x800.rgba /tmp/out.rgba 1280 800 200 rcas=0.2
} 2>&1 | tee "$R"
echo "-> $(pwd)/$R"
RUN
chmod +x "$OUT/run.sh"
cp ../LICENSE-NOTE.md "$OUT/" 2>/dev/null || true
tar -C build -czf build/deck-bench.tar.gz deck-bench
ls -la build/deck-bench.tar.gz
objdump -T "$OUT/harness" "$OUT/fsrcnnx_vk" | grep -o "GLIBC_[0-9.]*" | sort -Vu | tail -1
