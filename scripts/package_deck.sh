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
# Comparaison LS1 : rien de Lossless Scaling dans l'archive, run.sh lit le Lossless.dll installé par Steam.
g++ -O2 -std=c++17 -static-libstdc++ -static-libgcc harness/ls1_vk.cpp -lvulkan -o "$OUT/ls1_vk"
gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I/usr/include/vkd3d harness/ls1_translate.c -ldl -o "$OUT/ls1_translate"
cp scripts/ls1_extract.py "$OUT/"
# RAVU-Zoom (LGPL-3, third_party/mpv-prescalers), variantes luminance : une passe, une ou deux tables.
g++ -O2 -std=c++17 -static-libstdc++ -static-libgcc harness/ravu_vk.cpp -lvulkan -o "$OUT/ravu_vk"
for r in ravu-zoom-r2 ravu-lite-r2 ravu-lite-r3 ravu-lite-ar-r3 ravu-lite-r4; do
  python3 -I scripts/gen_ravu_vk.py third_party/mpv-prescalers/$r.hook "$OUT/$r" >/dev/null
  rm -f "$OUT/$r/"*.comp
done
mkdir -p "$OUT/licences" && cp third_party/mpv-prescalers/LICENSE-LGPL-3.0.txt third_party/mpv-prescalers/SOURCE.md "$OUT/licences/"

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
  # LS1 : DLL lu là où Steam l'installe, traduit par le libvkd3d-shader du Steam Linux Runtime, dans un dossier temporaire.
  LS1=""
  DLL="${LS1_DLL:-$HOME/.steam/steam/steamapps/common/Lossless Scaling/Lossless.dll}"
  if [ -f "$DLL" ]; then
    LS1=$(mktemp -d); trap 'rm -rf "$LS1"' EXIT
    ok=""
    for lib in ${VKD3D_SHADER_LIB:-} "$HOME"/.steam/steam/steamapps/common/SteamLinuxRuntime*/*/files/lib/x86_64-linux-gnu/libvkd3d-shader.so.1* \
               "$HOME"/.steam/steam/steamapps/common/SteamLinuxRuntime*/*/usr/lib/x86_64-linux-gnu/libvkd3d-shader.so.1* \
               /usr/lib/libvkd3d-shader.so.1 /usr/lib/x86_64-linux-gnu/libvkd3d-shader.so.1; do
      [ -f "$lib" ] || continue
      if VKD3D_SHADER_LIB="$lib" python3 -I ls1_extract.py --dll "$DLL" --out "$LS1" --translate ./ls1_translate >/dev/null 2>&1; then
        echo "LS1 : $(sha256sum "$DLL" | cut -c1-16), traduit avec $lib"; ok=1; break
      fi
    done
    [ -n "$ok" ] || { echo "LS1 : aucun libvkd3d-shader récent trouvé, LS1 ignoré"; LS1=""; }
  else
    echo "LS1 : $DLL absent, LS1 ignoré"
  fi
  for src in 640x400 720x450 800x500 960x600 1024x640; do
    echo; echo "## $src -> 1280x800"
    ./harness shaders/cs_easu_fp16.spv in_$src.rgba /tmp/out.rgba 1280 800 200
    ./fsrcnnx_vk fsrcnnx8 in_$src.rgba /tmp/out.rgba 1280 800 200 fp16 strength=0.5
    for r in ravu-zoom-r2 ravu-lite-r2 ravu-lite-r3 ravu-lite-ar-r3 ravu-lite-r4; do ./ravu_vk $r in_$src.rgba /tmp/out.rgba 1280 800 200; done
    if [ -n "$LS1" ]; then
      ./ls1_vk "$LS1" quality 2 in_$src.rgba /tmp/out.rgba 1280 800 200
      ./ls1_vk "$LS1" performance 2 in_$src.rgba /tmp/out.rgba 1280 800 200
    fi
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
