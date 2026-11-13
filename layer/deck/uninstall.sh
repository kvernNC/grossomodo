#!/usr/bin/env bash
# Retire GrossoModo (à lancer avec sudo). Gamescope revient au FSR 1 d'origine à son prochain démarrage.
set -eu
[ "$(id -u)" = 0 ] || { echo "à lancer avec sudo" >&2; exit 1; }
SETTINGS=/etc/vulkan/loader_settings.d/vk_loader_settings.json
if [ -e "$SETTINGS" ] && grep -q -E "VK_LAYER_GROSSOMODO_upscale|VK_LAYER_FS_upscale" "$SETTINGS"; then rm -f "$SETTINGS"; fi
rmdir /etc/vulkan/loader_settings.d 2>/dev/null || true
rm -rf /etc/grossomodo /etc/fs-upscale
echo "désinstallé."
