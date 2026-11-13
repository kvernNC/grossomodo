#!/usr/bin/env bash
# Installe GrossoModo (VK_LAYER_GROSSOMODO_upscale) pour gamescope uniquement. À lancer avec sudo.
# Remplace une ancienne installation « fs-upscale » (nom de travail du projet) si elle est présente.
set -euo pipefail
cd "$(dirname "$0")"
[ "$(id -u)" = 0 ] || { echo "à lancer avec sudo" >&2; exit 1; }
SETTINGS=/etc/vulkan/loader_settings.d/vk_loader_settings.json
if [ -e "$SETTINGS" ]; then
  if grep -q VK_LAYER_FS_upscale "$SETTINGS"; then
    echo "ancienne installation fs-upscale trouvée : remplacée"
    rm -rf /etc/fs-upscale
  elif ! grep -q VK_LAYER_GROSSOMODO_upscale "$SETTINGS"; then
    echo "$SETTINGS existe déjà et n'est pas le nôtre : installation arrêtée (rien n'a été modifié)." >&2
    exit 1
  fi
fi
install -d -m 755 /etc/grossomodo /etc/vulkan/loader_settings.d
install -m 755 libVkLayer_GROSSOMODO_upscale.so /etc/grossomodo/
install -m 644 VkLayer_GROSSOMODO_upscale.json /etc/grossomodo/
[ -e /etc/grossomodo/grossomodo.conf ] || install -m 644 grossomodo.conf /etc/grossomodo/
install -m 755 uninstall.sh /etc/grossomodo/
install -m 644 vk_loader_settings.json "$SETTINGS"
echo "installé. Actif au prochain démarrage de gamescope. Journal : /tmp/grossomodo-$(id -u "${SUDO_USER:-deck}" 2>/dev/null || echo 1000).log"
echo "réglages : /etc/grossomodo/grossomodo.conf (relu toutes les 2 s) ; désinstaller : sudo /etc/grossomodo/uninstall.sh"
