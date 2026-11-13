# Journal de développement de GrossoModo (ex-« gamescope-scaler », PoC)

Idée : remplacer la passe **EASU** (l'agrandissement de FSR 1) de gamescope par un autre agrandisseur spatial, via
une couche Vulkan chargée **dans gamescope** (pas dans le jeu), pour l'utiliser en ligne depuis le menu rapide du Deck
(filtre FSR ; le RCAS de Valve et le curseur « FSR Sharpness » restent). Premier candidat : le **MAKO Scaler**
(GPL-3.0-or-later, https://github.com/eugeniosegala/MAKO), sans rien de propriétaire.

## PoC-A : le MAKO Scaler peut-il prendre la place d'EASU, et fait-il mieux ?

- `shaders/cs_mako_easu.comp` : le calcul du MAKO Scaler (sans sa netteté) avec **exactement l'interface d'EASU**
  (liaisons 0/1/3, constantes `FsrEasuCon` dont on déduit les tailles, groupes de 64 invocations pour 16x16 pixels,
  sortie scRGB -> PQ comme EASU).
- `third_party/gamescope/` : en-têtes de shaders de gamescope (BSD-2) et `cs_easu*.comp`, commit 865a71c.
- `harness/harness.cpp` : exécute un des shaders avec la même disposition que gamescope (Vulkan, ici llvmpipe).
- `scripts/poc_a.py` : images de test, agrandissement x1,5 (rendu à 67 %), PSNR / SSIM contre l'original, planches.

```sh
for s in third_party/gamescope/cs_easu.comp third_party/gamescope/cs_easu_fp16.comp shaders/cs_mako_easu.comp; do
  glslangValidator -V --target-env vulkan1.3 -Ithird_party/gamescope "$s" -o build/$(basename $s .comp).spv; done
g++ -O2 -std=c++17 harness/harness.cpp -lvulkan -o build/harness
python3 -I scripts/poc_a.py
```

### Résultat (12/11/2026, x1,5, avant RCAS)

| image | bilinéaire | Lanczos (PIL) | EASU | EASU fp16 | MAKO |
|---|---|---|---|---|---|
| texte HUD | 28,00 / 0,9755 | 28,68 / 0,9795 | 28,62 / 0,9801 | 28,61 / 0,9801 | 28,70 / 0,9798 |
| diagonales, cercles | 14,73 / 0,3207 | 14,61 / 0,3776 | 14,80 / 0,3657 | 14,79 / 0,3652 | 14,70 / 0,3539 |
| mire de zone | 21,49 / 0,9597 | 23,37 / 0,9765 | 22,76 / 0,9723 | 22,76 / 0,9723 | 23,12 / 0,9744 |
| texture type feuillage | 24,42 / 0,6792 | 24,23 / 0,6955 | 24,36 / 0,6931 | 24,36 / 0,6929 | 24,32 / 0,6977 |
| photo (Deck) | 27,39 / 0,8920 | 28,77 / 0,9217 | 28,48 / 0,9141 | 28,48 / 0,9139 | 28,47 / 0,9168 |
| **moyenne** | 23,21 / 0,7654 | 23,93 / 0,7902 | **23,80 / 0,7851** | 23,80 / 0,7849 | **23,86 / 0,7845** |

(PSNR dB / SSIM de luminance, plus haut = mieux.)

- **Interface : validée.** Le shader tourne avec la disposition d'EASU, sans rien changer côté gamescope.
- **Qualité : pas de gain.** MAKO et EASU sont à égalité (écarts de quelques centièmes, dans un sens ou dans l'autre
  selon l'image) ; visuellement, les planches (`build/poc_a/*_planche.png`) sont quasi identiques.
- Limites : images de synthèse et une photo, pas de captures de jeu ; comparaison avant RCAS (la netteté de MAKO,
  pensée pour remplacer RCAS, n'est pas évaluée) ; temps mesurés sur CPU (llvmpipe), sans valeur pour le Deck.

Le marqueur pour reconnaître EASU dans le SPIR-V de gamescope : `OpName "easuPass(vu2;"` (compilé avec
`glslangValidator -V`, noms conservés), présent dans `cs_easu` et `cs_easu_fp16`.

## Autres candidats spatiaux (même banc, même interface EASU)

Ajoutés : **SGSR 1** (Qualcomm, BSD-3, déjà dans gamescope), **Lanczos-3 + anti-ringing** (`shaders/cs_lanczos3_easu.comp`,
36 lectures), **EWA Lanczos « sharp »** (`shaders/cs_ewa_easu.comp`, paramètres `ewa_lanczossharp` de libplacebo/mpv),
et 5 photos libres de scikit-image (`pip install scikit-image`). Moyenne sur 10 images, x1,5, avant RCAS :

| méthode | PSNR moyen (dB) | SSIM moyen |
|---|---|---|
| bilinéaire | 27,14 | 0,8365 |
| **EASU (FSR 1 actuel)** | **27,69** | **0,8521** |
| MAKO Scaler | 27,77 | 0,8529 |
| SGSR 1 | 27,26 | 0,8558 |
| EWA Lanczos sharp | 27,77 | 0,8520 |
| **Lanczos-3 + anti-ringing** | **27,89** | 0,8563 |
| Lanczos (PIL, sans anti-ringing) | 27,86 | 0,8568 |

Le meilleur filtre classique (Lanczos-3 + anti-ringing) gagne **+0,2 dB et +0,004 de SSIM** sur EASU : invisible à
l'oeil sur les planches. SGSR garde mieux la structure du texte et des diagonales (SSIM) mais perd en fidélité sur
les photos et la mire. **Conclusion : les filtres spatiaux classiques plafonnent au niveau d'EASU** ; un vrai gain ne
peut venir que d'un filtre appris (type RAVU / RAISR ou petit réseau convolutif), plus coûteux.

## FSRCNNX (filtre appris, x2)

`scripts/fsrcnnx.py` lit les shaders mpv de FSRCNNX (`third_party/fsrcnnx/`, LGPL-3, igv) et exécute le réseau avec
numpy (luminance seule, comme mpv ; chrominance en Lanczos) ; il compte aussi les multiplications-additions par
pixel. `scripts/poc_fsrcnnx.py` compare tout en x2 et en x1,5 (FSRCNNX : x2 puis réduction Lanczos). 10 images,
moyennes, avant RCAS :

| méthode | x2 PSNR / SSIM | x1,5 PSNR / SSIM |
|---|---|---|
| bilinéaire | 26,22 / 0,8082 | 27,15 / 0,8364 |
| **EASU (FSR 1)** | **27,65 / 0,8322** | **27,70 / 0,8521** |
| Lanczos-3 + anti-ringing | 27,93 / 0,8454 | 27,90 / 0,8563 |
| Lanczos (PIL) | 28,22 / 0,8486 | 27,86 / 0,8567 |
| SGSR 1 | 27,59 / 0,8441 | 27,27 / 0,8557 |
| MAKO Scaler | 27,49 / 0,8382 | 27,78 / 0,8528 |
| **FSRCNNX 8-0-4-1** | **28,66 / 0,8732** | 27,51 / 0,8514 |
| **FSRCNNX 16-0-4-1** | **28,91 / 0,8814** | 27,45 / 0,8475 |

- **En x2, FSRCNNX gagne nettement** : +1,0 à +1,3 dB et +0,04 à +0,05 de SSIM sur EASU, et c'est visible sur les
  planches (`build/poc_fsrcnnx/x2_*_planche.png`) : contours et texte plus nets.
- **En x1,5, il ne gagne rien** : le réseau est entraîné pour x2, et la réduction qui suit efface l'avantage.

Coût théorique sur Steam Deck (0,82 TMAC/s FP32, 1,64 FP16, 88 Go/s ; efficacité réelle supposée 30-50 %) :

| réseau | mult.-add. / pixel BR | x2 640x400 -> 1280x800 |
|---|---|---|
| FSRCNNX 8-0-4-1 (13 passes) | 2 864 | 0,73 G : **~1 à 1,5 ms en FP16** (1,8-3 ms en FP32), trafic ~80 Mo (>= 1,3 ms) |
| FSRCNNX 16-0-4-1 (25 passes) | 10 464 | 2,68 G : 3,3-5,4 ms en FP16 (6,5-11 ms en FP32) |
| EASU (référence) | ~150 op. / pixel de sortie | ~0,2 ms |

Seul **FSRCNNX 8-0-4-1 en x2** paraît compatible avec le budget du Deck (rendu à 50 %, 640x400). Il ne tient pas
dans une passe : la couche devrait enregistrer ses 14 passes (images intermédiaires 4 canaux) à la place d'EASU.

## Les facteurs réels d'Elden Ring

Le jeu ne propose que du 16:9 (wiki ELDEN RING, page System) : 800x450, 1024x576, 1152x648, 1280x720, 1440x810,
1600x900, ... Ni 16:10, ni 480p, ni 640x360. Sur le Deck, l'image du jeu fait donc 1280x720 (bandes noires), et les
facteurs possibles vers 720p sont x1,11 (1152x648), **x1,25 (1024x576)** et **x1,6 (800x450)**. FSRCNNX x2 n'a pas de
résolution source dans ce jeu. Même banc (`poc_a.py --factor`), 10 images, moyennes PSNR / SSIM, avant RCAS :

| méthode | x1,25 (1024x576) | x1,6 (800x450) |
|---|---|---|
| bilinéaire | 28,78 / 0,8923 | 27,48 / 0,8521 |
| **EASU (FSR 1)** | **29,31 / 0,9035** | **28,47 / 0,8683** |
| MAKO Scaler | 29,33 / 0,9063 | 28,60 / 0,8741 |
| SGSR 1 | 28,49 / 0,9002 | 28,26 / 0,8763 |
| EWA Lanczos sharp | 29,34 / 0,8994 | 28,55 / 0,8642 |
| **Lanczos-3 + anti-ringing** | **29,61 / 0,9111** | **28,87 / 0,8792** |
| Lanczos (PIL) | 29,56 / 0,9106 | 28,88 / 0,8794 |

(Les moyennes d'un facteur à l'autre ne sont pas comparables entre elles : la mire de diagonales crée un moiré qui
dépend du facteur. Seul le classement à facteur égal compte.) À x1,6, Lanczos-3 + anti-ringing gagne +0,4 dB et
+0,011 de SSIM sur EASU : l'écart le plus grand des filtres classiques, mais encore modeste.

## Sur de vraies images d'Elden Ring (`scripts/poc_er.py`)

5 captures 4K du jeu (réglages max ; 4 de la galerie PCMRace du fil Steam « My Screenshot Gallery & Videos @ 4K »,
1 d'un album imgur ; **non versionnées**, dans `data/er_4k/`). Référence : 1280x720 (moyenne de zones) ; entrées
aux résolutions du jeu. Avant RCAS, écart de PSNR avec EASU :

| méthode | 1152x648 (x1,11) | 1024x576 (x1,25) | 800x450 (x1,6) |
|---|---|---|---|
| EASU (FSR 1) | 39,01 dB | 37,57 dB | 35,54 dB |
| Lanczos-3 + anti-ringing | **+0,68** | **+0,54** | +0,40 |
| FSRCNNX 8 (x2 puis réduction) | +0,43 | +0,39 | **+0,45** |
| FSRCNNX 16 (x2 puis réduction) | +0,51 | +0,46 | **+0,51** |
| MAKO Scaler | +0,45 | +0,35 | +0,24 |
| EWA Lanczos sharp | +0,41 | +0,29 | +0,12 |
| SGSR 1 | -0,57 | -0,28 | +0,05 |
| bilinéaire | -0,78 | -0,72 | -0,62 |

- Sur ce jeu, **EASU est le plus faible des filtres sérieux** : tous ses concurrents directs le battent de 0,1 à
  0,7 dB.
- **Lanczos-3 + anti-ringing** est le meilleur rapport qualité / coût : premier à x1,11 et x1,25, à 0,05-0,1 dB de
  FSRCNNX à x1,6, pour une passe de 36 lectures (FSRCNNX 8 : 14 passes, ~10 fois plus de calcul).
- Visuellement (`build/poc_er/*_800x450_planche.png`), FSRCNNX et Lanczos-3 sont un peu plus nets qu'EASU sur
  l'architecture et le feuillage ; l'écart est réel mais discret.
- Limites : 5 images en réglages max (pas les réglages du Deck), réduction idéale (un vrai rendu basse résolution
  est plus crénelé), mesure avant RCAS (le RCAS de Valve s'ajoute après tous les filtres dans gamescope).

### Le détail fin (`scripts/poc_er_detail.py`)

Le PSNR favorise une image fidèle en moyenne, donc un peu douce. Mesure du détail fin (luminance passe-haut :
image - flou gaussien sigma 1) : énergie rapportée à la référence, et PSNR de cette seule bande.

| méthode | 1024x576 énergie / PSNR détail | 800x450 énergie / PSNR détail |
|---|---|---|
| bilinéaire | 60,7 % / 38,24 dB | 47,4 % / 36,64 dB |
| EASU (FSR 1) | 74,3 % / 38,83 dB | 59,2 % / 37,02 dB |
| Lanczos-3 + anti-ringing | 77,9 % / 39,30 dB | 61,5 % / 37,30 dB |
| **FSRCNNX 8** | **97,2 % / 39,28 dB** | **86,4 % / 37,48 dB** |

FSRCNNX 8 garde **86 % du détail fin à 800x450** contre 59 % pour EASU et 62 % pour Lanczos-3, **et** ce détail est
au bon endroit (meilleur PSNR de la bande). C'est ce que l'oeil voit sur les planches : il reconstruit du détail
plausible au lieu de seulement interpoler. Reste à comparer après RCAS, qui ajoute aussi du détail apparent à EASU.

## Images finales, avec RCAS (`shaders/cs_rcas.comp`, `scripts/poc_er_rcas.py`)

RCAS comme dans `cs_composite_rcas.comp` de gamescope (même `FsrRcasF`, constante `FsrRcasCon(nUpscaleSharpness / 10)`),
appliqué après chaque agrandisseur. Netteté gamescope : 0 = maximum, 20 = minimum, **2 par défaut** (`main.cpp`).
5 captures d'Elden Ring, extraits (détail = énergie du détail fin rapportée à la référence) :

| 800x450 -> 720p | netteté | PSNR | SSIM | détail | PSNR détail |
|---|---|---|---|---|---|
| EASU | 5 | 35,13 | 0,9355 | 85,8 % | 36,57 |
| EASU (**FSR actuel par défaut**) | 2 | 34,12 | 0,9226 | 102,5 % | 35,68 |
| Lanczos-3 + AR | 5 | 35,50 | 0,9407 | 86,8 % | 37,06 |
| Lanczos-3 + AR | 2 | 34,40 | 0,9278 | 102,7 % | 36,25 |
| **FSRCNNX 8** | sans RCAS | **35,99** | **0,9468** | 86,4 % | **37,48** |
| **FSRCNNX 8** | 10 | **34,95** | 0,9385 | 106,4 % | **36,50** |
| FSRCNNX 8 | 2 | 32,16 | 0,8993 | 143,4 % (sur-accentué) | 33,89 |

À **netteté perçue égale** (même énergie de détail), FSRCNNX 8 bat le FSR actuel de **+0,8 à +0,9 dB** à 800x450
(FSRCNNX sans RCAS contre EASU à 5 ; FSRCNNX à 10 contre EASU à 2, le défaut) et de +0,7 à +1,2 dB à 1024x576.
L'écart est **plus grand qu'avant RCAS** : RCAS accentue aussi les erreurs d'EASU, alors que FSRCNNX a besoin de peu
de netteté. Conséquence pratique : avec FSRCNNX à la place d'EASU, il faut **baisser la netteté** (10 à 20, ou
RCAS coupé), sinon l'image est sur-accentuée (143 % au réglage par défaut).

## FSRCNNX 8 en passes Vulkan (`scripts/gen_fsrcnnx_vk.py`, `harness/fsrcnnx_vk.cpp`)

- **Générateur** : lit le fichier mpv et écrit les shaders de calcul GLSL, **poids écrits en dur** (comme mpv, pas de
  buffer de poids comme d2-webgpu). Les 14 passes mpv deviennent **7 dispatches** (les deux moitiés de 4 canaux de
  chaque couche ensemble ; convolution sous-pixel + agrégation en un seul, qui écrit le bloc 2x2 et recombine la
  couleur : rgb bilinéaire + (Y réseau - Y bilinéaire), comme d2-webgpu), plus **1 dispatch de réduction** Lanczos-2
  vers la taille finale (x1,6 : 800x450 -> 1600x900 -> 1280x720). Cartes 8 canaux en image2DArray RGBA16F à 2
  couches, allocation FEAT / A / B selon la durée de vie des cartes. Variantes **FP32 et FP16** (`-DUSE_FP16`).
- **Banc multi-passes** : exécute le plan, mesure chaque passe avec des **timestamps GPU** (médiane sur N répétitions).
- **Validation** (llvmpipe, capture d'Elden Ring 800x450) : luminance x2 identique à la référence numpy à 0,25/255
  près en moyenne (stockage 16 bits, arrondi 8 bits) ; image finale 1280x720 : même PSNR (34,27 dB) que la version
  numpy, en FP32 comme en FP16.
- Le banc mono-passe (`harness.cpp`) mesure lui aussi le temps GPU par timestamps (EASU, Lanczos-3, RCAS).

### Mesurer sur le Steam Deck

`scripts/package_deck.sh` construit `build/deck-bench.tar.gz` (programmes Linux, libstdc++ liée en statique, glibc
>= 2.38 et libvulkan du système ; shaders ; images d'entrée synthétiques). Sur le Deck, jeu fermé :

```sh
tar xzf deck-bench.tar.gz && ./deck-bench/run.sh
```

`deck-bench/rapport.txt` donne, pour 1152x648, 1024x576 et 800x450 -> 1280x720 : EASU (FP32/FP16), Lanczos-3,
FSRCNNX 8 passe par passe (FP32/FP16), et RCAS à 1280x720.

### Premières mesures sur le Steam Deck (13/11/2026, `results/deck-2026-11-13-rapport1.txt`)

RADV Van Gogh, SteamOS, noyau 7.2.7-valve2. Médianes de 200 répétitions, temps GPU (timestamps) :

| 800x450 -> 1280x720 | FP32 | FP16 |
|---|---|---|
| EASU | 0,685 ms | **0,470 ms** |
| Lanczos-3 + AR | 0,917 ms | - |
| RCAS 1280x720 | 0,161 ms | - |
| FSRCNNX 8, réseau (passes 0-6) | 3,11 ms | **1,94 ms** |
| FSRCNNX 8, réduction Lanczos-2 (passe 7) | 1,78 ms | 1,78 ms |
| FSRCNNX 8, total | 4,89 ms | 3,72 ms |

- Le FP16 divise presque par deux les couches de mapping (0,295 contre 0,563 ms) : c'est la variante à garder.
- Le réseau coûte **1,94 ms** à 800x450 (2,70 ms à 1024x576), dans l'estimation (1,3-2 ms).
- La **réduction** (Lanczos-2, deux sinus par échantillon) coûtait presque autant que le réseau : remplacée le
  13/11 par un noyau cubique de Catmull-Rom séparable (même PSNR, 34,27 dB sur la capture de test ; environ 3 fois
  plus rapide relativement aux autres passes sous llvmpipe). Attendu sur le Deck : ~0,5-0,6 ms, soit **FSRCNNX 8
  ~2,5 ms au total, contre ~0,63 ms pour le FSR actuel (EASU FP16 + RCAS)**. À remesurer.

### Réduction par filtrage matériel (rapport Deck n°3, `results/deck-2026-11-13-rapport3.txt`)

La passe de réduction lisait l'image x2 par `imageLoad` (~25 lectures par pixel) : 1,43 ms même sans sinus. Deux
variantes lisent l'image comme une texture avec le filtrage bilinéaire du GPU. Steam Deck, 800x450 -> 1280x720, FP16,
et qualité moyenne sur les 5 captures d'Elden Ring (avant RCAS) :

| réduction | passe 7 (Deck) | FSRCNNX 8 total (Deck) | PSNR | SSIM | détail fin | PSNR détail |
|---|---|---|---|---|---|---|
| cubique, `imageLoad` | 1,433 ms | 3,35 ms | 35,90 | 0,9472 | 79,8 % | 37,63 |
| **Catmull-Rom, 9 lectures filtrées** | **0,289 ms** | **2,21 ms** | 35,73 | 0,9460 | **83,9 %** | 37,50 |
| bilinéaire, 1 lecture filtrée | 0,126 ms | 2,05 ms | 35,82 | 0,9451 | 74,0 % | 37,63 |

**Catmull-Rom 9 lectures devient la réduction par défaut** : elle garde le plus de détail (84 %, proche des 86 % de
la version numpy) pour 0,16 ms de plus que le bilinéaire. Bilan sur le Deck à 800x450 : **FSRCNNX 8 = 2,21 ms**
(réseau 1,92 + réduction 0,29) contre **0,47 ms pour EASU FP16** (+ 0,155 ms de RCAS dans les deux cas, RCAS étant
fusionné à la composition de gamescope). Surcoût : **~1,7 ms par image**.

## PoC-B : la couche dans gamescope (`layer/`)

`layer/grossomodo_layer.cpp` -> `libVkLayer_GROSSOMODO_upscale.so` (VK_LAYER_GROSSOMODO_upscale), shaders FSRCNNX 8 embarqués
(`layer/fsrcnnx8_spv.h`, généré par `scripts/embed_layer_shaders.py`). Elle s'appuie sur la façon dont gamescope
enregistre chaque dispatch (`CVulkanCmdBuffer::dispatch` : pipeline lié sans cache, descripteur neuf avec la sortie en
liaison 1 et l'image du jeu en liaison 3 case 0, images en GENERAL) :

- reconnaît les modules EASU (`easuPass(` dans le SPIR-V ; `FsrEasuH` = FP16) et leurs pipelines (constante de
  spécialisation 4 = espace de couleur du calque 0) ;
- suit pipeline et descripteur liés par command buffer ; au `vkCmdDispatch` d'un pipeline EASU, enregistre les 8
  passes FSRCNNX (image du jeu -> image de sortie de gamescope) au lieu du dispatch ; gamescope relie tout pour RCAS
  et pose lui-même la barrière sur l'image qu'il croit avoir écrite ;
- laisse passer le FSR 1 d'origine en cas de doute : calque HDR (scRGB, PQ), image inconnue, échec de création,
  `enabled=0` dans `/etc/grossomodo/grossomodo.conf` (relu toutes les 2 s) ;
- n'exporte que ses 3 points d'entrée (libstdc++ statique masquée), ne dépend que de la libc ; journal dans
  `/tmp/grossomodo-<uid>.log`.

**Test sans gamescope** (`harness/mock_gamescope.cpp`) : imitation de la séquence de gamescope (disposition à 7
liaisons, sortie B8G8R8A8, EASU puis RCAS, plusieurs images). Résultats : avec la couche, l'image de sortie est
**identique à l'octet** à celle du banc FSRCNNX ; avec `enabled=0`, **identique à l'octet** au FSR 1 d'origine ; RCAS
relit correctement le résultat.

**Chargement limité à gamescope** : sur le Deck, gamescope a `cap_sys_nice=eip` (exécution sécurisée) et ne charge que
les couches système (vu dans `/proc/<pid>/maps` : gamescope_wsi et device_select, pas lsfg-vk de `~/.local`). La couche
est donc installée dans `/etc`, **hors** des dossiers de couches implicites, et activée par un fichier de réglages du
chargeur (`/etc/vulkan/loader_settings.d/vk_loader_settings.json`) qui ne contient que des réglages pour
`app_keys: ["/usr/bin/gamescope"]` : pour tout autre programme (les jeux), le chargeur n'applique aucun réglage et ne
charge pas la couche. Vérifié ici avec le faux gamescope (couche chargée) et un autre programme (pas chargée, même avec
`GROSSOMODO_FORCE=1`). Le manifeste doit déclarer `disable_environment` (exigé pour un manifeste implicite ; sans lui,
le chargeur 1.3.275 plante dans `vkCreateInstance`).

### Installer sur le Deck (`scripts/package_layer.sh` -> `build/grossomodo-layer.tar.gz`)

1. **Prévoir un moyen de revenir en arrière** : le mode Bureau, ou SSH depuis un PC.
2. En mode Bureau : `tar xzf grossomodo-layer.tar.gz && sudo ./grossomodo-layer/install.sh`
3. **Test sans risque, gamescope imbriqué** : options de lancement d'un jeu dans Steam (mode Bureau)
   `gamescope -w 800 -h 450 -W 1280 -H 720 -F fsr -- %command%`, le jeu réglé en 800x450 ; puis
   `cat /tmp/grossomodo-1000.log` (doit contenir « première image ... par FSRCNNX 8 »). Comparaison en direct :
   `echo enabled=0 | sudo tee /etc/grossomodo/grossomodo.conf` (FSR 1), `enabled=1` (FSRCNNX).
4. Mode Jeu : jeu en 800x450, filtre « Net » (= FSR) dans le menu rapide, **netteté à baisser** (FSRCNNX en a peu
   besoin). **Écran partagé** pour comparer : `compare=1` dans `/etc/grossomodo/grossomodo.conf` (FSR 1 à gauche,
   FSRCNNX à droite, trait blanc au milieu ; vérifié à l'octet près avec le faux gamescope).
5. Désinstaller : `sudo /etc/grossomodo/uninstall.sh` (FSR 1 d'origine au prochain démarrage de gamescope).

### Premiers essais sur le Deck (13/11/2026)

- La couche se charge dans le vrai gamescope (mode Jeu), reconnaît EASU FP16 et le remplace : « 1280x720 -> 1422x800
  par FSRCNNX 8 » (mode « Remplir », x1,11 : cas gaspilleur), puis Arkham Knight en 640x480.
- Arkham Knight avec DittoSR dans le jeu + FSRCNNX dans gamescope : 576p « encore super beau », verrouillé à 90 i/s ;
  240p inacceptable, 480p correct. Textes du HUD jugés laids en 640x480 (netteté RCAS et source pauvre suspectées).
- Différence FSR 1 / FSRCNNX difficile à voir en basculant : d'où le mode écran partagé.

### Intensité : FSR 1 + strength x (FSRCNNX - FSR 1) (`scripts/poc_strength.py`)

Sur le Deck, Arkham Knight (jeu très crénelé) : FSRCNNX seul donne **plus de détail mais plus de scintillement et de
crénelage** que FSR 1, plus flou. Réglage `strength` (0 = FSR 1, 1 = FSRCNNX) : la dernière passe calcule aussi EASU
(même code FidelityFX, FP32) sur l'image du jeu et mélange (vérifié : strength=1 identique à l'octet à FSRCNNX ;
strength=0 égal à l'EASU FP16 de gamescope à 0,07/255 près). Mesure du scintillement : panoramique simulé (décalage
de 1/3 de pixel de sortie par image, 8 images) sur les 5 captures, rendu 750x422 -> 1200x675 (x1,6), avant RCAS ;
erreur temporelle = écart moyen entre la variation d'une image à l'autre en sortie et celle de la référence.

| strength | PSNR | détail fin | erreur temporelle |
|---|---|---|---|
| 0 (FSR 1) | 35,33 dB | 59,2 % | 1,463 |
| 0,25 | 35,70 dB | 63,8 % | **1,429** |
| **0,5** | **35,88 dB** | 69,8 % | **1,429** |
| 0,75 | 35,83 dB | 76,7 % | 1,448 |
| 1 (FSRCNNX) | 35,58 dB | 84,0 % | 1,486 |

**0,5 devient la valeur par défaut** : meilleur PSNR, moins de scintillement que FSR 1 et que FSRCNNX seuls, et +10
points de détail par rapport à FSR 1. Limite : les rendus de test sont réduits par moyenne (antialiasés) ; sur un jeu
plus crénelé comme Arkham Knight, l'écart de scintillement de FSRCNNX seul est sûrement plus grand.
