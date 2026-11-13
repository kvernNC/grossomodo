#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exécute un réseau FSRCNNX à partir de son fichier de shaders mpv (.glsl), et compte ce qu'il coûte.

Chaque passe d'un fichier FSRCNNX a la forme :

    vec4 res = vec4(biais);
    res += vec4(w) * float(LUMA_texOff(vec2(dx, dy)));        // 1re couche : luminance -> 4 canaux
    res += mat4(16 poids) * NOM_texOff(vec2(dx, dy));         // couches suivantes : 4 canaux -> 4 canaux
    res += NOM_texOff(0);                                      // connexion résiduelle (facultatif)
    res = max(res, vec4(0.0)) + vec4(a) * min(res, vec4(0.0)); // PReLU (facultatif)

et la dernière (« aggregation ») range les 4 canaux de SUBCONV1 en un bloc 2x2 de sortie. Le script lit ces termes,
les applique avec numpy (bord : répétition du pixel, comme un échantillonneur CLAMP_TO_EDGE), et compte pour chaque
passe les multiplications-additions et les lectures de texture par pixel basse résolution.
"""

import re

import numpy as np

NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _source(rest):
    """`NOM_texOff(vec2(dx,dy))` ou `NOM_texOff(0)` -> (NOM, dx, dy)."""
    m = re.search(r"(\w+)_texOff\(vec2\((" + NUM + r"),(" + NUM + r")\)\)", rest)
    if m:
        return m.group(1), int(float(m.group(2))), int(float(m.group(3)))
    return re.search(r"(\w+)_texOff\(0\)", rest).group(1), 0, 0


def _floats(s):
    return np.array([float(x) for x in re.findall(NUM, s)], dtype=np.float64)


class Pass:
    def __init__(self, desc, binds, save):
        self.desc, self.binds, self.save = desc, binds, save
        self.bias = None
        self.terms = []        # (source, dx, dy, poids : (4,) pour LUMA, (4, 4) colonnes pour mat4)
        self.residual = None   # nom de la texture ajoutée telle quelle
        self.prelu = None      # pente négative (4,)
        self.aggregate = False

    @property
    def macs(self):
        return sum(4 if w.shape == (4,) else 16 for _, _, _, w in self.terms) + (4 if self.residual else 0)

    @property
    def fetches(self):
        return len(self.terms) + (1 if self.residual else 0)


def parse(path):
    passes = []
    text = open(path, encoding="utf8").read()
    for block in text.split("//!HOOK")[1:]:
        meta = dict(desc="", binds=[], save=None)
        for line in block.splitlines():
            if line.startswith("//!DESC"):
                meta["desc"] = line[7:].strip()
            elif line.startswith("//!BIND"):
                meta["binds"].append(line[7:].strip())
            elif line.startswith("//!SAVE"):
                meta["save"] = line[7:].strip()
        p = Pass(**meta)
        body = block[block.index("{") + 1:block.rindex("}")]
        if "fcoord" in body:
            p.aggregate = True
            passes.append(p)
            continue
        for stmt in body.split(";"):
            stmt = stmt.strip()
            if stmt.startswith("vec4 res = vec4("):
                p.bias = _floats(stmt[len("vec4 res = vec4("):])
            elif stmt.startswith("res += vec4(") and "texOff" in stmt:
                w, rest = stmt[len("res += vec4("):].split(")", 1)
                p.terms.append((*_source(rest), _floats(w)))
            elif re.match(r"res \+= mat4(x4)?\(", stmt):
                w, rest = re.split(r"\)\s*\*", stmt[stmt.index("(") + 1:], maxsplit=1)
                cols = _floats(w).reshape(4, 4)  # mat4(...) est rempli colonne par colonne
                p.terms.append((*_source(rest), cols))
            elif re.match(r"res \+= \w+_texOff\(0\)", stmt):
                p.residual = re.match(r"res \+= (\w+)_texOff", stmt).group(1)
            elif stmt.startswith("res = max(res"):
                p.prelu = _floats(re.search(r"\+\s*vec4\(([^)]*)\)", stmt).group(1))
        passes.append(p)
    return passes


def _shift(img, dx, dy):
    """Lecture décalée de (dx, dy) pixels avec bords répétés : out[y, x] = img[y + dy, x + dx]."""
    h, w = img.shape[:2]
    ys = np.clip(np.arange(h) + dy, 0, h - 1)
    xs = np.clip(np.arange(w) + dx, 0, w - 1)
    return img[ys][:, xs]


def run(passes, luma):
    """luma : tableau (h, w) dans [0, 1]. Renvoie la luminance agrandie x2, (2h, 2w)."""
    tex = {"LUMA": luma[..., None]}
    h, w = luma.shape
    for p in passes:
        if p.aggregate:
            sub = tex[p.binds[0]]
            out = np.empty((2 * h, 2 * w))
            # aggregation : index = (x pair/impair, y pair/impair), composante index.x * 2 + index.y
            for ix in range(2):
                for iy in range(2):
                    out[iy::2, ix::2] = sub[..., ix * 2 + iy]
            return out
        res = np.broadcast_to(p.bias, (h, w, 4)).copy()
        for src, dx, dy, wgt in p.terms:
            s = _shift(tex[src], dx, dy)
            if wgt.shape == (4,):
                res += s[..., 0:1] * wgt
            else:
                res += s @ wgt  # res_j += sum_i v_i * colonne_i[j]
        if p.residual:
            res += tex[p.residual]
        if p.prelu is not None:
            res = np.maximum(res, 0) + p.prelu * np.minimum(res, 0)
        tex[p.save] = res
    raise ValueError("pas de passe d'agrégation")


def cost(passes):
    """(multiplications-additions, lectures, octets écrits) par pixel basse résolution, hors agrégation."""
    macs = sum(p.macs for p in passes if not p.aggregate)
    fetches = sum(p.fetches for p in passes if not p.aggregate)
    writes = 16 * sum(1 for p in passes if not p.aggregate)  # une texture RGBA16F ou 32F par passe : 8 ou 16 o
    return macs, fetches, writes
