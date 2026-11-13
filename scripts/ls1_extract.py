#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extrait les shaders LS1 (ressources RT_RCDATA en DXBC) du Lossless.dll de l'utilisateur, là où Steam l'installe,
et les traduit en SPIR-V avec libvkd3d-shader (ls1_translate). Rien n'est copié ailleurs que dans le dossier de sortie.

    python3 -I ls1_extract.py [--dll CHEMIN] --out DOSSIER --translate ./ls1_translate
"""
import argparse
import os
import struct
import subprocess

DEFAULT_DLL = os.path.expanduser("~/.steam/steam/steamapps/common/Lossless Scaling/Lossless.dll")
# (ressource, SRV, UAV, CBV, sampler), numérotation de MAKO
SHADERS = [(146, 2, 1, 1, 1)] + [(141 + v, 1, 1, 1, 0) for v in range(5)] + \
    [(147 + 3 * v + k, 1, 1, 0 if k == 1 else 1, 0) for v in range(5) for k in range(3)]


def rcdata(path):
    d = open(path, "rb").read()
    pe = struct.unpack_from("<I", d, 0x3C)[0]
    nsec = struct.unpack_from("<H", d, pe + 6)[0]
    opt = pe + 24
    rva_rsrc = struct.unpack_from("<I", d, opt + 112 + 2 * 8)[0]  # PE32+ : répertoire de données n°2
    secs = [struct.unpack_from("<8sIIII", d, opt + struct.unpack_from("<H", d, pe + 20)[0] + 40 * i) for i in range(nsec)]

    def off(rva):
        for _, vsize, va, rsize, raw in secs:
            if va <= rva < va + max(vsize, rsize):
                return rva - va + raw
        raise ValueError("RVA hors sections")

    base = off(rva_rsrc)

    def entries(o):
        n = sum(struct.unpack_from("<HH", d, base + o + 12))
        return [struct.unpack_from("<II", d, base + o + 16 + 8 * i) for i in range(n)]

    out = {}
    for tid, tgt in entries(0):
        if tid != 10 or not tgt & 0x80000000:
            continue
        for rid, rt in entries(tgt & 0x7FFFFFFF):
            langs = entries(rt & 0x7FFFFFFF)
            lt = next((t for i, t in langs if i == 0), langs[0][1])
            rva, size = struct.unpack_from("<II", d, base + lt)
            out[rid] = d[off(rva):off(rva) + size]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", default=DEFAULT_DLL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--translate", required=True)
    a = ap.parse_args()
    res = rcdata(a.dll)
    os.makedirs(a.out, exist_ok=True)
    for rid, *counts in SHADERS:
        if res.get(rid, b"")[:4] != b"DXBC":
            raise SystemExit(f"ressource {rid} absente ou pas en DXBC : version de Lossless Scaling non prise en charge")
        dx = os.path.join(a.out, f"{rid}.dxbc")
        open(dx, "wb").write(res[rid])
        subprocess.run([a.translate, dx, os.path.join(a.out, f"{rid}.spv"), *map(str, counts)], check=True)
        os.remove(dx)
    print(f"{len(SHADERS)} shaders LS1 traduits dans {a.out}")


if __name__ == "__main__":
    main()
