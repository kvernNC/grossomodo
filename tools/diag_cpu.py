#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Où part le temps processeur d'un jeu sous Proton ? Lecture seule de /proc et /sys, rien n'est injecté dans le jeu.

Pendant que le jeu tourne (en mode Jeu, lancer depuis SSH ou depuis le mode Bureau avec le jeu ouvert) :

    python3 -I diag_cpu.py [nom_du_processus] [secondes]      (défaut : eldenring.exe 30)

Mesure, sur la durée : l'usage de chaque fil d'exécution du jeu (100 % = un cœur entier), nommé (fils du jeu, de
vkd3d-proton, de Wine, de DXVK...) ; l'usage de chaque cœur ; la fréquence réelle du processeur ; la puissance et
l'occupation du GPU (amdgpu). Le résultat s'affiche et est écrit dans diag-<processus>-<date>.txt.
"""
import glob
import os
import sys
import time

HZ = os.sysconf("SC_CLK_TCK")


def find_pid(name):
    best = None
    for d in glob.glob("/proc/[0-9]*"):
        try:
            comm = open(d + "/comm").read().strip()
            cmd = open(d + "/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if name.lower() in comm.lower() or name.lower() in cmd.lower():
            n = len(os.listdir(d + "/task")) if os.path.isdir(d + "/task") else 0
            if not best or n > best[1]:  # le vrai processus du jeu a le plus de fils
                best = (int(d.split("/")[-1]), n, comm)
    return best


def thread_times(pid):
    out = {}
    for t in glob.glob(f"/proc/{pid}/task/*"):
        try:
            stat = open(t + "/stat").read()
            comm = open(t + "/comm").read().strip()
        except OSError:
            continue
        rest = stat[stat.rindex(")") + 2:].split()
        out[int(t.split("/")[-1])] = (comm, int(rest[11]) + int(rest[12]), int(rest[36]))  # utime+stime, dernier cœur
    return out


def cpu_times():
    cores = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            p = line.split()
            v = list(map(int, p[1:]))
            cores[p[0]] = (sum(v), v[3] + v[4])  # total, inactif (idle + iowait)
    return cores


def read(path, conv=float):
    try:
        return conv(open(path).read().strip())
    except (OSError, ValueError):
        return None


def gpu_info():
    for card in glob.glob("/sys/class/drm/card*/device"):
        busy = read(card + "/gpu_busy_percent")
        if busy is None:
            continue
        power = None
        for h in glob.glob(card + "/hwmon/hwmon*"):
            power = read(h + "/power1_average") or read(h + "/power1_input")
            if power:
                power /= 1e6
        return busy, power
    return None, None


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "eldenring.exe"
    secs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    found = find_pid(name)
    if not found:
        sys.exit(f"processus « {name} » introuvable : lancer le jeu d'abord (ou donner son nom, ex. : python3 -I diag_cpu.py sekiro.exe)")
    pid, nthreads, comm = found
    print(f"{comm} (pid {pid}, {nthreads} fils) : mesure pendant {secs} s, jouez normalement...", flush=True)

    t0, c0, w0 = thread_times(pid), cpu_times(), time.time()
    freqs, gbusy, gpow = [], [], []
    peak = {}  # usage maximal par fil sur une seconde
    prev = t0
    for _ in range(secs):
        time.sleep(1)
        cur = thread_times(pid)
        for tid, (cm, ticks, _) in cur.items():
            if tid in prev:
                peak[tid] = max(peak.get(tid, 0.0), 100.0 * (ticks - prev[tid][1]) / HZ)
        prev = cur
        f = [read(p) for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")]
        f = [x for x in f if x]
        if f:
            freqs.append((sum(f) / len(f) / 1e3, max(f) / 1e3))
        b, pw = gpu_info()
        if b is not None:
            gbusy.append(b)
        if pw:
            gpow.append(pw)
    t1, c1, w1 = thread_times(pid), cpu_times(), time.time()
    dt = w1 - w0

    lines = [f"== {time.strftime('%F %T')} {comm} pid {pid}, {len(t1)} fils, {dt:.0f} s",
             f"   noyau {os.uname().release}"]
    rows = []
    for tid, (cm, ticks, core) in t1.items():
        if tid in t0:
            pct = 100.0 * (ticks - t0[tid][1]) / HZ / dt
            rows.append((pct, peak.get(tid, 0.0), tid, cm, core))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    lines.append(f"\nFils du jeu (100 % = un cœur entier) ; total {total:.0f} % sur {len(c1)} cœurs logiques = "
                 f"{total / len(c1):.0f} % de la machine")
    lines.append(f"  {'moyen':>6} {'max/s':>6}  {'tid':>7}  {'dernier cœur':>12}  nom")
    for pct, pk, tid, cm, core in rows[:20]:
        if pct < 0.5:
            break
        lines.append(f"  {pct:5.1f}% {pk:5.1f}%  {tid:7d}  {core:12d}  {cm}")
    groups = {}
    for pct, _, _, cm, _ in rows:
        low = cm.lower()
        key = ("vkd3d-proton" if "vkd3d" in low else "dxvk" if "dxvk" in low else
               "wine" if low.startswith(("wine", "wineserver")) else "autres (jeu, pilote)")
        groups[key] = groups.get(key, 0.0) + pct
    lines.append("  par famille : " + ", ".join(f"{k} {v:.0f} %" for k, v in sorted(groups.items(), key=lambda x: -x[1])))

    lines.append("\nCœurs (usage moyen, tous programmes confondus)")
    for core in sorted(c1, key=lambda c: int(c[3:])):
        tot = c1[core][0] - c0[core][0]
        idle = c1[core][1] - c0[core][1]
        lines.append(f"  {core:>5} {100.0 * (tot - idle) / max(tot, 1):5.1f} %")
    if freqs:
        lines.append(f"\nFréquence CPU : moyenne {sum(a for a, _ in freqs) / len(freqs):.0f} MHz, "
                     f"cœur le plus rapide {sum(b for _, b in freqs) / len(freqs):.0f} MHz en moyenne")
    if gbusy:
        lines.append(f"GPU occupé : {sum(gbusy) / len(gbusy):.0f} % en moyenne")
    if gpow:
        lines.append(f"Puissance de l'APU : {sum(gpow) / len(gpow):.1f} W en moyenne (max {max(gpow):.1f} W)")
    text = "\n".join(lines)
    print(text)
    out = f"diag-{comm.replace('/', '_')}-{time.strftime('%Y%m%d-%H%M%S')}.txt"
    open(out, "w").write(text + "\n")
    print(f"\n-> {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
