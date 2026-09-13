"""
record_hardware.py  --  Reviewer response, Issue #14 (hardware spec for timings)
================================================================================
Captures the machine + software environment so the timing benchmarks in Section
5.4 (DPO solve / closed-loop run times) are reproducible. Writes a text file and
prints a paste-ready sentence for the manuscript.

This is the only script you must run ON THE SAME MACHINE that produced the
reported timings.

Results written to results/:
    * hardware_spec.txt

Run:
    python scripts/record_hardware.py
"""

import os
import platform
import sys


def _ram_gib():
    # Try psutil first (cross-platform), then Linux /proc, else unknown.
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = float(line.split()[1])
                        return kb / (1024 ** 2)
    except Exception:
        pass
    try:  # macOS
        out = os.popen("sysctl -n hw.memsize").read().strip()
        if out:
            return float(out) / (1024 ** 3)
    except Exception:
        pass
    return None


def _versions():
    out = {}
    for mod in ("numpy", "scipy", "matplotlib", "sklearn"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = "not installed"
    return out


def main():
    import _common as cm
    from src import config as C
    cm.banner(0, "Issue #14: record hardware / environment for timing benchmarks")

    cpu = platform.processor() or platform.machine() or "unknown"
    ram = _ram_gib()
    ram_str = f"{ram:.1f} GiB" if ram else "unknown"
    vers = _versions()

    lines = [
        f"CPU                : {cpu}",
        f"machine/arch       : {platform.machine()}",
        f"logical CPUs       : {os.cpu_count()}",
        f"RAM (total)        : {ram_str}",
        f"OS / platform      : {platform.platform()}",
        f"Python             : {sys.version.split()[0]} ({platform.python_implementation()})",
        f"numpy              : {vers['numpy']}",
        f"scipy              : {vers['scipy']}",
        f"matplotlib         : {vers['matplotlib']}",
        f"scikit-learn       : {vers['sklearn']}",
    ]
    print("\n".join("   " + l for l in lines))

    out = os.path.join(C.RESULTS_DIR, "hardware_spec.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n   wrote {out}")

    sentence = (
        f"All timing benchmarks were measured single-threaded on {cpu} "
        f"({os.cpu_count()} logical cores, {ram_str} RAM) running "
        f"Python {sys.version.split()[0]} with NumPy {vers['numpy']} and "
        f"SciPy {vers['scipy']}."
    )
    print("\n   PASTE INTO SECTION 5.4:\n   " + sentence)


if __name__ == "__main__":
    main()
