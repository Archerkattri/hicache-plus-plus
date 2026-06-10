#!/usr/bin/env python3
"""Run all HiCache++ CPU self-tests (no GPU / no model). Each module exits 0/1."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = ["hicache_pp.hermite", "hicache_pp.dmd", "hicache_pp.tree"]
SCRIPTS = ["tests/test_bench_dit.py", "tests/test_bench_checkpoint.py",
           "tests/test_version.py"]


def main() -> int:
    ok = True
    for m in MODULES:
        print(f"\n================ {m} ================", flush=True)
        rc = subprocess.run([sys.executable, "-m", m], cwd=str(ROOT)).returncode
        ok = ok and rc == 0
    for s in SCRIPTS:
        print(f"\n================ {s} ================", flush=True)
        rc = subprocess.run([sys.executable, str(ROOT / s)], cwd=str(ROOT)).returncode
        ok = ok and rc == 0
    print("\n" + ("ALL MODULES PASSED" if ok else "SOME MODULES FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
