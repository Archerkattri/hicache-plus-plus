#!/usr/bin/env python3
"""Run the retained CPU benchmark, adapter, and package integration checks."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    "tests/test_bench_checkpoint.py",
    "tests/test_bench_dit.py",
    "tests/test_h1_adapter_contract.py",
]


def main() -> int:
    ok = True
    for s in SCRIPTS:
        print(f"\n================ {s} ================", flush=True)
        rc = subprocess.run([sys.executable, str(ROOT / s)], cwd=str(ROOT)).returncode
        ok = ok and rc == 0
    print("\n" + ("ALL MODULES PASSED" if ok else "SOME MODULES FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
