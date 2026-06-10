#!/usr/bin/env python3
"""hicache_pp.__version__ must equal pyproject.toml's version.

Guards the literal-vs-metadata drift found in the 1.2.0 release prep
(``__version__`` said 1.1.0 while pyproject said 1.2.0.dev0). The package keeps a
synced literal instead of importlib.metadata because the benchmark harnesses import
the SOURCE TREE via sys.path while an older wheel may be installed in the same env;
metadata lookup would report the wheel's version for the source tree's code.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_version_synced_with_pyproject():
    sys.path.insert(0, str(ROOT))
    import hicache_pp

    # make sure we are testing the source tree, not an installed copy
    assert Path(hicache_pp.__file__).resolve().parent.parent == ROOT, hicache_pp.__file__
    m = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)
    assert m, "no version field in pyproject.toml"
    assert hicache_pp.__version__ == m.group(1), (
        f"__init__.__version__ {hicache_pp.__version__!r} != pyproject {m.group(1)!r}"
    )


if __name__ == "__main__":
    test_version_synced_with_pyproject()
    print("version sync OK")
