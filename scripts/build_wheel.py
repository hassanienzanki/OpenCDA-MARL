"""Build the engine wheel from a clean tree and verify every member has a source.

setuptools copies package data into ``build/lib`` incrementally and never
prunes a file that was deleted from the source tree, so a wheel built from
a long-lived checkout can ship files that no longer exist, and any project
that installed such a wheel could run profiles the framework no longer
defines. ``build/`` is git-ignored, so a fresh clone never shows the
problem and a working checkout always can.

So: remove ``build/`` and the egg-info first, build, then open the wheel
and refuse it if any member has no file behind it in the source tree.

    pixi run wheel            # -> dist/opencda_marl-<version>-py3-none-any.whl
    pixi run wheel some/dir   # -> that directory instead
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    out = Path(argv[1]).resolve() if len(argv) > 1 else REPO / "dist"

    for stale in [REPO / "build", *REPO.glob("*.egg-info")]:
        shutil.rmtree(stale, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("opencda_marl-*.whl"):
        old.unlink()

    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "-w", str(out), str(REPO)],
        check=True, cwd=REPO)

    wheel = max(out.glob("opencda_marl-*.whl"), key=lambda p: p.stat().st_mtime)
    members = [m for m in zipfile.ZipFile(wheel).namelist() if not m.endswith("/")]
    orphans = [m for m in members
               if ".dist-info/" not in m and not (REPO / m).is_file()]
    if orphans:
        print(f"{wheel.name}: {len(orphans)} member(s) have no source file, "
              "so this build tree was stale even after cleaning:", file=sys.stderr)
        for member in orphans[:20]:
            print(f"  {member}", file=sys.stderr)
        wheel.unlink()
        return 1

    size_mb = wheel.stat().st_size / 1048576
    print(f"{wheel}  ({size_mb:.1f} MB, {len(members)} files, every member has a source file)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
