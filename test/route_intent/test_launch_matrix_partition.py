"""Test for launch_matrix.py --partition K/N stride slicing.

Locks in the contract that ``--seeds 0 1 2 --partition K/3`` produces a
per-seed split (each partition contains all cells for exactly one seed).
This is the guarantee that lets the user split a large matrix across machines
to three machines with one partition per machine.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "scripts" / "experiments" / "launch_matrix.py"


def _dry_run(*args: str) -> str:
    """Run the launcher with --dry-run + extra args, return stdout text."""
    cmd = [
        sys.executable, str(LAUNCHER),
        "--conditions", "speed_only_pcmappo", "rule_path_pcmappo",
                        "learn_path_pcmappo",
        "--regimes", "aligned", "conflicting",
        "--seeds", "0", "1", "2",
        "--dry-run",
        "--carla-restart-mode", "none",  # no real CARLA touch in dry-run
        *args,
    ]
    result = subprocess.run(
        cmd, cwd=str(REPO), capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"launcher dry-run failed (rc={result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _slugs_in_output(out: str) -> list[str]:
    """Extract slugs the launcher reports it would launch."""
    slugs = []
    for line in out.splitlines():
        if "[matrix] launching " in line:
            # Format: "[matrix] launching <prefix>custom_<cond>_<reg>_seed<N> (custom) -> ..."
            chunk = line.split("launching ", 1)[1].split(" ", 1)[0]
            slugs.append(chunk)
    return slugs


def _seed_from_slug(slug: str) -> int:
    """Extract integer seed from a slug like 'matrix_custom_..._seed1'."""
    tail = slug.rsplit("_seed", 1)[-1]
    return int(tail)


def test_partition_1_of_3_is_seed_0_only():
    out = _dry_run("--partition", "1/3")
    slugs = _slugs_in_output(out)
    assert slugs, f"no slugs in dry-run output:\n{out}"
    seeds = {_seed_from_slug(s) for s in slugs}
    assert seeds == {0}, f"--partition 1/3 should be all seed=0, got seeds={seeds}"
    # 3 conditions x 2 regimes x 1 seed = 6 cells
    assert len(slugs) == 6, f"expected 6 cells, got {len(slugs)}: {slugs}"


def test_partition_2_of_3_is_seed_1_only():
    out = _dry_run("--partition", "2/3")
    slugs = _slugs_in_output(out)
    seeds = {_seed_from_slug(s) for s in slugs}
    assert seeds == {1}, f"--partition 2/3 should be all seed=1, got seeds={seeds}"
    assert len(slugs) == 6


def test_partition_3_of_3_is_seed_2_only():
    out = _dry_run("--partition", "3/3")
    slugs = _slugs_in_output(out)
    seeds = {_seed_from_slug(s) for s in slugs}
    assert seeds == {2}, f"--partition 3/3 should be all seed=2, got seeds={seeds}"
    assert len(slugs) == 6


def test_partition_union_equals_full_matrix():
    """Sanity: partitions 1/3, 2/3, 3/3 union back to the full cell list."""
    union_slugs: set[str] = set()
    for k in (1, 2, 3):
        out = _dry_run("--partition", f"{k}/3")
        union_slugs.update(_slugs_in_output(out))
    # 3 cond x 2 regimes x 3 seeds = 18 cells in the full matrix
    assert len(union_slugs) == 18, (
        f"three partitions should union to 18 cells, got {len(union_slugs)}"
    )


def test_partition_rejects_bad_format():
    """--partition with invalid format should fail with a clear SystemExit."""
    proc = subprocess.run(
        [
            sys.executable, str(LAUNCHER),
            "--conditions", "speed_only_pcmappo",
            "--regimes", "aligned",
            "--seeds", "0",
            "--dry-run",
            "--carla-restart-mode", "none",
            "--partition", "not-a-fraction",
        ],
        cwd=str(REPO), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0, "bad --partition should be rejected"
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "--partition" in combined, (
        f"error message should mention --partition; got:\n{combined}"
    )


def test_partition_rejects_out_of_range():
    """--partition 4/3 should fail (K > N)."""
    proc = subprocess.run(
        [
            sys.executable, str(LAUNCHER),
            "--conditions", "speed_only_pcmappo",
            "--regimes", "aligned",
            "--seeds", "0",
            "--dry-run",
            "--carla-restart-mode", "none",
            "--partition", "4/3",
        ],
        cwd=str(REPO), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0, "out-of-range --partition should be rejected"
