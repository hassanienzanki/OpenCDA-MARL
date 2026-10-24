"""The traffic-manager inspector must be able to load its spectator module.

``scripts/inspect_traffic_manager.py`` loads ``CarlaSpectator`` by file path at
import time (``pixi run traffic-inspect`` and the two review tasks), so a
module that moved leaves every task failing at import, before any CARLA
connection is attempted. The loader is exercised here under a stub for the
``carla`` package: nothing connects to a simulator.
"""
import ast
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "inspect_traffic_manager.py"

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="not a source checkout")


def _repo_root_paths(tree):
    """Every ``REPO_ROOT / "a" / "b"`` expression in the script, as a Path."""
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
            continue
        parts, cur = [], node
        while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
            if not (isinstance(cur.right, ast.Constant) and isinstance(cur.right.value, str)):
                parts = None
                break
            parts.append(cur.right.value)
            cur = cur.left
        if parts and isinstance(cur, ast.Name) and cur.id == "REPO_ROOT":
            found.append(REPO.joinpath(*reversed(parts)))
    return found


def test_repo_relative_paths_in_the_script_exist():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    paths = _repo_root_paths(tree)
    assert paths, "the script no longer builds any REPO_ROOT-relative path"
    missing = [str(p.relative_to(REPO)) for p in paths if not p.exists()]
    assert not missing, f"paths the script loads do not exist: {missing}"


def test_spectator_loader_returns_the_class():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    loader = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_carla_spectator_class"
    )
    namespace = {"REPO_ROOT": REPO, "importlib": importlib}
    exec(compile(ast.Module(body=[loader], type_ignores=[]), str(SCRIPT), "exec"), namespace)

    with mock.patch.dict(sys.modules, {"carla": mock.MagicMock()}):
        cls = namespace["_load_carla_spectator_class"]()

    assert cls.__name__ == "CarlaSpectator"
