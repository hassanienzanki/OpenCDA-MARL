"""What the wheel must contain, and what it must declare, to be a usable install.

Installing the opencda-marl wheel advertises a runnable RL environment. Two
ways that claim silently broke, both caught after the fact:

- nine engine modules import the vendored OpenCDA core and the wheel shipped
  none of it, so every CARLA scenario failed at import;
- the engine imports third-party packages the wheel never declared, so a
  clean install stopped short of what a run needs -- invisible from a
  checkout, where the development environment has everything anyway.

These tests pin the packaging declaration against the source tree. They
read `pyproject.toml` and walk imports with `ast`, so they build nothing
and run everywhere.
"""
import ast
import re
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, used by the CARLA environment
    import tomli as tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"

pytestmark = pytest.mark.skipif(not PYPROJECT.is_file(), reason="not a source checkout")

STDLIB = set(sys.stdlib_module_names) | {"__future__"}
#: import name -> distribution name, where the two differ. traci and sumolib
#: ship inside eclipse-sumo (under ``sumo/tools``); libsumo does not -- that
#: package carries only the native library, not the Python module.
DISTRIBUTION = {
    "yaml": "pyyaml", "cv2": "opencv-python", "PIL": "pillow",
    "sklearn": "scikit-learn", "skimage": "scikit-image",
    "traci": "eclipse-sumo", "sumolib": "eclipse-sumo",
    "git": "gitpython",
}
#: Imported by the engine but deliberately not declared: version-locked to
#: the simulator you install and not on PyPI for every interpreter. The
#: SUMO path guards its import; a CARLA run needs it on the path.
UNDECLARED_BY_DESIGN = {"carla"}
#: Optional accelerators: imported only inside ``try``/``except ImportError``
#: with a working fallback, so an install without them still runs.
#: libsumo (shared-memory SUMO binding) falls back to traci.
OPTIONAL_GUARDED = {"libsumo"}
OWN_PACKAGES = {"opencda_marl", "opencda"}


@pytest.fixture(scope="module")
def manifest():
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _requirement_name(spec: str) -> str:
    """``"opencv-python>=4 ; sys_platform"`` -> ``opencv-python``."""
    return re.split(r"[\s\[<>=!~;]", spec.strip(), maxsplit=1)[0].lower()


def _declared(manifest) -> set[str]:
    project = manifest["project"]
    names = {_requirement_name(d) for d in project.get("dependencies", [])}
    for extra in project.get("optional-dependencies", {}).values():
        names |= {_requirement_name(d) for d in extra}
    names.discard("opencda-marl")          # the self-referencing `all` extra
    return names


def _top_level_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:  # pragma: no cover - a broken file is another test's problem
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _dotted_imports(path: Path, prefix: str) -> set[str]:
    """Imports of ``prefix.*`` as dotted module names, for the reachability walk."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:  # pragma: no cover
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix):
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names |= {a.name for a in node.names if a.name.startswith(prefix)}
    return names


def _engine_modules() -> list[Path]:
    return sorted((REPO / "opencda_marl").rglob("*.py"))


def _core_modules_reachable_from_engine() -> dict[str, Path]:
    """The vendored-core modules the engine imports, transitively.

    The whole core ships in the wheel, but only what the engine reaches is
    the engine's responsibility to declare; OpenCDA's own scenario tooling
    under ``opencda/scenario_testing`` has needs of its own.
    """
    modules: dict[str, Path] = {}
    for path in (REPO / "opencda").rglob("*.py"):
        name = ".".join(path.relative_to(REPO).with_suffix("").parts)
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        modules[name] = path
    todo: set[str] = set()
    for path in _engine_modules():
        todo |= {m for m in _dotted_imports(path, "opencda.")}
    reached: dict[str, Path] = {}
    while todo:
        name = todo.pop()
        if name in reached or name not in modules:
            continue
        reached[name] = modules[name]
        todo |= _dotted_imports(modules[name], "opencda.")
    return reached


def _engine_modules_importing(package: str) -> list[str]:
    return sorted(str(p.relative_to(REPO)) for p in _engine_modules()
                  if package in _top_level_imports(p))


# ─────────────────────────────────────────────────────────────────────────────
# Contents
# ─────────────────────────────────────────────────────────────────────────────

def test_every_package_the_engine_imports_is_packaged(manifest):
    """The engine may not import a top-level package the wheel omits."""
    include = manifest["tool"]["setuptools"]["packages"]["find"]["include"]
    packaged = {pattern.rstrip("*") for pattern in include}
    importers = _engine_modules_importing("opencda")
    assert importers, "expected the engine to import the vendored core"
    assert "opencda" in packaged, (
        f"opencda_marl imports the vendored core from {len(importers)} modules "
        f"({importers[:3]} ...), so the wheel must ship it; packages.find.include is {include}")


def test_the_core_and_its_data_are_declared(manifest):
    package_data = manifest["tool"]["setuptools"]["package-data"]
    assert "opencda" in package_data, "the vendored core ships data files"
    assert any("blueprint_meta" in g for g in package_data["opencda"]), (
        "two shipped profiles read opencda/assets/blueprint_meta by path")


def test_the_unreal_map_package_stays_out(manifest):
    """The 19 MB .fbx has to be imported into the simulator anyway; shipping
    it would triple the wheel for something a wheel cannot deliver."""
    globs = manifest["tool"]["setuptools"]["package-data"]["opencda"]
    assert not any("2lane_freeway_complete" in g for g in globs)
    assert not any(g.startswith("assets/*") or g == "assets/**" for g in globs), (
        "a broad assets glob would sweep in the Unreal map package")


# ─────────────────────────────────────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────────────────────────────────────

def test_every_third_party_import_is_declared(manifest):
    """Everything the engine, or the core it reaches, imports is in the base
    dependencies or in an extra -- except the one documented exception."""
    declared = _declared(manifest)
    needed: dict[str, set[str]] = {}
    sources = list(_engine_modules()) + list(_core_modules_reachable_from_engine().values())
    for path in sources:
        for name in _top_level_imports(path):
            if name in STDLIB or name in OWN_PACKAGES:
                continue
            needed.setdefault(DISTRIBUTION.get(name, name).lower(), set()).add(
                str(path.relative_to(REPO)))
    missing = {d: sorted(f) for d, f in needed.items()
               if d not in declared and d not in UNDECLARED_BY_DESIGN
               and d not in OPTIONAL_GUARDED}
    assert not missing, (
        "imported but declared nowhere (add to dependencies or an extra):\n" +
        "\n".join(f"  {d}: {f[:3]}" for d, f in sorted(missing.items())))


_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _unguarded_imports(path: Path, names: set[str]) -> list[int]:
    """Line numbers where ``path`` imports one of ``names`` outside the body
    of a ``try`` that catches ImportError."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:  # pragma: no cover
        return []

    def catches_import_error(node: ast.Try) -> bool:
        for handler in node.handlers:
            if handler.type is None:
                return True
            types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
            if any(isinstance(t, ast.Name) and t.id in _IMPORT_ERRORS for t in types):
                return True
        return False

    lines: list[int] = []

    def visit(node: ast.AST, guarded: bool) -> None:
        if isinstance(node, ast.Import):
            hit = any(a.name.split(".")[0] in names for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            hit = bool(node.module) and node.level == 0 and node.module.split(".")[0] in names
        else:
            hit = False
        if hit and not guarded:
            lines.append(node.lineno)
        if isinstance(node, ast.Try):
            body_guarded = guarded or catches_import_error(node)
            for child in node.body:
                visit(child, body_guarded)
            for child in [*node.handlers, *node.orelse, *node.finalbody]:
                visit(child, guarded)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    visit(tree, False)
    return lines


def test_optional_bindings_are_imported_behind_a_fallback():
    """An undeclared optional binding is only safe while every import of it
    is guarded: a bare install without it must still run on the fallback."""
    sources = list(_engine_modules()) + list(_core_modules_reachable_from_engine().values())
    importers = {str(p.relative_to(REPO)): _unguarded_imports(p, OPTIONAL_GUARDED)
                 for p in sources}
    assert any(OPTIONAL_GUARDED & _top_level_imports(p) for p in sources), (
        "expected the SUMO binding selector to import libsumo")
    unguarded = {path: lines for path, lines in importers.items() if lines}
    assert not unguarded, (
        f"{sorted(OPTIONAL_GUARDED)} is undeclared, so every import needs an "
        f"ImportError fallback; unguarded at: {unguarded}")


def test_the_sumo_backend_needs_no_extra(manifest):
    """The default install must run the SUMO backend: its client is a base
    dependency, and the CARLA libraries are extras, never base."""
    base = {_requirement_name(d) for d in manifest["project"]["dependencies"]}
    assert "eclipse-sumo" in base
    extras = manifest["project"]["optional-dependencies"]
    assert "carla" in extras and "ports" in extras
    for heavy in ("open3d", "opencv-python", "matplotlib"):
        assert heavy not in base, f"{heavy} is a CARLA-path library and belongs in the carla extra"
    assert "carla" not in _declared(manifest), "carla is version-locked to the simulator"
