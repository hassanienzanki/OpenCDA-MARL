"""The eclipse-sumo package's own tools directory provides traci and sumolib.

``eclipse-sumo`` ships its Python bindings under ``site-packages/sumo/tools``,
which a plain ``pip install`` does not put on ``sys.path``; only an
environment that also installs the separate ``traci``/``sumolib``
distributions could import them. The resolver now puts the package's tools
directory first on ``sys.path`` itself, so the bindings import from the
package alone and always match its binary.
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

from opencda_marl._kernel.traffic import _sumo_home

_BINDINGS = ("traci", "sumolib")


def _is_binding(module_name: str) -> bool:
    return module_name.split(".")[0] in _BINDINGS


@pytest.fixture
def fresh_resolution(monkeypatch):
    """A first resolution in this process: the bindings unimported and the
    SUMO variables unset; sys.path, sys.modules and the environment are
    restored afterwards."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = {name: module for name, module in sys.modules.items() if _is_binding(name)}
    for name in saved:
        del sys.modules[name]
    for var in ("SUMO_HOME", "PROJ_LIB", "PROJ_DATA"):
        monkeypatch.setenv(var, "placeholder")   # recorded, so teardown restores the original
        monkeypatch.delenv(var)
    monkeypatch.setattr(_sumo_home, "_RESOLVED", None)
    yield
    for name in [name for name in sys.modules if _is_binding(name)]:
        del sys.modules[name]
    sys.modules.update(saved)


def _package(tmp_path):
    """A stand-in eclipse-sumo package: bin/ plus tools/ holding both bindings,
    whose metadata-less version is the placeholder 0.0.0."""
    root = tmp_path / "sumo"
    (root / "bin").mkdir(parents=True)
    for name in _BINDINGS:
        package = root / "tools" / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            '__version__ = "0.0.0"\nFROM_PACKAGE_TOOLS = True\n', encoding="utf-8")
    return root


def test_the_package_tools_make_both_bindings_importable(fresh_resolution, monkeypatch, tmp_path):
    root = _package(tmp_path)
    monkeypatch.setattr(_sumo_home, "_package_sumo_home", lambda: root)
    info = _sumo_home.resolve_sumo_home()
    assert sys.path[0] == str(root / "tools")
    for name in _BINDINGS:
        module = importlib.import_module(name)
        assert getattr(module, "FROM_PACKAGE_TOOLS", False), module.__file__
    assert os.environ["SUMO_HOME"] == str(root)
    # The tools copy has no distribution metadata, so it reports 0.0.0; that
    # is not a client version to hold against the binary's.
    assert info["client_version"] is None


def test_a_stale_sumo_home_tools_entry_never_shadows_the_package(fresh_resolution, monkeypatch, tmp_path):
    root = _package(tmp_path)
    stale = tmp_path / "older_sumo"
    monkeypatch.setenv("SUMO_HOME", str(stale))
    sys.path.insert(0, str(stale / "tools"))
    monkeypatch.setattr(_sumo_home, "_package_sumo_home", lambda: root)
    _sumo_home.resolve_sumo_home()
    assert str(stale / "tools") not in sys.path
    assert sys.path[0] == str(root / "tools")
    monkeypatch.setattr(_sumo_home, "_RESOLVED", None)
    _sumo_home.resolve_sumo_home()
    assert sys.path.count(str(root / "tools")) == 1, "a second resolution adds no duplicate"


def test_the_installed_package_provides_both_bindings_from_its_tools(fresh_resolution):
    root = _sumo_home._package_sumo_home()
    if root is None:
        pytest.skip("eclipse-sumo is not installed in this environment")
    _sumo_home.resolve_sumo_home()
    tools = (root / "tools").resolve()
    for name in _BINDINGS:
        module = importlib.import_module(name)
        assert tools in Path(module.__file__).resolve().parents, module.__file__


def test_every_other_sumo_tools_directory_leaves_sys_path(fresh_resolution, monkeypatch, tmp_path):
    """A system SUMO's tools on PYTHONPATH (not named by SUMO_HOME) is dropped too.

    With SUMO_HOME pointing at the package (or unset), a user-level
    PYTHONPATH entry for another SUMO's tools used to survive on sys.path,
    one import-order accident away from pairing that version's bindings with
    the package's binary.
    """
    root = _package(tmp_path)
    # A system install laid out as <prefix>/Sumo/tools, shipping both bindings ...
    system_tools = tmp_path / "Eclipse" / "Sumo" / "tools"
    for name in _BINDINGS:
        (system_tools / name).mkdir(parents=True)
    # ... and one under any other name that ships both bindings ...
    other_tools = tmp_path / "opt" / "traffic-sim" / "tools"
    for name in _BINDINGS:
        (other_tools / name).mkdir(parents=True)
    # ... and paths that are not SUMO tools: a plain library directory, a
    # project's own tools/ without the bindings (also under a sumo* folder),
    # and a site-packages that holds the separate traci/sumolib distributions.
    plain = tmp_path / "lib"
    plain.mkdir()
    project_tools = tmp_path / "project" / "tools"
    project_tools.mkdir(parents=True)
    named_project_tools = tmp_path / "sumo-experiments" / "tools"
    named_project_tools.mkdir(parents=True)
    site_packages = tmp_path / "site-packages"
    for name in _BINDINGS:
        (site_packages / name).mkdir(parents=True)
    # Trailing separator, as PYTHONPATH may carry it.
    sys.path.insert(0, str(system_tools) + os.sep)
    sys.path.insert(1, str(other_tools))
    sys.path.extend([str(plain), str(project_tools), str(named_project_tools),
                     str(site_packages)])
    monkeypatch.setenv("SUMO_HOME", str(root))
    monkeypatch.setattr(_sumo_home, "_package_sumo_home", lambda: root)

    _sumo_home.resolve_sumo_home()

    assert sys.path[0] == str(root / "tools")
    remaining = {os.path.normcase(os.path.normpath(p)) for p in sys.path if isinstance(p, str) and p}
    for foreign in (system_tools, other_tools):
        assert os.path.normcase(str(foreign)) not in remaining, sys.path
    for kept in (plain, project_tools, named_project_tools, site_packages):
        assert str(kept) in sys.path, kept


def test_the_package_tools_move_first_when_already_on_sys_path(fresh_resolution, monkeypatch, tmp_path):
    root = _package(tmp_path)
    plain = tmp_path / "lib"
    plain.mkdir()
    sys.path.insert(0, str(plain))
    sys.path.append(str(root / "tools"))
    monkeypatch.setattr(_sumo_home, "_package_sumo_home", lambda: root)
    _sumo_home.resolve_sumo_home()
    assert sys.path[0] == str(root / "tools")
    assert sys.path.count(str(root / "tools")) == 1
    assert str(plain) in sys.path
