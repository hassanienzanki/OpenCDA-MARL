"""The topology is enforced, not aspirational.

The kernel (``opencda_marl/_kernel``) owns the complete lifecycle and is
sealed: it may not import a concrete plugin implementation
(``opencda_marl.builtin``) — every replaceable part is reached through
``opencda_marl.api`` and the plugin registry. Conversely the built-in
plugin set may not reach into the kernel: it implements ``api`` contracts
only. ``api`` and ``plugins`` are the meeting point and depend on neither
side.

One documented exception: ``_kernel.traffic.sumo_adapter`` is the typed
CARLA stand-in for slim (no-carla) installs; ``api``/``builtin`` modules
may import it *as carla* in their ImportError fallback and nowhere else.
"""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "opencda_marl"

SUMO_ADAPTER_SHIM = "opencda_marl._kernel.traffic.sumo_adapter"


def _imports_of(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                yield f"{node.module}.{alias.name}"


def _violations(subtree: str, forbidden_prefix: str, *, allow_shim: bool = False):
    found = []
    for path in sorted((PACKAGE / subtree).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for module in _imports_of(path):
            if not module.startswith(forbidden_prefix):
                continue
            if allow_shim and (module == SUMO_ADAPTER_SHIM
                               or module.startswith(SUMO_ADAPTER_SHIM + ".")):
                continue
            found.append(f"{path.relative_to(PACKAGE.parent)} imports {module}")
    return found


def test_kernel_never_imports_builtin():
    assert not _violations("_kernel", "opencda_marl.builtin"), \
        "\n".join(_violations("_kernel", "opencda_marl.builtin"))


def test_builtin_never_imports_kernel():
    bad = _violations("builtin", "opencda_marl._kernel", allow_shim=True)
    assert not bad, "\n".join(bad)


def test_api_never_imports_kernel_or_builtin():
    bad = (_violations("api", "opencda_marl._kernel", allow_shim=True)
           + _violations("api", "opencda_marl.builtin"))
    assert not bad, "\n".join(bad)


def test_plugin_registry_imports_neither_side():
    bad = (_violations("plugins", "opencda_marl._kernel")
           + _violations("plugins", "opencda_marl.builtin"))
    assert not bad, "\n".join(bad)


def test_kernel_is_underscore_private_and_documented():
    init = PACKAGE / "_kernel" / "__init__.py"
    assert init.exists()
    text = init.read_text(encoding="utf-8")
    assert "Sealed" in text and "architecture test" in text
