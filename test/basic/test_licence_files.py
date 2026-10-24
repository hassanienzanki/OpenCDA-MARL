"""The licence files say what the repository contains, and stay true.

OpenCDA-MARL's own code is Apache-2.0 (``LICENSE``); the vendored OpenCDA core
keeps UCLA's Academic Software License (``opencda/LICENSE``), which
``THIRD_PARTY_NOTICES.md`` reproduces and which the wheel carries beside the
Apache text. These checks fail when a file is edited, moved or dropped without
the others.
"""
import hashlib
import re
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, used by the Linux and CARLA 0.9.15 environments
    import tomli as tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NOTICES = REPO / "THIRD_PARTY_NOTICES.md"

pytestmark = pytest.mark.skipif(not NOTICES.is_file(), reason="not a source checkout")

# The text the Apache Software Foundation publishes at
# https://www.apache.org/licenses/LICENSE-2.0.txt
APACHE_2_0_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def test_root_licence_is_the_unmodified_apache_text():
    digest = hashlib.sha256((REPO / "LICENSE").read_bytes()).hexdigest()
    assert digest == APACHE_2_0_SHA256


def test_ucla_licence_sits_beside_the_core_and_is_reproduced_verbatim():
    ucla = (REPO / "opencda" / "LICENSE").read_text(encoding="utf-8").strip("\n")
    assert ucla.startswith("Academic Software License")
    assert ucla in NOTICES.read_text(encoding="utf-8")


def test_the_wheel_names_both_licences_and_every_licence_file_exists():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "Apache-2.0" in project["license"]
    assert "MIT" in project["license"].split()  # the CARLA files the wheel ships
    assert "LicenseRef-UCLA-Academic-Software-License" in project["license"]
    missing = [name for name in project["license-files"] if not (REPO / name).is_file()]
    assert not missing, f"license-files that do not exist: {missing}"
    assert {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "opencda/LICENSE"} <= set(project["license-files"])


def test_every_path_the_notices_name_exists():
    text = NOTICES.read_text(encoding="utf-8")
    roots = ("opencda/", "opencda_marl/", "test/", "scripts/", "docs/", "dependencies/")
    named = {m for m in re.findall(r"`([^`\n]+)`", text) if m.startswith(roots) and " " not in m}
    assert named, "the notices no longer name any repository path"
    missing = sorted(p for p in named if not (REPO / p).exists())
    assert not missing, f"THIRD_PARTY_NOTICES.md names paths that do not exist: {missing}"


_CODE_DIRS = ("opencda", "opencda_marl", "scripts", "test")


def _python_files():
    yield from (p for d in _CODE_DIRS for p in (REPO / d).rglob("*.py"))
    launcher = REPO / "opencda.py"
    if launcher.is_file():
        yield launcher


def test_no_reserved_rights_header_is_left_in_the_code():
    # The editor-generated header claimed every right over files that are Apache-2.0 or UCLA's.
    phrase = "All Rights" + " Reserved"
    left = [str(p.relative_to(REPO)) for p in _python_files()
            if phrase in p.read_text(encoding="utf-8", errors="replace")]
    assert not left, f"files still carry the old header line: {left}"


def test_a_file_that_declares_apache_also_names_the_copyright_holder():
    holder = "Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona"
    spdx = "SPDX-License-Identifier: Apache-2.0"
    bare = [str(p.relative_to(REPO)) for p in _python_files()
            if spdx in (text := p.read_text(encoding="utf-8", errors="replace")) and holder not in text]
    assert not bare, f"files declare {spdx} without the holder line: {bare}"


def test_the_notice_and_readme_say_the_complete_system_is_academic_only():
    # Our files stay Apache-2.0; running the system includes UCLA's core, which limits use to academic research.
    for name in ("NOTICE", "README.md"):
        text = " ".join((REPO / name).read_text(encoding="utf-8").split())
        assert "academic and nonprofit research" in text, name
