"""Documentation boundary and freshness.

- The code index and the conventions name only paths that exist, so they
  stay maps rather than stories.
- The version is declared in three places that must agree, and the
  changelog page, its nav entry and the README heading carry that version.
- No public file or engine docstring points plugin authors at the
  ``opencda_marl.algorithms`` entry-point group: the loader scans only
  ``opencda_marl.plugins``, so a package following such a pointer would
  register under a group nothing reads.
- The public agent docs (``docs/agents/``) are built at their portal URLs
  but kept out of the human nav (``not_in_nav``); ``AGENTS.md`` is
  the only entry file, with no per-tool twin, so the rules exist once and
  cannot drift between tools.
- Every agent guide carries a ``Guide version X.Y — <milestone>`` line,
  the same X.Y across the set, and opens by saying when it is worth
  opening.

Every check reads only public files. The checks that guard the boundary
with the maintainer's private tooling live with that tooling, not here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

_PATH_RE = re.compile(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+|[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]*)`")
_FILE_SUFFIXES = (".py", ".md", ".yaml", ".yml", ".json", ".toml", ".html", ".cff", ".txt", ".sh")
#: Files a run writes; named in the index but not part of the tree.
_RUNTIME_FILES = {"training_metadata.json", "metrics.sqlite",
                  "artifacts/_runtime/", "artifacts/recordings/"}


def _public_files():
    """Everything a public reader sees, except the entry files (checked
    separately)."""
    for pattern in ("docs/**/*.md", ".github/**/*.md", "README.md", "CITATION.cff"):
        yield from REPO.glob(pattern)


@pytest.mark.parametrize("guide", ["code-index.md", "conventions.md"])
def test_map_guides_name_only_existing_paths(guide):
    text = (DOCS / "agents" / guide).read_text(encoding="utf-8")
    missing = []
    for match in _PATH_RE.finditer(text):
        token = match.group(1)
        if token in _RUNTIME_FILES or token.startswith(("http", "-", "../")):
            continue
        if "/" not in token and not token.endswith(_FILE_SUFFIXES):
            continue                                  # `env.episode`, `module.attr`
        if not (REPO / token).exists():
            # Bare module and family names are listed inside their section;
            # resolve them against the trees the guides cover.
            candidates = [c for root in ("opencda_marl", "opencda", "scripts", "test")
                          if (REPO / root).exists()
                          for c in (REPO / root).glob(f"**/{token}")]
            if not candidates:
                missing.append(token)
    assert not missing, f"{guide} names paths that do not exist: {missing}"


def test_version_declared_consistently():
    pixi = (REPO / "pixi.toml").read_text(encoding="utf-8")
    pixi_version = re.search(r'^version\s*=\s*"([^"]+)"', pixi, re.M).group(1)
    init = (REPO / "opencda_marl" / "__init__.py").read_text(encoding="utf-8")
    init_version = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.M).group(1)
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text(encoding="utf-8"))
    assert pixi_version == init_version == str(cff["version"]), (
        pixi_version, init_version, cff["version"])


class _MkDocsLoader(yaml.SafeLoader):
    """Safe loader that reads mkdocs.yml's ``!!python/...`` tags as None."""


_MkDocsLoader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def test_changelog_and_readme_name_the_declared_version():
    init = (REPO / "opencda_marl" / "__init__.py").read_text(encoding="utf-8")
    version = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.M).group(1)
    page = f"marl/changelog/v{version}.md"
    assert (DOCS / page).exists(), f"no changelog page for the declared version: docs/{page}"
    title = (DOCS / page).read_text(encoding="utf-8").splitlines()[0]
    assert title.startswith(f"# v{version}"), f"docs/{page} is titled {title!r}"
    config = yaml.load((REPO / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_MkDocsLoader)
    assert page in set(_nav_pages(config["nav"])), f"docs/{page} is not in the mkdocs nav"
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert re.search(rf"^### v{re.escape(version)}\s*$", readme, re.M), (
        f"README.md has no '### v{version}' heading")


#: The retired group name followed by "entry" (entry point / entry-point),
#: with any quoting or markup between them.
_RETIRED_GROUP_RE = re.compile(r"opencda_marl\.algorithms\W{0,4}\s*entry", re.I)


def test_no_public_text_names_the_retired_entry_point_group():
    sources = list(_public_files()) + list((REPO / "opencda_marl").rglob("*.py"))
    offenders = [str(path.relative_to(REPO)) for path in sources
                 if _RETIRED_GROUP_RE.search(path.read_text(encoding="utf-8", errors="replace"))]
    assert not offenders, (
        "these files point at the 'opencda_marl.algorithms' entry-point group; "
        f"the loader scans 'opencda_marl.plugins': {offenders}")


def _nav_pages(node):
    """Every page path in a (nested) MkDocs ``nav`` value."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _nav_pages(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _nav_pages(value)


def test_agent_guides_are_built_but_kept_out_of_the_nav():
    config = yaml.load((REPO / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_MkDocsLoader)
    guides = sorted(p.relative_to(DOCS).as_posix() for p in (DOCS / "agents").glob("*.md"))
    assert guides, "docs/agents/ holds no guides"
    assert "exclude_docs" not in config, "the agent guides must stay built at their URLs"
    hidden = {line.strip() for line in (config.get("not_in_nav") or "").splitlines() if line.strip()}
    nav = set(_nav_pages(config["nav"]))
    for page in guides:
        assert page in hidden, f"{page} is not listed under not_in_nav"
        assert page not in nav, f"{page} is in the human nav; agent guides stay out of it"


def test_agents_md_is_the_only_entry_file():
    assert (REPO / "AGENTS.md").exists(), "AGENTS.md must exist on every tree"
    assert not (REPO / "CLAUDE.md").exists(), (
        "CLAUDE.md is retired — AGENTS.md is the one entry file, with no per-tool twin")


#: Title, then ``Guide version X.Y — <milestone>``, then the "when to open" line.
_VERSION_RE = re.compile(r"\A# .+\n\nGuide version (\d+\.\d+) — .+\n\n(Open this|When it is worth opening)")


def test_guides_carry_lockstep_version_lines():
    versions = {}
    for path in (DOCS / "agents").glob("*.md"):
        match = _VERSION_RE.match(path.read_text(encoding="utf-8"))
        assert match, (f"{path.name} must open with its title, a 'Guide version X.Y — "
                       "<milestone>' line and a line saying when it is worth opening")
        versions[path.name] = match.group(1)
    assert versions, "no agent guides found"
    assert len(set(versions.values())) == 1, f"guide versions are not in lockstep: {versions}"


def test_agent_docs_open_with_jump_lists():
    for path in (DOCS / "agents").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert re.search(r"^- \[.+\]\(#.+\)$", text, re.M), f"{path.name} has no jump list"
