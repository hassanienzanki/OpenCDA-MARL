import os
from pathlib import Path
from typing import Iterable


MARL_CONFIG_FAMILIES = ("marl", "pcma", "momarl", "sumo-marl", "cosim-marl")
MARL_CONFIG_ROOTS_ENV = "OPENCDA_MARL_CONFIG_ROOTS"


def resolve_marl_config_paths(
    base_dir: str | Path,
    scenario_ref: str,
    custom_config_roots: str | Path | Iterable[str | Path] | None = None,
) -> dict:
    """
    Resolve MARL-family configuration paths.

    The default config root is ``configs/``. Additional config roots can be
    supplied with ``custom_config_roots`` or the ``OPENCDA_MARL_CONFIG_ROOTS``
    environment variable. Each root must contain family directories such as
    ``marl/``, ``pcma/``, or ``momarl/``.

    Supported forms:
    - ``<family>/<scenario>/<profile>``
      Profile YAML at ``<config_root>/<family>/<scenario>/<profile>.yaml``.
    - ``<family>/<scenario>/<subdir>/<profile>``
      Profile YAML at
      ``<config_root>/<family>/<scenario>/<subdir>/<profile>.yaml``.
      The scenario base.yaml stays one level above the profile YAML.

    Examples:
    - ``marl/intersection/vanilla``
    - ``pcma/intersection/pcmasac_directpref_vector``
    - ``momarl/intersection/learn_path_pcmappo``
    """
    root = Path(base_dir)
    normalized_ref = scenario_ref.replace("\\", "/").strip("/")
    if not normalized_ref:
        raise ValueError("Scenario reference cannot be empty.")

    explicit_family, scenario_name, sub_dirs, profile_name = _split_explicit_family(
        normalized_ref
    )
    search_families: Iterable[str] = (explicit_family,)

    searched_paths: list[str] = []
    for family in search_families:
        family_roots = _candidate_family_roots(root, family, custom_config_roots)
        for family_root in family_roots:
            profile_dir = family_root / scenario_name
            for sub in sub_dirs:
                profile_dir = profile_dir / sub
            scenario_yaml = profile_dir / f"{profile_name}.yaml"
            searched_paths.append(str(scenario_yaml))
            if scenario_yaml.is_file():
                # The family default and the scenario base may live in any
                # candidate root (this one first, then the others in order):
                # an external package can ship a single profile YAML under
                # its own root and inherit the framework's defaults.
                default_yaml = _first_existing(
                    [family_root / "default.yaml"]
                    + [r / "default.yaml" for r in family_roots if r != family_root]
                )
                scenario_base_yaml = _first_existing(
                    [family_root / scenario_name / "base.yaml"]
                    + [r / scenario_name / "base.yaml" for r in family_roots if r != family_root]
                )
                if default_yaml is None:
                    raise FileNotFoundError(
                        f"MARL default config not found: {family_root / 'default.yaml'}"
                    )
                if scenario_base_yaml is None:
                    raise FileNotFoundError(
                        "MARL scenario base config not found: "
                        f"{family_root / scenario_name / 'base.yaml'}"
                    )
                return {
                    "family": family,
                    "family_root": family_root,
                    "scenario_name": scenario_name,
                    "profile_name": profile_name,
                    "default_yaml": default_yaml,
                    "scenario_base_yaml": scenario_base_yaml,
                    "scenario_yaml": scenario_yaml,
                }

    searched = "\n".join(searched_paths)
    raise FileNotFoundError(
        f"MARL scenario config not found for '{scenario_ref}'. Searched:\n{searched}"
    )


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _split_explicit_family(normalized_ref: str) -> tuple[str, str, tuple[str, ...], str]:
    parts = normalized_ref.split("/")
    if len(parts) < 3 or parts[0] not in MARL_CONFIG_FAMILIES:
        raise ValueError(
            "MARL scenario references must be in the form "
            "'<family>/<scenario>/<profile>' or "
            "'<family>/<scenario>/<subdir>/.../<profile>'."
        )
    family = parts[0]
    scenario_name = parts[1]
    profile_name = parts[-1]
    sub_dirs = tuple(parts[2:-1])  # may be empty for the 3-part form
    return family, scenario_name, sub_dirs, profile_name


def _candidate_family_roots(
    root: Path,
    family: str,
    custom_config_roots: str | Path | Iterable[str | Path] | None = None,
) -> tuple[Path, ...]:
    return tuple(
        config_root / family
        for config_root in _candidate_config_roots(root, custom_config_roots)
    )


def _candidate_config_roots(
    root: Path,
    custom_config_roots: str | Path | Iterable[str | Path] | None = None,
) -> tuple[Path, ...]:
    roots: list[Path] = [root / "configs"]
    env_roots = os.environ.get(MARL_CONFIG_ROOTS_ENV, "")
    extra_roots: list[str | Path] = []
    if env_roots:
        extra_roots.extend(value for value in env_roots.split(os.pathsep) if value)
    if isinstance(custom_config_roots, (str, Path)):
        extra_roots.append(custom_config_roots)
    elif custom_config_roots is not None:
        extra_roots.extend(custom_config_roots)

    for value in extra_roots:
        path = Path(value).expanduser()
        roots.append(path if path.is_absolute() else root / path)

    # Last resort: the config family bundled into the engine wheel, so a
    # pip-installed framework resolves scenarios without a repository
    # checkout. A checkout's own configs/ always wins (it comes first).
    try:
        from opencda_marl._bundled import bundled_config_root

        bundled = bundled_config_root()
        if bundled is not None:
            roots.append(bundled)
    except Exception:  # pragma: no cover - the bundle is optional
        pass

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for path in roots:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique_roots.append(path)
    return tuple(unique_roots)


def load_profile_with_includes(profile_path: str | Path):
    """Load a profile YAML, composing its ``include:`` list first.

    A profile may name fragment files to merge underneath itself::

        include:
          - fragments/dense_traffic      # same scenario directory
          - ../shared/short_episodes     # any path relative to the profile

    Fragments merge in list order (later fragments override earlier ones)
    and the profile's own keys override them all — the profile stays "only
    what it changes", and a project composes shared fragments instead of
    copying blocks. Fragments may include further fragments; cycles fail
    loudly. Returns an OmegaConf config with the ``include`` key removed.
    """
    from omegaconf import OmegaConf

    from opencda_marl.utils.config_merger import smart_merge

    def _load(path: Path, stack: tuple) -> "object":
        path = path.resolve()
        if path in stack:
            chain = " -> ".join(str(p) for p in (*stack, path))
            raise ValueError(f"config include cycle: {chain}")
        cfg = OmegaConf.load(path)
        includes = cfg.pop("include", None) if hasattr(cfg, "pop") else None
        if not includes:
            return cfg
        if isinstance(includes, str):
            includes = [includes]
        merged = None
        for ref in includes:
            ref_path = (path.parent / str(ref)).with_suffix(".yaml")
            if not ref_path.is_file():
                raise FileNotFoundError(
                    f"{path.name} includes {ref!r}, but {ref_path} does not exist")
            fragment = _load(ref_path, (*stack, path))
            merged = fragment if merged is None else smart_merge(merged, fragment)
        return smart_merge(merged, cfg) if merged is not None else cfg

    return _load(Path(profile_path), ())
