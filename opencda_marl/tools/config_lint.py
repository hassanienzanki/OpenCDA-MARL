"""Static conformance checks for every shipped scenario config.

Config mistakes such as a legacy algorithm name or a missing
``MARL.action_space`` otherwise surface only when a scenario is *launched*.
This module makes the same checks a static invariant: resolve + merge every
profile under the config roots exactly as ``opencda.py`` would, then validate
the ``MARL`` block through the same resolvers the manager uses — without
building an environment or touching a simulator.

Run as a test (``test/basic/test_config_lint.py``) or ad hoc::

    pixi run python -m opencda_marl.tools.config_lint            # all roots
    pixi run python -m opencda_marl.tools.config_lint configs/marl
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

from omegaconf import OmegaConf

from opencda_marl.utils import config_locator, config_merger

# Family directories that hold scenario profiles (family/scenario/profile.yaml).
_SKIP_FILES = {"default.yaml", "base.yaml"}


@dataclass
class LintResult:
    scenario_ref: str
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def iter_scenario_refs(config_root: Path) -> Iterable[str]:
    """Yield ``family/scenario/profile`` refs for every profile YAML."""
    for family_dir in sorted(p for p in config_root.iterdir() if p.is_dir()):
        for scenario_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
            if scenario_dir.name.startswith("_"):
                continue
            for profile in sorted(scenario_dir.glob("*.yaml")):
                if profile.name in _SKIP_FILES:
                    continue
                yield f"{family_dir.name}/{scenario_dir.name}/{profile.stem}"


def merged_config(repo_root: Path, scenario_ref: str, config_root: Path) -> dict:
    resolved = config_locator.resolve_marl_config_paths(
        repo_root, scenario_ref, custom_config_roots=[config_root])
    merged = config_merger.smart_merge(
        OmegaConf.load(resolved["default_yaml"]),
        OmegaConf.load(resolved["scenario_base_yaml"]))
    merged = config_merger.smart_merge(merged, OmegaConf.load(resolved["scenario_yaml"]))
    return OmegaConf.to_container(merged, resolve=True)


def lint_marl_block(marl_cfg: dict, agents_cfg: dict | None = None) -> List[str]:
    """Validate a merged ``MARL`` block through the runtime resolvers."""
    from opencda_marl._kernel.policy.action_space import resolve_action_space_config
    from opencda_marl.api.algorithms import resolve_algorithm_spec
    from opencda_marl.api.rewards import resolve_reward_configuration

    errors: List[str] = []
    algorithm = marl_cfg.get("algorithm")
    spec = None
    if algorithm is not None:
        try:
            spec = resolve_algorithm_spec(algorithm)
        except Exception as exc:  # ValueError / NotImplementedError
            errors.append(f"MARL.algorithm: {exc}")
    try:
        resolve_action_space_config(marl_cfg)
    except Exception as exc:
        errors.append(f"MARL.action_space: {exc}")
    try:
        resolve_reward_configuration(marl_cfg)
    except Exception as exc:
        errors.append(f"MARL reward config: {exc}")

    if spec is not None:
        section = marl_cfg.get(spec.config_section)
        if not isinstance(section, dict):
            errors.append(
                f"MARL.{spec.config_section}: missing algorithm section for "
                f"algorithm={algorithm!r}")
        elif spec.capabilities.multi_agent and not section.get("features"):
            errors.append(
                f"MARL.{spec.config_section}.features: multi-agent algorithm "
                "needs an observation feature map")
        elif spec.capabilities.multi_agent:
            width = sum(int(v) for v in section["features"].values())
            pref_dim = _preference_dim(marl_cfg)
            configured = marl_cfg.get("state_dim")
            if configured is not None and int(configured) != width + pref_dim:
                errors.append(
                    f"MARL.state_dim={configured} != features({width}) + "
                    f"preference({pref_dim}) = {width + pref_dim}")
    return errors


def _preference_dim(marl_cfg: dict) -> int:
    """Mirror ``MARLManager.preference_dim``: augmentation is gated by the
    objective spec, and the width is ``MARL.preference.dim`` or the number of
    objectives."""
    from opencda_marl.api.objectives import resolve_objective_configuration

    try:
        spec = resolve_objective_configuration(marl_cfg).objective_spec
    except Exception:
        return 0
    if spec is None:
        return 0
    configured = (marl_cfg.get("preference") or {}).get("dim")
    return int(configured) if configured else len(spec.objectives)


def lint_scenario(repo_root: Path, scenario_ref: str, config_root: Path) -> LintResult:
    result = LintResult(scenario_ref)
    try:
        cfg = merged_config(repo_root, scenario_ref, config_root)
    except Exception as exc:
        result.errors.append(f"resolve/merge failed: {exc}")
        return result
    marl_cfg = cfg.get("MARL")
    if isinstance(marl_cfg, dict) and marl_cfg.get("algorithm") is not None:
        result.errors.extend(lint_marl_block(marl_cfg, cfg.get("agents")))
    return result


def lint_all(repo_root: Path, config_root: Path) -> List[LintResult]:
    return [lint_scenario(repo_root, ref, config_root)
            for ref in iter_scenario_refs(config_root)]


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    repo_root = Path(__file__).resolve().parents[2]
    config_root = repo_root / "configs"
    if not config_root.is_dir():
        # The shipped families live inside the package; a checkout-level
        # configs/ (an external root) takes precedence only when present.
        from opencda_marl._bundled import BUNDLED_CONFIG_ROOT
        config_root = BUNDLED_CONFIG_ROOT
    refs = list(iter_scenario_refs(config_root))
    if argv:
        wanted = [a.replace("\\", "/").rstrip("/") for a in argv]
        refs = [r for r in refs
                if any(r.startswith(w.replace("configs/", "")) for w in wanted)]
    failures = 0
    for ref in refs:
        result = lint_scenario(repo_root, ref, config_root)
        status = "ok " if result.ok else "ERR"
        print(f"[{status}] {ref}")
        for err in result.errors:
            failures += 1
            print(f"       - {err}")
    print(f"\n{len(refs)} profiles checked, {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
