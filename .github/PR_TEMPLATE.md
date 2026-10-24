<!--
Keep the title short and in conventional-commit form: feat(scope): summary.
Fill every section; delete the example text. Agent-authored PRs are welcome
— say which agent, and be explicit about what could not be verified.
-->

## What and why

<!-- One paragraph: what was wrong or missing, what this changes, what it deliberately does not change. -->

- **Kind**: bug fix / feature / performance / docs / refactor
- **Scope**: (`opencda_marl/...`, `opencda_marl/_bundled/configs/...`, `scripts/experiments/...`, `docs/...`)
- **Authored by**: human / agent (`name`) reviewed by human
- **Related**: issue or PR, if any

## How it was verified

<!-- Paste the commands and the result lines you actually observed. -->

```
pixi run pytest test -q            -> N passed, M skipped
<smoke run command>                -> episodes completed, counters read (training_step, updates, buffer)
```

Not verified here (say why, e.g. no CARLA / no GPU):

## Checklist

- [ ] One concern per PR; unrelated cleanups are separate PRs
- [ ] Tests added or updated; the suite is green on the platform I changed
- [ ] Live smoke run for lifecycle / launcher / checkpoint / GUI / monitor changes
- [ ] `opencda/` edits are recorded in `docs/opencda/marl_patches.md`
- [ ] Docs updated on the owning page (`docs/run-experiments.md`, `docs/marl/custom_algorithms.md`, `docs/agents/workflow.md`, `docs/agents/code-index.md`); `pixi run -e docs mkdocs build --strict` passes
- [ ] No new algorithm-name dispatch — capabilities declared through `register_algorithm`
- [ ] No generated files (`artifacts/_runtime/`, `artifacts/`, checkpoints), no changes to `opencda_marl/_bundled/recordings/*.json`
- [ ] No secrets, hostnames, absolute paths or private URLs
