# Contributing

Contributions — by people and by coding agents — are welcome. The process
is the one in [Agent Workflow](agents/workflow.md); this page is the short
form.

Development here is specified **agent-first**: the canonical guide is
[`AGENTS.md`](https://github.com/radar-lab/OpenCDA-MARL/blob/main/AGENTS.md)
at the repository root, routing to the same
[Agent Workflow](agents/workflow.md), [Code Index](agents/code-index.md)
and [Conventions](agents/conventions.md) this site hosts. Point a coding
agent at a checkout and it reads exactly the rules a human contributor
does — one rule set, no drift.

Contributions are accepted under the Apache License 2.0 that covers the
OpenCDA-MARL code (see
[`LICENSE`](https://github.com/radar-lab/OpenCDA-MARL/blob/main/LICENSE)).
Files from third parties keep their own terms; see
[`THIRD_PARTY_NOTICES.md`](https://github.com/radar-lab/OpenCDA-MARL/blob/main/THIRD_PARTY_NOTICES.md).

1. **Fork** `radar-lab/OpenCDA-MARL` and branch from `main`
   (`git switch -c feat/short-summary`).
2. **Make one verified change.** Tests ship with the change; `pixi run
   pytest test -q` is green; lifecycle, launcher and monitor changes get
   a live smoke run. Docs are updated on the page that owns the feature.
3. **Open a pull request** against `main` with the
   [PR template](https://github.com/radar-lab/OpenCDA-MARL/blob/main/.github/PR_TEMPLATE.md)
   filled in — what changed, how it was verified, what you could not verify.
   Maintainers review and squash-merge.

Where things go:

- A new **algorithm** does not need a PR at all: package it as described in
  [Add an Algorithm](marl/custom_algorithms.md) and it plugs into the
  framework from outside. PR it only if you want it shipped as a built-in.
- A new **scenario or profile** is a YAML file under
  `opencda_marl/_bundled/configs/<family>/`
  that states only what it changes from the family default and scenario
  base.
- A **framework fix** keeps the seams listed in the workflow page: one
  episode loop, registration instead of name dispatch, every `opencda/`
  patch recorded in the [patch ledger](opencda/marl_patches.md).

Bugs and proposals go to
[issues](https://github.com/radar-lab/OpenCDA-MARL/issues); include the
scenario reference, the command, and the log lines around the failure.
