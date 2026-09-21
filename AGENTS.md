# Project instructions

- Read [README](README.md) and the assigned work package in
  [the roadmap](docs/ROADMAP.md), then follow [the agent workflow](docs/AGENT_WORKFLOW.md).
- Follow the roadmap's current recovery sequence before advancing P0–P10. Give
  each assignment acceptance evidence and a diagnostic for its likely failure;
  choose routine implementation details autonomously.
- Resume from STATUS's current handoff after interruption; do not restart finished
  experiments or infer a user-decision blocker from a failed performance gate.
- Respect the latest user instructions over planning defaults. Preserve the shared
  learned feeling, deadline-first objective, physical load limits, and required
  learning-progress visualization.
- Use GPU models and GPU simulation where the measured capacity permits. Report
  explicit fallbacks; do not silently substitute CPU execution.
- The GPU is accessible outside the sandbox. Run GPU checks/work on the authorized
  host through the project wrapper; sandbox driver failure is not a CPU-fallback reason.
- Keep environments, downloaded tools/interpreters, all caches, temporary files,
  datasets, and outputs inside this project. Use the local wrapper specified in
  [LOCAL_SETUP](docs/LOCAL_SETUP.md); do not change host Python, drivers, or shell config.
- Implement one bounded package at a time and update [STATUS](docs/STATUS.md) with
  actual acceptance evidence. Do not mark research gates complete from smoke tests.
- For new copyright notices use `Copyright (C) 2026 Florian Loitsch. All rights reserved.` with the
  comment syntax appropriate to the file language.
