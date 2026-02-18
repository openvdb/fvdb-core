# AGENTS.md — AI Agent Guidelines for fVDB

This file provides persistent instructions for AI coding agents (Cursor, Copilot,
Codex, etc.) working on this codebase. Human contributors should also refer to
[CONTRIBUTING.md](CONTRIBUTING.md).

## Git Commits

- All commits **must** include a DCO sign-off (`git commit --signoff` / `-s`).
- All commits **must** be SSH-signed (`git commit --gpg-sign` / `-S`).
- **Never** skip hooks or signing: do not use `--no-verify` or `--no-gpg-sign`.

## General Style

- Avoid excessive use of emoji and non-ascii characters in code and documentation except
  where it is helpful to user experience or user interfaces.

## Python Code Style

- Format Python code with **black** using the exact flags from CI:

  ```
  black --target-version=py311 --line-length=120 --extend-exclude='wip/' .
  ```

- **Do not** run `black` with default settings — the default line-length (88) does
  not match this project's setting (120).
- There is currently no `[tool.black]` section in `pyproject.toml`; a future PR may
  add one so that plain `black .` picks up the correct settings automatically.

## C++ Code Style

- Format C++ source files under `src/` with **clang-format 18** using the
  repository's `.clang-format` style file:

  ```
  clang-format -i --style=file src/**/*.{h,cpp,cc,cu,cuh}
  ```

- Applicable extensions: `.h`, `.cpp`, `.cc`, `.cu`, `.cuh`.

## License Headers

Every source file **must** include the Apache-2.0 SPDX identifier.

Python files:

```python
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
```

C++ files:

```cpp
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
```

## Whitespace

- No trailing whitespace (CI enforces this; `.wlt` files and `wip/` are excluded).
- Use spaces, not tabs (binary and a few config files are excluded from this check).

## Testing

- Run relevant tests before pushing:

  ```
  python -m pytest tests/ -v
  ```

- The `wip/` directory is excluded from test collection (see `norecursedirs` in
  `pyproject.toml`).

## Opening Issues

- Reference any related issues or PRs.
- Set appropriate labels (e.g. `bug`, `enhancement`, `documentation`).
- For bugs: provide clear reproduction steps, expected vs actual behavior, and
  environment details.

## Opening Pull Requests

- Reference the issue being fixed (e.g. "Fixes #NNN").
- Include a test plan with specific commands to verify the change.
- Ensure all CI checks pass before requesting review (DCO, codestyle, tests).
- Keep PRs focused on a single concern.
- Set appropriate labels matching the linked issue.

**Do not commit unless directed:**
- `.vscode/settings.json` - local IDE settings
- `.cursor/` - local Cursor config
- Test-specific matrices or configs with hardcoded paths
