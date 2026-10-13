# Contributing to Aether3D

Thanks for your interest in Aether3D! This guide explains how to set up a
development environment and what we expect from changes.

## Development setup

See [docs/INSTALL.md](docs/INSTALL.md) for platform specifics. In short:

```bash
git clone https://github.com/Murisixz/Aether3D.git
cd Aether3D
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Before you open a pull request

Run the same checks as CI:

```bash
ruff check .            # lint
ruff format --check .   # formatting (run `ruff format .` to fix)
pytest                  # tests
```

All three must pass.

## Guidelines

- **Scope.** Keep changes focused. Large features or new subsystems need an
  issue and agreement on the design before code is written. Do not add
  placeholder classes or speculative abstractions for features that do not
  exist yet.
- **Read the architecture.** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
  describes module boundaries and conventions. Respect the dependency
  direction between modules.
- **Tests.** Every behaviour change needs tests. Bug fixes should come with a
  test that fails without the fix.
- **Public API.** Anything user-facing is exported from
  `src/aether3d/__init__.py`, has a docstring, and is documented in
  [docs/API.md](docs/API.md).
- **Errors.** Raise a subclass of `AetherError` with a message that says what
  was wrong and what value was received. Validate before mutating, so a
  failing call leaves state unchanged.
- **File format.** Any change to what is written to `.aether` files must bump
  `FORMAT_VERSION`, add a migration from the previous version with a test,
  and update [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md).
- **Dependencies.** New runtime dependencies need a strong justification and
  must use a license compatible with Apache-2.0.
- **Style.** Formatting and linting are enforced by ruff (configured in
  `pyproject.toml`). Use type hints on public functions.
- **Portability.** Use `pathlib`, avoid OS-specific assumptions, and keep CI
  green on Linux, macOS and Windows.

## Commits and pull requests

- Write commit messages in the imperative mood ("Add sphere segment
  validation"), with a short summary line and an explanatory body when useful.
- Keep pull requests small enough to review, and describe what changed and why.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
