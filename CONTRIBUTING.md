# How to contribute to LeRobot 3D

Everyone is welcome to contribute, and we value every contribution. Code isn't the only
way to help — reporting bugs, answering questions, and improving documentation all
matter too. Please be respectful and constructive in issues and pull requests.

## Ways to contribute

- **Fixing issues** — resolve bugs or improve existing code.
- **New features** — see the [roadmap](#roadmap) below for ideas, or open an issue to
  propose your own.
- **Extend** — support new robots/arms, cameras, or calibration methods.
- **Documentation** — improve the README, docstrings, or code comments.
- **Feedback** — open an issue for bugs or feature requests.

## Development setup

### 1. Fork and clone

```bash
git clone https://github.com/<your-handle>/lerobot_3d.git
cd lerobot_3d
git remote add upstream https://github.com/SergioMOrozco/lerobot_3d.git
```

### 2. Install

```bash
pip install -e ".[realsense,test]"
```

See the [README](README.md#install) for details on the `realsense` extra.

## Running tests & quality checks

### Linting

We use [ruff](https://docs.astral.sh/ruff/):

```bash
ruff check .
```

This is the same check CI runs; please run it before opening a PR.

### Tests

Tests are split into two tiers (see `tests/pure/` and `tests/hardware_stack/`):

```bash
pytest tests/pure -q       # stdlib + PyYAML only, always runnable
pytest tests -q            # needs open3d/urchin/scipy/lerobot/viser installed
```

No physical robot or camera is required for either tier.

## Submitting issues & pull requests

- **Issues**: include a clear description and, for bugs, steps to reproduce.
- **Pull requests**: branch off `main` (don't work directly on it), rebase on
  `upstream/main` before opening, run `ruff check .` and the test suite locally, and
  describe *why* the change is needed, not just what it does.

## Roadmap

Contribution ideas we'd like to see tackled — feel free to open an issue to discuss
approach before starting, or just open a draft PR:

- [ ] Calibrating multiple robots in a single environment
- [ ] 3D mapping for mobile manipulation
- [ ] Calibration using multiple pictures per camera (instead of a single capture)
