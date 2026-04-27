"""Filesystem locations for bundled calibration assets (URDF, meshes)."""
from __future__ import annotations

import os
from pathlib import Path

# ``lerobot_playground`` package root (directory containing ``calibration/``).
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
CALIBRATION_DIR: Path = PACKAGE_ROOT / "calibration"


def resolve_extrinsic_calibration_json(path: str | Path) -> Path:
    """Resolve ``extrinsic_calibration.json`` (or any extrinsic JSON path).

    Order:

    1. ``LEROBOT_PLAYGROUND_EXTRINSIC_JSON`` if set (file must exist).
    2. ``path`` if absolute and exists.
    3. ``Path.cwd() / path`` if it exists.
    4. If ``path`` has no directory component, ``PACKAGE_ROOT.parent / name``
       (editable install / dev checkout: file next to the package under ``src/``).
    """
    env = os.environ.get("LEROBOT_PLAYGROUND_EXTRINSIC_JSON")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"LEROBOT_PLAYGROUND_EXTRINSIC_JSON is set but not a file: {p}"
            )
        return p

    path = Path(path).expanduser()
    tried: list[Path] = []

    if path.is_absolute():
        tried.append(path)
        if path.is_file():
            return path.resolve()

    cwd_path = (Path.cwd() / path).resolve()
    tried.append(cwd_path)
    if cwd_path.is_file():
        return cwd_path

    if path.parent == Path("."):
        repo_adjacent = (PACKAGE_ROOT.parent / path.name).resolve()
        tried.append(repo_adjacent)
        if repo_adjacent.is_file():
            return repo_adjacent

    msg = "Extrinsic calibration JSON not found. Tried:\n  " + "\n  ".join(str(t) for t in tried)
    msg += (
        "\nSet LEROBOT_PLAYGROUND_EXTRINSIC_JSON to the file path, "
        "or run from a directory that contains it, "
        "or place it as src/extrinsic_calibration.json next to the installed package source."
    )
    raise FileNotFoundError(msg)
