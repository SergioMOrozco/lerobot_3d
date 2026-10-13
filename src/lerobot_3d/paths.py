"""Filesystem locations for bundled calibration assets (URDF, meshes)."""
from __future__ import annotations

import os
from pathlib import Path

# ``lerobot_3d`` package root (directory containing ``calibration/``).
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
CALIBRATION_DIR: Path = PACKAGE_ROOT / "calibration"
# I2RT YAM v1 / linear_4310 station URDF, vendored from i2rt-robotics/i2rt@120c3c8 (MIT).
YAM_URDF_PATH: Path = CALIBRATION_DIR / "yam" / "yam_station_linear_4310_d405.urdf"


def _resolve_repo_file(path: str | Path, *, env_var: str, not_found_msg: str) -> Path:
    """Resolve a user-editable file that lives next to the package (dev checkout) or cwd.

    Order:

    1. ``env_var`` if set (file must exist).
    2. ``path`` if absolute and exists.
    3. ``Path.cwd() / path`` if it exists.
    4. If ``path`` has no directory component, ``PACKAGE_ROOT.parent / name``
       (editable install / dev checkout: file next to the package under ``src/``).
    """
    env = os.environ.get(env_var)
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"{env_var} is set but not a file: {p}")
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

    msg = f"{not_found_msg} Tried:\n  " + "\n  ".join(str(t) for t in tried)
    msg += (
        f"\nSet {env_var} to the file path, "
        "or run from a directory that contains it, "
        f"or place it as src/{path.name} next to the installed package source."
    )
    raise FileNotFoundError(msg)


def resolve_extrinsic_calibration_json(path: str | Path) -> Path:
    """Resolve ``extrinsic_calibration.json`` (or any extrinsic JSON path).

    See :func:`_resolve_repo_file` for the search order.
    """
    return _resolve_repo_file(
        path,
        env_var="LEROBOT_3D_EXTRINSIC_JSON",
        not_found_msg="Extrinsic calibration JSON not found.",
    )


def resolve_teleop_config_yaml(path: str | Path) -> Path:
    """Resolve ``teleop_config.yaml`` (the full :class:`TeleopSystemConfig`, as YAML).

    See :func:`_resolve_repo_file` for the search order.
    """
    return _resolve_repo_file(
        path,
        env_var="LEROBOT_3D_TELEOP_CONFIG",
        not_found_msg="Teleop config YAML not found.",
    )
