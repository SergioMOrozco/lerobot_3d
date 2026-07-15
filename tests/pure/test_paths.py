import pytest

from lerobot_3d import paths
from lerobot_3d.paths import (
    _resolve_repo_file,
    resolve_extrinsic_calibration_json,
    resolve_teleop_config_yaml,
)

pytestmark = pytest.mark.pure


def test_env_var_valid_file(tmp_path, monkeypatch):
    f = tmp_path / "cfg.yaml"
    f.write_text("x")
    monkeypatch.setenv("MY_ENV", str(f))

    result = _resolve_repo_file("ignored.yaml", env_var="MY_ENV", not_found_msg="not found")

    assert result == f.resolve()


def test_env_var_set_but_not_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_ENV", str(tmp_path / "missing.yaml"))

    with pytest.raises(FileNotFoundError, match="MY_ENV"):
        _resolve_repo_file("ignored.yaml", env_var="MY_ENV", not_found_msg="not found")


def test_absolute_path_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("MY_ENV", raising=False)
    f = tmp_path / "cfg.yaml"
    f.write_text("x")

    result = _resolve_repo_file(f, env_var="MY_ENV", not_found_msg="not found")

    assert result == f.resolve()


def test_cwd_relative_match(tmp_path, monkeypatch):
    monkeypatch.delenv("MY_ENV", raising=False)
    monkeypatch.chdir(tmp_path)
    subdir = tmp_path / "sub"
    subdir.mkdir()
    f = subdir / "cfg.yaml"
    f.write_text("x")

    result = _resolve_repo_file("sub/cfg.yaml", env_var="MY_ENV", not_found_msg="not found")

    assert result == f.resolve()


def test_package_adjacent_dev_checkout_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("MY_ENV", raising=False)
    empty_cwd = tmp_path / "somewhere_else"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)

    fake_pkg_root = tmp_path / "fake_src" / "lerobot_3d"
    fake_pkg_root.mkdir(parents=True)
    monkeypatch.setattr(paths, "PACKAGE_ROOT", fake_pkg_root)

    f = tmp_path / "fake_src" / "cfg.yaml"
    f.write_text("x")

    result = _resolve_repo_file("cfg.yaml", env_var="MY_ENV", not_found_msg="not found")

    assert result == f.resolve()


def test_nothing_found_lists_tried_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("MY_ENV", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        _resolve_repo_file("does_not_exist.yaml", env_var="MY_ENV", not_found_msg="Custom not found msg.")

    msg = str(exc_info.value)
    assert "Custom not found msg." in msg
    assert "MY_ENV" in msg


def test_resolve_extrinsic_calibration_json_uses_its_own_env_var(tmp_path, monkeypatch):
    f = tmp_path / "extrinsics.json"
    f.write_text("{}")
    monkeypatch.setenv("LEROBOT_3D_EXTRINSIC_JSON", str(f))

    assert resolve_extrinsic_calibration_json("ignored.json") == f.resolve()


def test_resolve_teleop_config_yaml_uses_its_own_env_var(tmp_path, monkeypatch):
    f = tmp_path / "teleop_config.yaml"
    f.write_text("{}")
    monkeypatch.setenv("LEROBOT_3D_TELEOP_CONFIG", str(f))

    assert resolve_teleop_config_yaml("ignored.yaml") == f.resolve()
