
import pytest

from pytti.config.structured_config import CONFIG_VERSION
from pytti.warmup import ensure_configs_exist, migrate_local_config


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_fresh_config_is_current_version(workdir):
    ensure_configs_exist()
    text = (workdir / "config" / "default.yaml").read_text()
    assert f"config_version: {CONFIG_VERSION}" in text
    migrate_local_config()  # no-op on a fresh config
    assert not list((workdir / "config").glob("*.bak"))


def test_v1_config_is_migrated_with_backup(workdir):
    conf = workdir / "config" / "conf"
    conf.mkdir(parents=True)
    # a v1 default.yaml: no config_version, carries removed keys
    (workdir / "config" / "default.yaml").write_text(
        "scenes: ''\nshow_palette: false\nuse_tensorboard: false\n"
    )
    (conf / "mypreset.yaml").write_text(
        "# @package _global_\nscenes: a scene\nshow_palette: true\n"
        "clear_every: 0\nwidth: 256\n"
    )

    migrate_local_config()

    new_default = (workdir / "config" / "default.yaml").read_text()
    assert f"config_version: {CONFIG_VERSION}" in new_default
    assert "show_palette" not in new_default
    assert (workdir / "config" / "default.yaml.v1.bak").exists()

    preset = (conf / "mypreset.yaml").read_text()
    assert "# @package _global_" in preset  # header survives
    assert "width: 256" in preset  # real settings survive
    assert "show_palette" not in preset
    assert "clear_every" not in preset


def test_migration_is_idempotent(workdir):
    ensure_configs_exist()
    before = (workdir / "config" / "default.yaml").read_text()
    migrate_local_config()
    migrate_local_config()
    assert (workdir / "config" / "default.yaml").read_text() == before


def test_no_config_dir_is_a_noop(workdir):
    migrate_local_config()
    assert not (workdir / "config").exists()
