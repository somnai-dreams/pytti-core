"""
Explicit startup helpers for the CLI entry point.

Nothing in here runs at import time: the CLI calls `register_resolvers()` and
`ensure_configs_exist()` before Hydra composes the config.
"""

from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

_ASSETS_DIR = Path(__file__).parent / "assets"

# This must match the path used by PyttiLocalConfigSearchPathPlugin
# (hydra_plugins/pytti_local_config_searchpath_plugin).
def local_config_dir() -> Path:
    return Path.cwd() / "config"


def ensure_configs_exist():
    """
    If ./config doesn't exist, create it with the shipped default and demo
    configs so `pytti` can run from any directory.
    """
    local_path = local_config_dir()
    conf_dir = local_path / "conf"
    if local_path.exists():
        logger.debug("Local config directory detected.")
        return
    logger.info("Local config directory not detected.")
    logger.info("Creating local config directory with default and demo configs")
    conf_dir.mkdir(parents=True, exist_ok=True)
    (local_path / "default.yaml").write_text(
        (_ASSETS_DIR / "default.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (conf_dir / "demo.yaml").write_text(
        (_ASSETS_DIR / "demo.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (conf_dir / "_empty.yaml").write_text("\n", encoding="utf-8")


def register_resolvers():
    OmegaConf.register_new_resolver(
        "user_cache",
        lambda: str((Path.home() / ".cache").resolve()),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "path_join",
        lambda a, b: str((Path(a) / Path(b)).resolve()),
        replace=True,
    )
