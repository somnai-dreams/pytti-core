"""
Explicit startup helpers for the CLI entry point.

Nothing in here runs at import time: the CLI calls `register_resolvers()` and
`ensure_configs_exist()` before Hydra composes the config.
"""

import re
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

from pytti.config.structured_config import CONFIG_VERSION

_ASSETS_DIR = Path(__file__).parent / "assets"

# config keys that existed in v1 local configs but no longer exist in the
# schema; composing a config that still carries one fails loudly, so
# migration strips them from presets
_REMOVED_KEYS = (
    "show_palette",
    "use_tensorboard",
    "use_mmc",
    "mmc_models",
    "show_graphs",
    "clear_every",
    "display_scale",
)

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


def migrate_local_config():
    """
    Upgrade an existing ./config directory to the current CONFIG_VERSION.

    The v1 layout was a copy-once-never-update snapshot, so new keys never
    reached existing users and removed keys lingered until they crashed
    composition. v1 -> v2: replace default.yaml with the current shipped one
    (the old file is kept as a .bak) and strip removed keys from presets.
    """
    default_path = local_config_dir() / "default.yaml"
    if not default_path.exists():
        return
    data = OmegaConf.load(default_path)
    version = data.get("config_version", 1)
    if version >= CONFIG_VERSION:
        return

    backup = default_path.with_suffix(f".yaml.v{version}.bak")
    default_path.rename(backup)
    default_path.write_text(
        (_ASSETS_DIR / "default.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    logger.warning(
        f"Migrated ./config/default.yaml from v{version} to v{CONFIG_VERSION}; "
        f"your old file is at {backup.name}. Settings now live in presets and "
        "the schema — re-apply any edits you had made to default.yaml."
    )

    # strip removed keys from presets textually, preserving comments and the
    # `# @package _global_` header that OmegaConf.save would drop
    pattern = re.compile(rf"^({'|'.join(_REMOVED_KEYS)}):.*\n", re.MULTILINE)
    for preset in sorted((local_config_dir() / "conf").glob("*.yaml")):
        text = preset.read_text(encoding="utf-8")
        cleaned, n = pattern.subn("", text)
        if n:
            preset.write_text(cleaned, encoding="utf-8")
            logger.warning(
                f"Removed {n} obsolete setting(s) from preset {preset.name}"
            )


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
