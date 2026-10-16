"""Path helpers for Flashbax vaults from multiple MARL dataset sources."""

import os
from pathlib import Path


def get_vault_root() -> Path:
    """Return the common directory containing source-specific vault folders."""
    return Path(os.environ.get('MARL_VAULT_ROOT', '~/vaults')).expanduser()


def get_source_vault_dir(source: str) -> Path:
    """Return the data directory for a source, preserving the old SMAC override."""
    if source == 'og_marl' and 'OG_MARL_DATA_DIR' in os.environ:
        return Path(os.environ['OG_MARL_DATA_DIR']).expanduser()
    return get_vault_root() / source
