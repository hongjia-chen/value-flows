"""OMAR MPE vault loading for offline multi-agent continuous control."""

from pathlib import Path

import jax
import numpy as np
from flashbax.vault import Vault

from utils.vault_paths import get_source_vault_dir


DATASET_QUALITIES = ('Expert', 'Medium', 'Medium-Replay', 'Random')


def parse_env_name(env_name: str) -> tuple[str, str]:
    """Parse ``mpe_<scenario>_<quality>`` without changing UID capitalization."""
    if not env_name.startswith('mpe_'):
        raise ValueError(f'Unrecognized MPE environment name: {env_name}')
    payload = env_name.removeprefix('mpe_')
    try:
        scenario, quality = payload.rsplit('_', 1)
    except ValueError as error:
        raise ValueError(
            'MPE names must use mpe_<scenario>_<quality>, for example '
            'mpe_simple_spread_Medium-Replay.'
        ) from error
    if scenario != 'simple_spread':
        raise ValueError(f'Only the OMAR simple_spread scenario is supported, got {scenario!r}.')
    if quality not in DATASET_QUALITIES:
        raise ValueError(
            f'Unknown Simple Spread quality {quality!r}; expected one of {DATASET_QUALITIES}.'
        )
    return scenario, quality


def get_vault_path(scenario: str) -> Path:
    return get_source_vault_dir('omar') / 'mpe' / f'{scenario}.vlt'


def make_env(env_name: str):
    """Environment construction is deferred until main has the experiment seed."""
    parse_env_name(env_name)
    return None


def _flatten_parallel_buffers(value):
    value = np.asarray(value)
    if value.ndim < 2:
        raise ValueError(f'Expected Flashbax (buffers, time, ...) axes, got shape {value.shape}.')
    return value.reshape((value.shape[0] * value.shape[1], *value.shape[2:]))


def _next_values(values: np.ndarray, episode_end: np.ndarray) -> np.ndarray:
    next_values = np.concatenate([values[1:], values[-1:]], axis=0)
    selector = episode_end.reshape((-1,) + (1,) * (values.ndim - 1))
    return np.where(selector, values, next_values)


def experience_to_dataset(experience) -> dict[str, np.ndarray]:
    """Convert an OMAR Flashbax trajectory buffer to transition arrays."""
    experience = jax.tree.map(np.asarray, experience)
    num_buffers, buffer_length = np.asarray(experience['observations']).shape[:2]
    observations = _flatten_parallel_buffers(experience['observations']).astype(np.float32)
    actions = _flatten_parallel_buffers(experience['actions']).astype(np.float32)
    rewards_all = _flatten_parallel_buffers(experience['rewards']).astype(np.float32)
    terminals_all = _flatten_parallel_buffers(experience['terminals']).astype(np.float32)
    truncations_all = _flatten_parallel_buffers(experience['truncations']).astype(np.float32)

    # Simple Spread is cooperative, so the per-agent reward/end flags are replicas.
    rewards = rewards_all[:, 0] if rewards_all.ndim > 1 else rewards_all
    terminated = terminals_all[:, 0] if terminals_all.ndim > 1 else terminals_all
    truncated = truncations_all[:, 0] if truncations_all.ndim > 1 else truncations_all
    episode_end = (terminated + truncated) > 0
    # Never build a next transition from the end of one parallel buffer to the
    # beginning of another, even if a source vault omitted the final end flag.
    buffer_ends = np.arange(buffer_length - 1, num_buffers * buffer_length, buffer_length)
    episode_end[buffer_ends] = True

    dataset = dict(
        observations=observations,
        next_observations=_next_values(observations, episode_end),
        actions=np.clip(actions, -1.0, 1.0),
        rewards=rewards,
        terminals=episode_end.astype(np.float32),
        truncations=truncated,
        masks=(1.0 - terminated).astype(np.float32),
    )

    state = experience.get('infos', {}).get('state')
    if state is not None:
        states = _flatten_parallel_buffers(state).astype(np.float32)
        dataset['states'] = states
        dataset['next_states'] = _next_values(states, episode_end)
    return dataset


def get_dataset(env, env_name: str) -> dict[str, np.ndarray]:
    """Load one quality UID from the OMAR Simple Spread vault."""
    del env
    scenario, quality = parse_env_name(env_name)
    vault_path = get_vault_path(scenario)
    if not vault_path.is_dir():
        raise FileNotFoundError(
            f'Missing OMAR vault {vault_path}. Download it with '
            "download_and_unzip_vault('omar', 'mpe', 'simple_spread', dataset_base_dir=...)."
        )
    experience = Vault(str(vault_path), vault_uid=quality).read().experience
    return experience_to_dataset(experience)
