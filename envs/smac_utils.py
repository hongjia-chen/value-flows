# This script is used to load the SMAC benchmark dataset
# Then turn into (env, eval_env, train_dataset, val_dataset) tuple for offline RL training and evaluation


import numpy as np

from utils.download_vault import download_and_unzip_vault
from utils.datasets import Dataset

# path = download_and_unzip_vault(
#     dataset_source="og_marl",
#     env_name="smac_v1",
#     scenario_name="3m",
#     dataset_base_dir="/Users/hc33998/Projects/data",   # any path you want
# )
# print(path)

import os
import subprocess
import sys
import jax
from flashbax.vault import Vault
from utils.vault_paths import get_source_vault_dir

## The datatype is a TrajectoryBufferState
## .experience calls the actual data dictionary
## vault.read()

# print(jax.tree.map(lambda x: x.shape, experience))
# print(jax.tree.map(lambda x: x.dtype, experience))

# This is the main data structure of the dataset
# The leading (1, ...) is flashbax's parallel buffer dimension
# Then followed by number of transitions.

# {'actions': (1, 996366, 3), - 3 marines
#   'infos': {'legals': (1, 996366, 3, 9), - 3 marines, binary encoding on 9 possible actions, indicating which ones are legal at the moment
#   'state': (1, 996366, 48)}, - 48-dim full game state (centralized, global state)
#   'observations': (1, 996366, 3, 30),  - 3 marines, each with 30-dim local observation 
#   'rewards': (1, 996366, 3), - Team reward, replicated for each agent
#   'terminals': (1, 996366, 3), - Replicated terminal state for all agents
#   'truncations': (1, 996366, 3)} - Replicated time-limit signal for all agents 

# {'actions': dtype('int32'), 
#   'infos': {'legals': dtype('float32'), 
#               'state': dtype('float32')}, 
#   'observations': dtype('float32'), 
#   'rewards': dtype('float32'), 
#   'terminals': dtype('float32'), 
# 'truncations': dtype('float32')}

def make_env(env_name):
    return None


_OBS_PROBE = (
    'import sys\n'
    'from og_marl.environments import get_environment\n'
    'size = get_environment(sys.argv[1], sys.argv[2], sys.argv[3], seed=0)._environment.get_obs_size()\n'
    'print("OBS_DIM", size[0] if isinstance(size, (list, tuple)) else size)\n')


def _env_obs_dim(env):
    # The OMIGA env returns a list whose [0] is the total; vanilla SMAC returns an int.
    size = env._environment.get_obs_size()
    return int(size[0] if isinstance(size, (list, tuple)) else size)


def _probe_obs_dim(source, map_source, scenario):
    out = subprocess.run([sys.executable, '-c', _OBS_PROBE, source, map_source, scenario],
                         capture_output=True, text=True, check=True).stdout
    return int(out.rsplit('OBS_DIM', 1)[1].split()[0])


def make_smac_eval_env(map_source, scenario, obs_dim, source='auto', seed=0):
    """(eval_env, source, env_obs_dim) for a vault with `obs_dim`-dim observations.

    source='auto' picks the og_marl env, else (SMACv1 only) the OMIGA env, whichever
    emits obs_dim; with no match it falls back to og_marl and the caller's parity check
    fails. The candidates are probed in SUBPROCESSES and only the chosen env is built
    here: smac's and the OMIGA env's map registries both define '3m', '8m', ..., and
    with both imported pysc2 raises DuplicateMapError at the first SC2 launch.
    Never reset() the env here (see main.py).
    """
    from og_marl.environments import get_environment
    if source == 'auto':
        candidates = ('og_marl', 'omiga') if map_source == 'smac_v1' else ('og_marl',)
        source = next((s for s in candidates
                       if _probe_obs_dim(s, map_source, scenario) == obs_dim), candidates[0])
    eval_env = get_environment(source, map_source, scenario, seed=seed)
    return eval_env, source, _env_obs_dim(eval_env)


def resolve_quality(vault_dir, quality):
    """The dataset uid in `vault_dir` matching `quality` case-insensitively.

    og_marl and OMIGA name them Good/Medium/Poor; CFCQL uses Expert, Medium,
    Medium-Replay and Mixed, which str.capitalize() would mangle ('Medium-replay').
    """
    uids = sorted(d for d in os.listdir(vault_dir) if os.path.isdir(os.path.join(vault_dir, d)))
    for uid in uids:
        if uid.lower() == quality.lower():
            return uid
    raise FileNotFoundError(f'no dataset {quality!r} in {vault_dir}; available: {uids}')


def episode_flags(terminals, truncations):
    """Per-step (terminal, truncation, notes) from a vault's (T, K) per-agent flags.

    The vaults under smac_v1/ come from three recorders:
      og_marl core (3m, 8m, 2s3z, 5m_vs_6m, 3s5z_vs_3s6z): one episode-level flag,
        replicated across agents; truncations are all zero, so a time-out is recorded
        as a terminal.
      OMIGA (2c_vs_64zg, 6h_vs_8z, corridor): per-agent DONE flags. An agent's flag goes
        to 1 on the step it dies and stays 1, so agent 0's column alone marks every step
        after agent 0's death as an episode end (2c_vs_64zg Medium: 2380 "episodes" from
        1001 real ones). The episode ends where EVERY agent is done.
      CFCQL (3s_vs_5z): replicated flags, but truncations is a copy of terminals.
    OMIGA and CFCQL both copy terminals into truncations, so truncations carries no
    time-limit information there; it is dropped, which makes time-outs terminals, the
    same convention the og_marl core vaults have. On core vaults this function returns
    exactly agent 0's columns, i.e. what the loader read before 10/17.
    """
    notes = []
    if (terminals != terminals[:, :1]).any():
        term = terminals.all(axis=1).astype(np.float32)
        trunc = truncations.all(axis=1).astype(np.float32)
        notes.append('per-agent done flags -> episode end = all agents done')
    else:
        term = terminals[:, 0].astype(np.float32)
        trunc = truncations[:, 0].astype(np.float32)
    if term.any() and np.array_equal(term, trunc):
        trunc = np.zeros_like(trunc)
        notes.append('truncations copied terminals -> dropped (time-outs are terminals)')
    return term, trunc, notes

def get_dataset(env, env_name):
    ## need to encode data quality in env_name
    # FIX: data path when submitted to cluster


    """Make SMAC dataset.

    Args:
        env: The environment.
        env_name: Name of the environment.
    """
    del env

    import os
    parts = env_name.split("_")
    quality = parts[-1].capitalize()

    if parts[0] == 'smac':
        env_kind = 'smac_v1'
        map_name = "_".join(parts[1:-1])
    elif parts[0] == 'smacv2':
        env_kind = 'smac_v2'
        map_name = "_".join(parts[1:-1])
    else:
        raise ValueError(f"Unrecognized env_name format: {env_name}")

    data_dir = get_source_vault_dir('og_marl')
    vault_dir = data_dir / env_kind / f'{map_name}.vlt'
    uid = resolve_quality(vault_dir, quality)
    vault = Vault(str(vault_dir), vault_uid=uid)

    experience = vault.read().experience
    experience = jax.tree.map(np.asarray, experience)

    # WAS (pre-10/17): agent 0's columns, which on the OMIGA vaults are agent 0's
    # death flags rather than episode ends (see episode_flags) --
    #   term_raw = experience['terminals'][0, :, 0]
    #   trunc_raw = experience['truncations'][0, :, 0].astype(np.float32)
    term_raw, trunc_raw, flag_notes = episode_flags(  # episode end / bootstrap-from-next flags
        experience['terminals'][0], experience['truncations'][0])
    rewards = experience['rewards'][0, :, 0].astype(np.float32)

    ## Any episode end (terminal or time-limit)
    terminals = ((term_raw + trunc_raw) > 0).astype(np.float32)
    ## Only record the real episode ends
    masks = (1.0 - term_raw).astype(np.float32)

    print(f"[MASK CHECK] term*trunc overlap = {(term_raw * trunc_raw).sum():.0f}  "
          f"(want 0)   n_term={term_raw.sum():.0f}  n_trunc={trunc_raw.sum():.0f}")
    # No time-limit count here: it needs smac's map registry, and importing that next to
    # the OMIGA env's registry makes pysc2 raise DuplicateMapError at the first SC2
    # launch. scripts/prepare_smac_v1.py reports it.
    ends = np.flatnonzero(terminals > 0)
    lens = np.diff(np.concatenate([[-1], ends])) if len(ends) else np.zeros(1, int)
    print(f"[EPISODE CHECK] {vault_dir.name}/{uid}: episodes={len(ends)} "
          f"len mean={lens.mean():.1f} max={lens.max()}"
          + ''.join(f'\n                {n}' for n in flag_notes))

    ## Need to create next observations
    obs = experience['observations'][0].astype(np.float32)
    next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
    
    is_episode_end = terminals > 0 
    next_obs = np.where(is_episode_end[:, None, None], obs, next_obs)


    states = experience['infos']['state'][0].astype(np.float32)           # (T, 48)
    next_states = np.concatenate([states[1:], states[-1:]], axis=0)
    next_states = np.where(is_episode_end[:, None], states, next_states)    # (T, 1) broadcast for 2D

    legals = experience['infos']['legals'][0].astype(np.float32)         # (T, 3, 9)
    next_legals = np.concatenate([legals[1:], legals[-1:]], axis=0)
    next_legals = np.where(is_episode_end[:, None, None], legals, next_legals)

    ## vs. Dataset.create() here?

    return dict(
        actions=experience['actions'][0].astype(np.int32),
        legals = legals,
        next_legals = next_legals,
        states = states,
        next_states = next_states,
        observations= obs,
        next_observations= next_obs,
        rewards=rewards,
        terminals=terminals,
        truncations=trunc_raw,
        masks = masks,
    )


# {'actions': (1, 996366, 3), - 3 marines
#   'infos': {'legals': (1, 996366, 3, 9), - 3 marines, binary encoding on 9 possible actions, indicating which ones are legal at the moment
#   'state': (1, 996366, 48)}, - 48-dim full game state (centralized, global state)
#   'observations': (1, 996366, 3, 30),  - 3 marines, each with 30-dim local observation 
#   'rewards': (1, 996366, 3), - Team reward, replicated for each agent
#   'terminals': (1, 996366, 3), - Replicated terminal state for all agents
#   'truncations': (1, 996366, 3)} - Replicated time-limit signal for all agents 
