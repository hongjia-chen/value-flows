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

from flashbax.vault import Vault
import jax

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

def get_dataset(env, env_name):
    ## need to encode data quality in env_name
    # FIX: data path when submitted to cluster


    """Make SMAC dataset.

    Args:
        env: The environment.
        env_name: Name of the environment.
    """
    del env

    parts = env_name.split("_")
    assert parts[0] == 'smac', f"Expected smac_ prefix, got {env_name}"
    quality = parts[-1].capitalize()
    map_name = "_".join(parts[1:-1])

    # Load the dataset from the vault
    vault = Vault(f"/Users/hc33998/Projects/data/og_marl/smac_v1/{map_name}.vlt", vault_uid = quality)
    experience = vault.read().experience
    experience = jax.tree.map(np.asarray, experience)

    term_raw = experience['terminals'][0, :, 0]  # Indicate the end of an episode.
    trunc_raw = experience['truncations'][0, :, 0]  # Indicate whether we should bootstrap from the next state.
    rewards = experience['rewards'][0, :, 0].astype(np.float32)
    masks = np.zeros_like(experience['rewards'][0, :, 0])  

    ## Any episode end (terminal or time-limit)
    terminals = ((term_raw + trunc_raw) > 0).astype(np.float32)
    ## Only record the real episode ends
    masks = (1.0 - term_raw).astype(np.float32)


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

