"""SMAC eval via OG-MARL's gym-style API.

Mirrors MAC-Flow's _evaluate() function but adapted for MAV-Flow's tensor-based
sample_actions. Works for both SMACv1 and SMACv2 — OG-MARL abstracts the difference.

Reports mean episodic return, matching MAC-Flow's Table 1 convention.
"""

import numpy as np
import jax
from tqdm import trange

from utils.marl_utils import batch_concat_agent_id_to_obs


def _extract_smac_win(info):
    """Return SMAC's explicit terminal win flag, including nested wrappers."""
    if not isinstance(info, dict):
        return None
    for key in ('battle_won', 'won', 'win'):
        if key in info:
            value = np.asarray(info[key])
            if value.size == 1:
                return bool(value.item())
            return bool(value.all())
    for value in info.values():
        result = _extract_smac_win(value)
        if result is not None:
            return result
    return None


def _get_smac_battles_won(env):
    """Read SMAC's cumulative win counter through OG-MARL wrappers."""
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, 'battles_won'):
            return int(current.battles_won)
        get_stats = getattr(current, 'get_stats', None)
        if callable(get_stats):
            try:
                stats = get_stats()
            except (AttributeError, KeyError, TypeError, ZeroDivisionError):
                stats = None
            if isinstance(stats, dict) and 'battles_won' in stats:
                return int(stats['battles_won'])
        current = getattr(current, '_environment', None)
    return None


def evaluate_smac(agent, env, num_episodes=50, seed=0, verbose=False):
    """Run num_episodes rollouts. Returns mean episode return.

    Args:
        agent: trained MAVFlowAgent (with .sample_actions).
        env: OG-MARL env from get_environment(...).
        num_episodes: number of full rollouts.
        seed: RNG seed.
        verbose: tqdm progress bar.

    Returns:
        dict with return statistics, win_rate, mean_length, raw_returns,
        and raw_wins.
    """
    rng = jax.random.PRNGKey(seed)
    returns, lengths, wins = [], [], []
    iterator = trange(num_episodes, desc='SMAC eval', dynamic_ncols=True) if verbose else range(num_episodes)

    for _ in iterator:
        wins_before = _get_smac_battles_won(env)
        obs_dict, info = env.reset()
        agent_names = list(obs_dict.keys())   # e.g. ['agent_0', 'agent_1', 'agent_2']

        episode_return = 0.0
        episode_len = 0

        while True:
            # Dict → Tensor: stack per-agent dict → (1, K, obs_dim)
            obs_arr = np.stack([obs_dict[ag] for ag in agent_names], axis=0)[None]
            legals_arr = np.stack(
                [info['legals'][ag] for ag in agent_names], axis=0)[None]

            obs_with_id = batch_concat_agent_id_to_obs(obs_arr)

            # Sample actions
            rng, act_rng = jax.random.split(rng)
            actions_tensor = agent.sample_actions_rejection(obs_with_id, legals_arr, act_rng)
            actions_np = np.asarray(actions_tensor[0])

            # Defensive: ensure every action is legal
            for i, ag in enumerate(agent_names):
                if info['legals'][ag][actions_np[i]] == 0:
                    actions_np[i] = int(np.argmax(info['legals'][ag]))

            # Tensor → Dict
            actions_dict = {ag: int(actions_np[i]) for i, ag in enumerate(agent_names)}

            obs_dict, rewards, terminals, truncations, info = env.step(actions_dict)

            episode_return += float(np.mean(list(rewards.values())))
            episode_len += 1

            done = all(terminals.values()) or all(truncations.values())
            if done:
                break

        returns.append(episode_return)
        episode_won = _extract_smac_win(info)
        if episode_won is None:
            wins_after = _get_smac_battles_won(env)
            if wins_before is None or wins_after is None:
                raise KeyError(
                    'SMAC exposed neither a terminal win flag nor a cumulative '
                    'battles_won statistic. '
                    f'Available info keys: '
                    f'{tuple(info.keys()) if isinstance(info, dict) else type(info)}')
            episode_won = wins_after > wins_before
        wins.append(episode_won)
        lengths.append(episode_len)

        if verbose:
            iterator.set_postfix({
                'ret': f'{episode_return:.2f}',
                'len': episode_len,
                'mean': f'{np.mean(returns):.2f}',
                'win': int(episode_won),
            })

    return {
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'max_return': float(np.max(returns)),
        'min_return': float(np.min(returns)),
        'win_rate': float(np.mean(wins)),
        'mean_length': float(np.mean(lengths)),
        'raw_returns': returns,
        'raw_wins': wins,
    }
