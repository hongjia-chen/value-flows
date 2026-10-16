"""Eval for harness envs (COGNAC / discrete MPE), mirroring utils/evaluate_smac.py.

Same agent call as SMAC: per-agent obs + one-hot id -> sample_actions_rejection.
Differences from evaluate_smac, both deliberate:
  * Episode i is ALWAYS reset with seed episode_seed_base + i, so every checkpoint and
    every arm sees identical initial conditions and env noise (paired by construction).
    `seed` only drives the agent's own sampling key, as in evaluate_smac.
  * Return = sum over steps of the harness team reward (info['team_reward']), the exact
    quantity the dataset stores; it is asserted (to float tolerance) equal to
    mean(rewards.values()), the SMAC-convention readout.
"""

import jax
import numpy as np
from tqdm import trange

from envs.marl_harness import EVAL_SEED_BASE
from utils.marl_utils import batch_concat_agent_id_to_obs


def evaluate_marl(agent, env, num_episodes=50, seed=0, episode_seed_base=EVAL_SEED_BASE,
                  verbose=False):
    rng = jax.random.PRNGKey(seed)
    returns, lengths, successes, seeds = [], [], [], []
    iterator = trange(num_episodes, desc='MARL eval', dynamic_ncols=True) if verbose else range(num_episodes)

    for ep in iterator:
        episode_seed = episode_seed_base + ep
        obs_dict, info = env.reset(seed=episode_seed)
        agent_names = list(env.agents)
        episode_return, episode_len = 0.0, 0

        while True:
            obs_arr = np.stack([obs_dict[ag] for ag in agent_names], axis=0)[None]
            legals_arr = np.stack([info['legals'][ag] for ag in agent_names], axis=0)[None]
            obs_with_id = batch_concat_agent_id_to_obs(obs_arr)

            rng, act_rng = jax.random.split(rng)
            actions_np = np.asarray(agent.sample_actions_rejection(obs_with_id, legals_arr, act_rng)[0])
            for i, ag in enumerate(agent_names):
                if info['legals'][ag][actions_np[i]] == 0:
                    raise RuntimeError(f'Agent sampled illegal action {actions_np[i]} for {ag}.')

            obs_dict, rewards, terminals, truncations, info = env.step(
                {ag: int(actions_np[i]) for i, ag in enumerate(agent_names)})
            team = float(info['team_reward'])
            assert np.isclose(np.mean(list(rewards.values())), team), (rewards, team)
            episode_return += team
            episode_len += 1
            if all(terminals.values()) or all(truncations.values()):
                break

        returns.append(episode_return)
        lengths.append(episode_len)
        successes.append(info.get('success'))
        seeds.append(episode_seed)
        if verbose:
            iterator.set_postfix({'ret': f'{episode_return:.2f}', 'mean': f'{np.mean(returns):.2f}'})

    has_success = all(s is not None for s in successes)
    return {
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'max_return': float(np.max(returns)),
        'min_return': float(np.min(returns)),
        'success_rate': float(np.mean(successes)) if has_success else None,
        'mean_length': float(np.mean(lengths)),
        'raw_returns': returns,
        'raw_success': successes,
        'episode_seeds': seeds,
    }
