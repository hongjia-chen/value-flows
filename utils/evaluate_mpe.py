"""Evaluation loop for dictionary-based OMAR MPE environments."""

import jax
import numpy as np


def evaluate_mpe(agent, env, num_episodes=10, seed=0):
    returns = []
    lengths = []
    rng = jax.random.PRNGKey(seed)

    for _ in range(num_episodes):
        observations, _ = env.reset()
        episode_return = 0.0
        episode_length = 0

        while True:
            rng, action_rng = jax.random.split(rng)
            actions = agent.sample_actions(observations, seed=action_rng)
            actions = {name: np.asarray(action) for name, action in actions.items()}
            observations, rewards, terminals, truncations, _ = env.step(actions)
            episode_return += float(np.mean(list(rewards.values())))
            episode_length += 1
            if all(terminals.values()) or all(truncations.values()):
                break

        returns.append(episode_return)
        lengths.append(episode_length)

    returns = np.asarray(returns, dtype=np.float32)
    return {
        'mean_return': float(returns.mean()),
        'std_return': float(returns.std()),
        'max_return': float(returns.max()),
        'min_return': float(returns.min()),
        'mean_length': float(np.mean(lengths)),
    }
