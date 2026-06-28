"""Helpers for multi-agent observation handling."""
import jax
import jax.numpy as jnp


def concat_agent_id_to_obs(obs, agent_id, num_agents):
    """Append one-hot agent ID to a single agent's observation.
    
    Used at deployment when querying the policy one agent at a time.
    
    Args:
        obs: Array of shape (..., obs_dim) for one agent's observation.
        agent_id: Integer index of the agent.
        num_agents: Total number of agents.
    
    Returns:
        Array of shape (..., obs_dim + num_agents).
    """
    one_hot = jax.nn.one_hot(agent_id, num_agents, dtype=obs.dtype)
    one_hot = jnp.broadcast_to(one_hot, (*obs.shape[:-1], num_agents))
    return jnp.concatenate([obs, one_hot], axis=-1)


def batch_concat_agent_id_to_obs(obs):
    """Append one-hot agent ID to all agents' observations in a batch.
    
    Used during training on batched observations from the offline dataset.
    
    Args:
        obs: Array of shape (..., num_agents, obs_dim).
    
    Returns:
        Array of shape (..., num_agents, obs_dim + num_agents).
    """
    num_agents = obs.shape[-2]
    agent_ids = jnp.eye(num_agents, dtype=obs.dtype)  # (K, K)
    while agent_ids.ndim < obs.ndim:
        agent_ids = agent_ids[None]
    agent_ids = jnp.broadcast_to(agent_ids, (*obs.shape[:-1], num_agents))
    return jnp.concatenate([obs, agent_ids], axis=-1)

