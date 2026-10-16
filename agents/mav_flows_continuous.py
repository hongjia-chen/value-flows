"""Continuous-action MAV-Flow agent for OMAR MPE benchmarks.

This keeps the existing discrete SMAC agent untouched.  The actor is shared
across agents, conditions on local observations plus agent IDs, and is executed
decentrally.  Twin per-agent critics are used during centralized training.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.marl_utils import batch_concat_agent_id_to_obs, concat_agent_id_to_obs
from utils.networks import ActorVectorField, Value


class MAVFlowContinuousAgent(flax.struct.PyTreeNode):
    """Flow policy and twin critics for cooperative continuous-action MARL."""

    rng: Any
    network: Any
    agent_names: tuple[str, ...] = nonpytree_field()
    config: Any = nonpytree_field()

    def _sample_action_array(self, observations, rng):
        noises = jax.random.normal(
            rng, (*observations.shape[:-1], self.config['action_dim'])
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        return jnp.clip(actions, -1.0, 1.0)

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        for step in range(self.config['flow_steps']):
            times = jnp.full(
                (*observations.shape[:-1], 1),
                step / self.config['flow_steps'],
            )
            velocities = self.network.select('actor_bc_flow')(
                observations, actions, times, is_encoded=True
            )
            actions = actions + velocities / self.config['flow_steps']
        return jnp.clip(actions, -1.0, 1.0)

    def critic_loss(self, batch, grad_params, rng):
        observations = batch_concat_agent_id_to_obs(batch['observations'])
        next_observations = batch_concat_agent_id_to_obs(batch['next_observations'])
        next_actions = self._sample_action_array(next_observations, rng)

        next_qs = self.network.select('target_critic')(next_observations, next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        rewards = batch['rewards']
        masks = batch['masks']
        if rewards.ndim == 1:
            rewards = rewards[:, None]
        if masks.ndim == 1:
            masks = masks[:, None]
        target_q = rewards + self.config['discount'] * masks * next_q

        qs = self.network.select('critic')(
            observations, batch['actions'], params=grad_params
        )
        critic_loss = jnp.square(qs - target_q[None]).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
            'target_q_mean': target_q.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        observations = batch_concat_agent_id_to_obs(batch['observations'])
        rng, bc_noise_rng, time_rng, distill_rng, metric_rng = jax.random.split(rng, 5)

        noises = jax.random.normal(bc_noise_rng, batch['actions'].shape)
        times = jax.random.uniform(time_rng, (*batch['actions'].shape[:-1], 1))
        noisy_actions = (1.0 - times) * noises + times * batch['actions']
        target_velocity = batch['actions'] - noises
        velocity = self.network.select('actor_bc_flow')(
            observations, noisy_actions, times, params=grad_params
        )
        bc_flow_loss = jnp.square(velocity - target_velocity).mean()

        distill_noises = jax.random.normal(distill_rng, batch['actions'].shape)
        target_actions = jax.lax.stop_gradient(
            self.compute_flow_actions(observations, distill_noises)
        )
        actor_actions = self.network.select('actor_onestep_flow')(
            observations, distill_noises, params=grad_params
        )
        distill_loss = jnp.square(actor_actions - target_actions).mean()

        clipped_actions = jnp.clip(actor_actions, -1.0, 1.0)
        qs = self.network.select('critic')(observations, clipped_actions)
        q = qs.mean(axis=0)
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            scale = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-6))
            q_loss = scale * q_loss

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss
        sampled = self._sample_action_array(observations, metric_rng)
        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q_mean': q.mean(),
            'action_mse': jnp.square(sampled - batch['actions']).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        actor_rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info = {f'critic/{key}': value for key, value in critic_info.items()}
        info.update({f'actor/{key}': value for key, value in actor_info.items()})
        info['total_loss'] = critic_loss + actor_loss
        return critic_loss + actor_loss, info

    def _target_update(self, network):
        target_params = jax.tree.map(
            lambda online, target: (
                self.config['tau'] * online + (1.0 - self.config['tau']) * target
            ),
            self.network.params['modules_critic'],
            self.network.params['modules_target_critic'],
        )
        network.params['modules_target_critic'] = target_params

    @jax.jit
    def update(self, batch):
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn)
        self._target_update(new_network)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        del temperature
        if seed is None:
            seed = self.rng
        if isinstance(observations, dict):
            with_ids = [
                concat_agent_id_to_obs(
                    jnp.asarray(observations[name]), index, self.config['num_agents']
                )
                for index, name in enumerate(self.agent_names)
            ]
            observation_array = jnp.stack(with_ids, axis=0)
            actions = self._sample_action_array(observation_array, seed)
            return {name: actions[index] for index, name in enumerate(self.agent_names)}

        observation_array = batch_concat_agent_id_to_obs(jnp.asarray(observations))
        return self._sample_action_array(observation_array, seed)

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        observations = jnp.asarray(example_batch['observations'])
        actions = jnp.asarray(example_batch['actions'])
        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError(
                'Continuous MAV-Flow expects observations (B, agents, obs_dim) and '
                f'actions (B, agents, action_dim); got {observations.shape} and {actions.shape}.'
            )
        num_agents = observations.shape[-2]
        action_dim = actions.shape[-1]
        observations_with_id = batch_concat_agent_id_to_obs(observations)
        times = actions[..., :1]

        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        critic = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = {
            'critic': (critic, (observations_with_id, actions)),
            'target_critic': (copy.deepcopy(critic), (observations_with_id, actions)),
            'actor_bc_flow': (actor_bc_flow, (observations_with_id, actions, times)),
            'actor_onestep_flow': (actor_onestep_flow, (observations_with_id, actions)),
        }
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (
                encoders['actor_bc_flow'],
                (observations_with_id,),
            )
        network_def = ModuleDict({key: value[0] for key, value in network_info.items()})
        network_args = {key: value[1] for key, value in network_info.items()}
        params = network_def.init(init_rng, **network_args)['params']
        params['modules_target_critic'] = params['modules_critic']
        network = TrainState.create(
            network_def,
            params,
            tx=optax.adam(learning_rate=config['lr']),
        )

        config['ob_dims'] = observations.shape[1:]
        config['action_dim'] = action_dim
        config['num_agents'] = num_agents
        agent_names = tuple(f'agent_{index}' for index in range(num_agents))
        return cls(
            rng=rng,
            network=network,
            agent_names=agent_names,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='mav_flow_continuous',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            num_agents=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_layer_norm=True,
            discount=0.995,
            tau=0.005,
            q_agg='mean',
            alpha=1.0,
            flow_steps=10,
            normalize_q_loss=True,
            encoder=None,
        )
    )
