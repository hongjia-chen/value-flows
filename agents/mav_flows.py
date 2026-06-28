import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ValueVectorField, ActorVectorField, Value
from utils.marl_utils import batch_concat_agent_id_to_obs, concat_agent_id_to_obs

## Design Choices
# 1. Critic Architecture: 
#       k vector fields that transports \eps to Z_i, per-agent return distribution
#       Mixer: Sum-mixer (VDN-style): Q_tot(s,a) = sum_i^K Q_i; where Q_i is E[Z_i] for each agent
#       Actor loss is updated with 
#       Distribtional mixing (DFAC-style) is what we want to eventually implement
#           CVaR-style, or some other metric that utilizes the full distributional information
#       6/26 - Train joint distributional critic instead -> Then factorize to individual
#           1. Train Z_joint(s,a) -> Calculate V_joint with by taking E ()
#           2. Distill the flow critic into student flows that follow VDN-shaped architecture
# 2. Actor Architecture: CTDE of MAC Flow
#       Training centralized BC flow on the joint action distribution
#       Distill into per-agent one-step policies for decentrailized execution
# 3. Action Discretization and Representation
#       One-hot encoding of discrete actions for SMAC
#       Distilled BC Flow transports \eps to logits in R^9
#       Argmax(logits) to get discrete actions
#       Legal-action masking: -inf on illegal logits
# 3. Q-guidance: Softmax (relaxed) to allow Q-loss gradients to flow to actors
# 4. Confidence weighting


class MAVFlowAgent(flax.struct.PyTreeNode):
    """MAV-Flow agent: distributional flow critic for cooperative MARL."""
    
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def joint_critic_bcfm_loss(self, batch, grad_params, rng):

        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']
        rng, action_rng, target_noise_rng, bcfm_noise_rng, time_rng = jax.random.split(rng, 5)
        
        # target_noise_rng -> ε for target critic single-step trick
        # bcfm_noise_rng   -> ε for BCFM noise sample
        # time_rng         -> t for flow interpolation


        states = batch['states']
        next_states = batch['next_states']
        actions_oh = jax.nn.one_hot(batch['actions'], A)
        joint_actions = actions_oh.reshape(batch_size, K * A)

        rng, action_rng = jax.random.split(rng)
        next_obs_with_id = batch_concat_agent_id_to_obs(batch['next_observations'])
        next_actions_int = self.sample_actions(
            next_obs_with_id, batch['next_legals'], action_rng)
        next_actions_oh = jax.nn.one_hot(next_actions_int, A)
        next_joint_actions = next_actions_oh.reshape(batch_size, K * A)

        # Return: use next-state expected Z 
        next_q_noises = jax.random.normal(target_noise_rng, (batch_size, 1))
        next_z1 = next_q_noises + self.network.select('target_joint_critic_flow1')(
            next_q_noises, jnp.zeros_like(next_q_noises), next_states, next_joint_actions)
        next_z2 = next_q_noises + self.network.select('target_joint_critic_flow2')(
            next_q_noises, jnp.zeros_like(next_q_noises), next_states, next_joint_actions)

        if self.config['ret_agg'] == 'min':
            next_returns = jnp.minimum(next_z1, next_z2)
        else:
            next_returns = (next_z1 + next_z2) / 2

        # Bellman target: G = r + γ * mask * E[Z_tot(s', a')]
        returns = (jnp.expand_dims(batch['rewards'], axis=-1)
                + self.config['discount']
                * jnp.expand_dims(batch['masks'], axis=-1)
                * next_returns)                      

        # BCFM regularization loss
        noises = jax.random.normal(bcfm_noise_rng, (batch_size, 1))   # eps
        times = jax.random.uniform(time_rng, (batch_size, 1))   # t in [0, 1]
        noisy_returns = times * returns + (1 - times) * noises  # G_t
        target_vector_field = returns - noises                  

        vf1 = self.network.select('joint_critic_flow1')(
            noisy_returns, times, states, joint_actions, params=grad_params)
        vf2 = self.network.select('joint_critic_flow2')(
            noisy_returns, times, states, joint_actions, params=grad_params)

        bcfm_loss = ((vf1 - target_vector_field) ** 2
                    + (vf2 - target_vector_field) ** 2).mean()
        
        assert returns.shape == (batch_size, 1)
        assert noisy_returns.shape == (batch_size, 1)
        assert vf1.shape == (batch_size, 1) and vf2.shape == (batch_size, 1)

        return bcfm_loss, {
            'bcfm_loss': bcfm_loss,
            'returns_mean': returns.mean(),
            'returns_std': returns.std(),
            'vf1_mean': vf1.mean(),
            'vf2_mean': vf2.mean(),
        }
    
    def sample_actions(self, obs_with_id, legals, rng):
        ## Sample discrete actions from the one-step actor with legal-action masking
        B, K = obs_with_id.shape[:2]
        A = self.config['action_dim']
        noises = jax.random.normal(rng, (B, K, A))
        # Call WITHOUT params=grad_params — no gradient flows back to actor
        logits = self.network.select('actor_onestep_flow')(obs_with_id, noises)
        masked_logits = jnp.where(legals > 0, logits, -1e9)
        actions_int = jnp.argmax(masked_logits, axis=-1)
        return actions_int

    def compute_flow_returns(
        self,
        noises,
        observations,
        actions,
        init_times=None,
        end_times=None,
        flow_network_name=None,
        return_jac_eps_prod=False,
    ):
        """Compute returns from the return flow model using the Euler method."""
        noisy_returns = noises
        noisy_jac_eps_prod = jnp.ones_like(noises)
        if init_times is None:
            init_times = jnp.zeros((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        if end_times is None:
            end_times = jnp.ones((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def func(carry, i):
            """
            carry: (noisy_returns, )
            i: current step index
            """
            (noisy_returns, noisy_jac_eps_prod) = carry

            times = i * step_size + init_times
            vector_field, jac_eps_prod = jax.jvp(
                lambda ret: self.network.select(flow_network_name)(ret, times, observations, actions),
                (noisy_returns, ),
                (noisy_jac_eps_prod, ),
            )

            new_noisy_returns = noisy_returns + step_size * vector_field
            new_noisy_jac_eps_prod = noisy_jac_eps_prod + step_size * jac_eps_prod
            if self.config['clip_flow_returns']:
                new_noisy_returns = jnp.clip(
                    new_noisy_returns,
                    self.config['min_reward'] / (1 - self.config['discount']),
                    self.config['max_reward'] / (1 - self.config['discount']),
                )

            return (new_noisy_returns, new_noisy_jac_eps_prod), None

        # Use lax.scan to do the iteration
        (noisy_returns, noisy_jac_eps_prod), _ = jax.lax.scan(
            func, (noisy_returns, noisy_jac_eps_prod), jnp.arange(self.config['num_flow_steps']))

        if return_jac_eps_prod:
            return noisy_returns, noisy_jac_eps_prod
        else:
            return noisy_returns
    
    def joint_critic_dcfm_loss(self, batch, grad_params, rng):
        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']
        rng, target_noise_rng, time_rng = jax.random.split(rng, 3)

        states = batch['states']
        next_states = batch['next_states']
        actions_oh = jax.nn.one_hot(batch['actions'], A)
        joint_actions = actions_oh.reshape(batch_size, K * A)

        # Next-action placeholder (same as BCFM; replace with actor in step 4)
        next_joint_actions = joint_actions  # TODO(step 4): use actor samples

        times = jax.random.uniform(time_rng, (batch_size, 1))            # t ∈ [0, 1]
        noises = jax.random.normal(target_noise_rng, (batch_size, 1))    # ε

        # Integrate target critic flow from t=0 to t

        noisy_next_returns1 = self.compute_flow_returns(
            noises, next_states, next_joint_actions, end_times=times,
            flow_network_name='target_joint_critic_flow1')
        noisy_next_returns2 = self.compute_flow_returns(
            noises, next_states, next_joint_actions, end_times=times,
            flow_network_name='target_joint_critic_flow2')

        if self.config['ret_agg'] == 'min':
            noisy_next_returns = jnp.minimum(noisy_next_returns1, noisy_next_returns2)
        else:
            noisy_next_returns = (noisy_next_returns1 + noisy_next_returns2) / 2

        # Distributional Bellman change of variables
        noisy_returns = (jnp.expand_dims(batch['rewards'], axis=-1)
                        + self.config['discount']
                        * jnp.expand_dims(batch['masks'], axis=-1)
                        * noisy_next_returns)                            # (B, 1)

        vf1 = self.network.select('joint_critic_flow1')(
            noisy_returns, times, states, joint_actions, params=grad_params)
        vf2 = self.network.select('joint_critic_flow2')(
            noisy_returns, times, states, joint_actions, params=grad_params)

        # Target critic velocity at next (s', a') — frozen (no grad_params).
        target_vf1 = self.network.select('target_joint_critic_flow1')(
            noisy_next_returns, times, next_states, next_joint_actions)
        target_vf2 = self.network.select('target_joint_critic_flow2')(
            noisy_next_returns, times, next_states, next_joint_actions)

        if self.config['ret_agg'] == 'min':
            target_vf = jnp.minimum(target_vf1, target_vf2)
        else:
            target_vf = (target_vf1 + target_vf2) / 2
        target_vf = jax.lax.stop_gradient(target_vf)

        dcfm_loss = ((vf1 - target_vf) ** 2 + (vf2 - target_vf) ** 2).mean()

        # --- Shape assertions ---
        assert noisy_next_returns.shape == (batch_size, 1)
        assert noisy_returns.shape == (batch_size, 1)
        assert vf1.shape == (batch_size, 1) and vf2.shape == (batch_size, 1)
        assert target_vf.shape == (batch_size, 1)

        return dcfm_loss, {
            'dcfm_loss': dcfm_loss,
            'noisy_next_returns_mean': noisy_next_returns.mean(),
            'noisy_returns_mean': noisy_returns.mean(),
            'target_vf_mean': target_vf.mean(),
            'vf1_mean': vf1.mean(),
            'vf2_mean': vf2.mean(),
        }
    
    def q_head_factorization_loss(self, batch, grad_params, rng):
        ## Distilling V_joint (teacher), Expectation of Z_joint(s,a) into student q-values
        # Sum q-values to follow IGM (VDN-style, per-agent head)
        # Loss is MSE between student and teacher

        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']
        rng, q_rng = jax.random.split(rng)

        states = batch['states']
        actions_oh = jax.nn.one_hot(batch['actions'], A)
        joint_actions = actions_oh.reshape(batch_size, K * A)

        # Teacher: expectation on main joint critic (V_joint)
        q_noises = jax.random.normal(q_rng, (batch_size, 1))
        z1 = q_noises + self.network.select('joint_critic_flow1')(
        q_noises, jnp.zeros_like(q_noises), states, joint_actions)       
        z2 = q_noises + self.network.select('joint_critic_flow2')(
        q_noises, jnp.zeros_like(q_noises), states, joint_actions)

        if self.config['clip_flow_returns']:
            return_min = self.config['min_reward'] / (1 - self.config['discount'])
            return_max = self.config['max_reward'] / (1 - self.config['discount'])
            z1 = jnp.clip(z1, return_min, return_max)
            z2 = jnp.clip(z2, return_min, return_max)
        
        if self.config['q_agg'] == 'min':
            v_teacher = jnp.minimum(z1, z2)
        else:
            v_teacher = (z1 + z2) / 2

        v_teacher = jax.lax.stop_gradient(v_teacher)   

        # VDN sum over per-agent Q-heads
        obs_with_id = batch_concat_agent_id_to_obs(batch['observations'])     # (B, K, obs_dim + K)
        q_per_agent = self.network.select('q_heads')(
            obs_with_id, actions_oh, params=grad_params)                      # (B, K, 1)
        q_tot_factor = q_per_agent.sum(axis=-1, keepdims = True)

        # MSE Loss
        loss = ((q_tot_factor - v_teacher) ** 2).mean()


        assert q_per_agent.shape == (batch_size, K), f"q_per_agent: {q_per_agent.shape}"
        assert q_tot_factor.shape == (batch_size, 1), f"q_tot_factor: {q_tot_factor.shape}"
        assert v_teacher.shape == (batch_size, 1), f"v_teacher: {v_teacher.shape}"

        return loss, {
        'q_head_loss': loss,
        'v_teacher_mean': v_teacher.mean(),
        'q_tot_factor_mean': q_tot_factor.mean(),
        'q_per_agent_mean': q_per_agent.mean(),
        'q_per_agent_std_across_agents': q_per_agent.std(axis=-1).mean(),
        }
    

    def _integrate_actor_flow(self, noises, observations, flow_network_name):
        num_steps = self.config['num_flow_steps']
        step_size = 1.0 / num_steps
        times_shape = (*noises.shape[:-1], 1)   # (B, K, 1)

        def step_fn(carry, i):
            x = carry
            t = jnp.full(times_shape, i * step_size, dtype=x.dtype)
            vf = self.network.select(flow_network_name)(observations, x, t)
            return x + step_size * vf, None

        final, _ = jax.lax.scan(step_fn, noises, jnp.arange(num_steps))
        return final


    def actor_loss(self, batch, grad_params, rng):

        # BC Flow 
        # Distillation target
        # Q-guidance

        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']
        alpha = self.config['alpha']

        rng, bc_noise_rng, bc_time_rng, distill_noise_rng, qguide_noise_rng = jax.random.split(rng, 5)


        obs_with_id = batch_concat_agent_id_to_obs(batch['observations'])     # (B, K, obs_dim + K)
        actions_oh = jax.nn.one_hot(batch['actions'], A)                      # (B, K, A)
        legals = batch['legals']                                              # (B, K, A)

        # 1. BC flow loss — train multi-step actor to imitate dataset
        bc_noises = jax.random.normal(bc_noise_rng, (batch_size, K, A))       # ε in action space
        bc_times = jax.random.uniform(bc_time_rng, (batch_size, K, 1))        # t per agent
        bc_noisy_actions = bc_times * actions_oh + (1 - bc_times) * bc_noises
        bc_target_vf = actions_oh - bc_noises                                 # (B, K, A)

        bc_vf = self.network.select('actor_bc_flow')(
            obs_with_id, bc_noisy_actions, bc_times, params=grad_params)      # (B, K, A)

        bc_loss = ((bc_vf - bc_target_vf) ** 2).mean()

        # 2. Distillation loss — one-step actor matches argmax of multi-step rollout
        # Integrate multi-step flow from noise to action-space; argmax for target label.

        distill_noises = jax.random.normal(distill_noise_rng, (batch_size, K, A))
        multistep_logits = self._integrate_actor_flow(
            distill_noises, obs_with_id, flow_network_name='actor_bc_flow'
        )                                                                     # (B, K, A)
        target_labels = jnp.argmax(multistep_logits, axis=-1)                 # (B, K), integer
        target_labels = jax.lax.stop_gradient(target_labels)

        onestep_logits_for_distill = self.network.select('actor_onestep_flow')(
            obs_with_id, distill_noises, params=grad_params)                  # (B, K, A)

        # Cross-entropy: -log p(target_label) under one-step actor's softmax
        log_probs = jax.nn.log_softmax(onestep_logits_for_distill, axis=-1)
        distill_loss = -jnp.take_along_axis(
            log_probs, target_labels[..., None], axis=-1
        ).squeeze(-1).mean()


        # 3. Q-guidance loss — one-step actor maximizes Q_tot^fac
        qguide_noises = jax.random.normal(qguide_noise_rng, (batch_size, K, A))
        onestep_logits_for_q = self.network.select('actor_onestep_flow')(
            obs_with_id, qguide_noises, params=grad_params)                   # (B, K, A)

        # Mask illegal actions before softmax
        masked_logits = jnp.where(legals > 0, onestep_logits_for_q, -1e9)
        soft_actions = jax.nn.softmax(masked_logits, axis=-1)                 # (B, K, A) differentiable

        # Q-heads called WITHOUT params=grad_params → no gradient to heads
        q_per_agent = self.network.select('q_heads')(obs_with_id, soft_actions)   # (B, K)
        q_tot = q_per_agent.sum(axis=-1, keepdims=True)                       # (B, 1)

        if self.config['normalize_q_loss']:
            # Normalize by absolute Q magnitude (MAC-Flow convention) — keeps Q-loss scale
            # comparable to BC/distill losses as Q magnitudes drift during training.
            q_loss = -q_tot.mean() / (jax.lax.stop_gradient(jnp.abs(q_tot).mean()) + 1e-6)
        else:
            q_loss = -q_tot.mean()

        # Combine
        actor_loss = bc_loss + alpha * distill_loss + q_loss

        # --- Shape assertions ---
        assert bc_vf.shape == (batch_size, K, A)
        assert multistep_logits.shape == (batch_size, K, A)
        assert target_labels.shape == (batch_size, K)
        assert soft_actions.shape == (batch_size, K, A)
        assert q_tot.shape == (batch_size, 1)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q_tot_mean': q_tot.mean(),
            'multistep_argmax_entropy': -(
                jax.nn.softmax(multistep_logits, axis=-1)
                * jax.nn.log_softmax(multistep_logits, axis=-1)
            ).sum(axis=-1).mean(),
            'onestep_entropy': -(
                jax.nn.softmax(masked_logits, axis=-1)
                * jax.nn.log_softmax(masked_logits, axis=-1)
            ).sum(axis=-1).mean(),
        }
    
    ## create() needs to 1). extract shapes from transitions, 
    #   2). Define neural nets: 4 critic flows (2 main + 2 targets), 2 actor flows
    @classmethod
    def create(
        cls,
        seed: int,
        example_batch,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_obs = example_batch['observations']
        ex_states = example_batch['states']
        ex_actions = example_batch['actions']
        ex_legals = example_batch['legals']

        num_agents = ex_obs.shape[1]
        ob_dims = ex_obs.shape[-1]
        state_dim = ex_states.shape[-1]
        action_dim = ex_legals.shape[-1]
        min_reward = float(example_batch['min_reward'])
        max_reward = float(example_batch['max_reward'])

        ex_actions_per_agent_oh = jax.nn.one_hot(ex_actions, action_dim)

        ex_returns_joint = jnp.zeros((1, 1), dtype = jnp.float32)
        ex_times_joint = jnp.zeros((1, 1), dtype = jnp.float32)
        ex_actions_joint_oh = ex_actions_per_agent_oh.reshape(1, num_agents * action_dim)
        ex_joint_critic_state = ex_states

        ex_obs_with_id = batch_concat_agent_id_to_obs(ex_obs)        # (1, K, obs_dim + K)
        ex_actor_obs = ex_obs_with_id
        ex_times_per_agent = jnp.zeros((1, num_agents, 1), dtype = jnp.float32)

        ## SMAC doesn't require a CNN encoder; MAC-FLOW uses LSTMs to handle partial observability

        # Define networks.
        joint_critic_flow1_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=None,
        )
        joint_critic_flow2_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=None,
        )
        # declare the target critics explicitly to prevent errors for visual tasks
        target_joint_critic_flow1_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=None,
        )
        target_joint_critic_flow2_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=None,
        )
        
        q_heads_def = Value(
            hidden_dims=config['value_hidden_dims'],
            value_dim=1,
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=None,
            )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=None,
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=None,
        )


        network_info = dict(
            joint_critic_flow1=(joint_critic_flow1_def, (ex_returns_joint, ex_times_joint, ex_joint_critic_state, ex_actions_joint_oh)),
            joint_critic_flow2=(joint_critic_flow2_def, (ex_returns_joint, ex_times_joint, ex_joint_critic_state, ex_actions_joint_oh)),
            target_joint_critic_flow1=(target_joint_critic_flow1_def, (ex_returns_joint, ex_times_joint, ex_joint_critic_state, ex_actions_joint_oh)),
            target_joint_critic_flow2=(target_joint_critic_flow2_def, (ex_returns_joint, ex_times_joint, ex_joint_critic_state, ex_actions_joint_oh)),
            ## Q-heads don't see the full state -> preserves IGM
            q_heads=(q_heads_def, (ex_obs_with_id, ex_actions_per_agent_oh)),
            actor_bc_flow=(actor_bc_flow_def, (ex_actor_obs, ex_actions_per_agent_oh, ex_times_per_agent)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_actor_obs, ex_actions_per_agent_oh)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network_params['modules_target_joint_critic_flow1'] = network_params['modules_joint_critic_flow1']
        network_params['modules_target_joint_critic_flow2'] = network_params['modules_joint_critic_flow2']

        network = TrainState.create(network_def, network_params, tx=network_tx)

        


        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['num_agents'] = num_agents
        config['state_dim'] = state_dim
        config['min_reward'] = min_reward
        config['max_reward'] = max_reward
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
    
    def total_loss(self, batch, grad_params, rng):
        rng, q_head_rng, bcfm_rng, dcfm_rng, actor_rng = jax.random.split(rng, 5)

        q_head_loss, q_head_info = self.q_head_factorization_loss(
            batch, grad_params, q_head_rng)
        bcfm_loss, bcfm_info = self.joint_critic_bcfm_loss(
            batch, grad_params, bcfm_rng)
        dcfm_loss, dcfm_info = self.joint_critic_dcfm_loss(
            batch, grad_params, dcfm_rng)
        a_loss, actor_info = self.actor_loss(
            batch, grad_params, actor_rng)

        # Combine with lambdas
        critic_loss = (self.config['bcfm_lambda'] * bcfm_loss
                    + self.config['dcfm_lambda'] * dcfm_loss)
        total = critic_loss + q_head_loss + a_loss

        info = {'total_loss': total, 'critic_loss': critic_loss}
        info.update({f'q_head/{k}': v for k, v in q_head_info.items()})
        info.update({f'bcfm/{k}': v for k, v in bcfm_info.items()})
        info.update({f'dcfm/{k}': v for k, v in dcfm_info.items()})
        info.update({f'actor/{k}': v for k, v in actor_info.items()})
        return total, info
    
    def target_update(self, network, module_name):
        """Polyak EMA: target ← τ · main + (1 - τ) · target. Mutates network.params in place."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        ## One gradient step on all losses + EMA target update.
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        # Polyak EMA on both joint critics
        self.target_update(new_network, 'joint_critic_flow1')
        self.target_update(new_network, 'joint_critic_flow2')

        return self.replace(network=new_network, rng=new_rng), info
  
def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='mav_flow',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(int),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            num_agents = ml_collections.config_dict.placeholder(int),
            state_dim = ml_collections.config_dict.placeholder(int),
            min_reward=ml_collections.config_dict.placeholder(float),  # Minimum reward (will be set automatically).
            max_reward=ml_collections.config_dict.placeholder(float),  # Maximum reward (will be set automatically).
            encoder = ml_collections.config_dict.placeholder(str),

            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            value_layer_norm=True,  # Whether to use layer normalization for the value and the critic.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.

            ret_agg='mean',  # Aggregation method for return values.
            q_agg='mean',  # Aggregation method for Q values.
            clip_flow_actions=False,  # Whether to clip the intermediate flow actions.
            clip_flow_returns=True,  # Whether to clip flow returns.
            confidence_weight_temp=0.3,  # Temperature for the confidence weights.
            dcfm_lambda=1.0,  # Distributional conditional flow matching loss coefficient.
            bcfm_lambda=1.0,  # Bootstrapped conditional flow matching loss coefficient.
            alpha=10.0,  # Flow distillation coefficient.
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            num_flow_steps=10,  # Number of flow steps.
        )
    )
    return config
    
