import copy
from functools import partial
from logging import config
from typing import Any, Dict, Sequence

import flax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

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


## 7/10 - VDN style factorization of joint Q encounters a un-identifiability problem?
#   We need factorization to be clean -> so that rejection sampling in actor is effective.
#.  Q-MIX still has identifiability problem. But rejection sampling able to side-step as it only cares about ranking


## Q-Mixing Structure ported from JAXMARL library:
# https://github.com/FLAIROx/JaxMARL/blob/main/baselines/QLearning/qmix_rnn.py
class HyperNetwork(nn.Module):
    hidden_dim: int
    output_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(x)
        return x
    
class MixingNetwork(nn.Module):
    """QMIX monotonic mixer, no time axis. q_vals (B,K), states (B,state_dim) -> (B,1)."""
    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, q_vals, states):
        # q_vals: (B, K)   states: (B, state_dim)
        B, K = q_vals.shape

        w_1 = HyperNetwork(self.hypernet_hidden_dim, self.embedding_dim * K,
                           self.init_scale)(states)
        b_1 = nn.Dense(self.embedding_dim, kernel_init=orthogonal(self.init_scale),
                       bias_init=constant(0.0))(states)
        w_2 = HyperNetwork(self.hypernet_hidden_dim, self.embedding_dim,
                           self.init_scale)(states)
        b_2 = HyperNetwork(self.embedding_dim, 1, self.init_scale)(states)

        # monotonicity via |w|, reshape (no time axis)
        w_1 = jnp.abs(w_1).reshape(B, K, self.embedding_dim)   # (B, K, E)
        b_1 = b_1.reshape(B, 1, self.embedding_dim)            # (B, 1, E)
        w_2 = jnp.abs(w_2).reshape(B, self.embedding_dim, 1)   # (B, E, 1)
        b_2 = b_2.reshape(B, 1, 1)                             # (B, 1, 1)

        hidden = nn.elu(jnp.matmul(q_vals[:, None, :], w_1) + b_1)  # (B,1,E)
        q_tot = jnp.matmul(hidden, w_2) + b_2                       # (B,1,1)
        return q_tot.reshape(B, 1)   

class MAVFlowAgent(flax.struct.PyTreeNode):
    """MAV-Flow agent: distributional flow critic for cooperative MARL."""
    
    rng: Any
    network: Any
    config: Any = nonpytree_field()

        # -------------------- Sinusoidal Embedding (discrete only) -------------------- #
    def _time_sin_embed(self, ts):
        kfreq = int(self.config.get('t_embed_frequencies', 8))
        freqs = jnp.asarray([2 ** i for i in range(kfreq)], dtype=ts.dtype) * jnp.pi
        ang = ts * freqs
        return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    

    def _sample_candidates(self, obs_with_id, legals, rng, num_candidates):
        """N candidate actions per agent from the multi-step BC flow.
        Returns cand_int (N, B, K) and cand_oh (N, B, K, A)."""
        B, K = obs_with_id.shape[:2]
        A = self.config['action_dim']
        noises = jax.random.normal(rng, (num_candidates, B, K, A))
        logits = jax.vmap(
            lambda eps: self._integrate_actor_flow(eps, obs_with_id, 'actor_bc_flow')
        )(noises)                                                  # (N, B, K, A)
        masked = jnp.where(legals[None] > 0, logits, -1e9)
        cand_int = jnp.argmax(masked, axis=-1)                     # (N, B, K)
        return cand_int, jax.nn.one_hot(cand_int, A)


    ## Decentralized execution: needs to stay per-agent with local observations
    def sample_actions_rejection(self, obs_with_id, legals, rng):
        N = self.config['num_candidates']
        cand_int, cand_oh = self._sample_candidates(obs_with_id, legals, rng, N)
        q = jax.vmap(lambda a: self.network.select('q_heads')(obs_with_id, a))(cand_oh)  # (N, B, K)
        best = jnp.argmax(q, axis=0)                               # (B, K)
        return jnp.take_along_axis(cand_int, best[None], axis=0).squeeze(0)
    
    def _select_next_joint_action_teacher(self, next_obs_with_id, next_states,
                                      next_legals, rng):
        N = self.config.get('num_target_candidates', self.config['num_candidates'])
        B = next_states.shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']
        rng, cand_rng, score_rng = jax.random.split(rng, 3)

        # N coordinated candidates from the joint BC flow (same sampler execution uses)
        _, cand_oh = self._sample_candidates(
            next_obs_with_id, next_legals, cand_rng, N)      # (N,B,K), (N,B,K,A)

        L = self.config['teacher_mc_samples']
        use_full_ode = self.config.get('target_select_full_ode', False)

        def score(a_oh, key):
            joint_a = a_oh.reshape(B, K * A)
            eps = jax.random.normal(key, (L, B, 1))
            def one(e):
                if use_full_ode:                              # 10-step, consistent w/ bootstrap
                    z1 = self.compute_flow_returns(
                        e, next_states, joint_a,
                        flow_network_name='target_joint_critic_flow1')
                    z2 = self.compute_flow_returns(
                        e, next_states, joint_a,
                        flow_network_name='target_joint_critic_flow2')
                else:                                         # 1-step E[Z|ε], cheap ranking
                    z1 = e + self.network.select('target_joint_critic_flow1')(
                        e, jnp.zeros_like(e), next_states, joint_a)
                    z2 = e + self.network.select('target_joint_critic_flow2')(
                        e, jnp.zeros_like(e), next_states, joint_a)
                return jnp.minimum(z1, z2) if self.config['ret_agg'] == 'min' \
                    else 0.5 * (z1 + z2)
            return jax.vmap(one)(eps).mean(0)                 # (B, 1)

        keys = jax.random.split(score_rng, N)
        v_cand = jax.vmap(score)(cand_oh, keys).squeeze(-1)   # (N, B)
        best = jnp.argmax(v_cand, axis=0)                     # (B,)
        best_oh = cand_oh[best, jnp.arange(B)]                # (B, K, A) advanced indexing
        return jax.lax.stop_gradient(best_oh.reshape(B, K * A))


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
        
        ## Full ODE integration instead of one-step
    # One-step conditional mean E[Z | ε], ε is noise at t = 0
    def joint_critic_bcfm_loss(self, batch, grad_params, rng):

        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']
        rng, action_rng, target_noise_rng, bcfm_n1_rng, bcfm_n2_rng, t1_rng, t2_rng = jax.random.split(rng, 7)
        # target_noise_rng -> ε for target critic single-step trick
        # bcfm_noise_rng   -> ε for BCFM noise sample
        # time_rng         -> t for flow interpolation


        states = batch['states']
        next_states = batch['next_states']
        actions_oh = jax.nn.one_hot(batch['actions'], A)
        joint_actions = actions_oh.reshape(batch_size, K * A)

        next_obs_with_id = batch_concat_agent_id_to_obs(batch['next_observations'])
        next_joint_actions = self._select_next_joint_action_teacher(
            next_obs_with_id, next_states, batch['next_legals'], action_rng)

        # Return: use next-state expected Z (Changed to distributional instead)
        # self.network.select calls on the velocity at a given time, which is the conditional mean
        next_q_noises = jax.random.normal(target_noise_rng, (batch_size, 1))
        next_z1 = self.compute_flow_returns(
            next_q_noises, next_states, next_joint_actions, flow_network_name='target_joint_critic_flow1'
        )
        next_z2 = self.compute_flow_returns(
            next_q_noises, next_states, next_joint_actions, flow_network_name='target_joint_critic_flow2'
        )

        if self.config['ret_agg'] == 'min':
            next_returns = jnp.minimum(next_z1, next_z2)
        else:
            next_returns = (next_z1 + next_z2) / 2

        next_returns = jax.lax.stop_gradient(next_returns)   # Freezing boostrap target?

        # Bellman target: G = r + γ * mask * E[Z_tot(s', a')]
        returns = (jnp.expand_dims(batch['rewards'], axis=-1)
                + self.config['discount']
                * jnp.expand_dims(batch['masks'], axis=-1)
                * next_returns)                      

        # Independent CFM noise + time per twin. Both critics regress toward the SAME
        # bootstrap target `returns` (pessimism preserved) but see it at different
        # interpolation points, so their gradients decorrelate and the twins stop
        # collapsing to identical functions.
        noises1 = jax.random.normal(bcfm_n1_rng, (batch_size, 1))
        noises2 = jax.random.normal(bcfm_n2_rng, (batch_size, 1))
        times1  = jax.random.uniform(t1_rng, (batch_size, 1))
        times2  = jax.random.uniform(t2_rng, (batch_size, 1))

        noisy_returns1 = times1 * returns + (1 - times1) * noises1
        noisy_returns2 = times2 * returns + (1 - times2) * noises2
        target_vf1 = returns - noises1
        target_vf2 = returns - noises2

        vf1 = self.network.select('joint_critic_flow1')(
            noisy_returns1, times1, states, joint_actions, params=grad_params)
        vf2 = self.network.select('joint_critic_flow2')(
            noisy_returns2, times2, states, joint_actions, params=grad_params)

        bcfm_loss = ((vf1 - target_vf1) ** 2 + (vf2 - target_vf2) ** 2).mean()
        
        assert returns.shape == (batch_size, 1)
        assert noisy_returns1.shape == (batch_size, 1)
        assert noisy_returns2.shape == (batch_size, 1)
        assert vf1.shape == (batch_size, 1) and vf2.shape == (batch_size, 1)

        return bcfm_loss, {
            'bcfm_loss': bcfm_loss,
            'returns_mean': returns.mean(),
            'returns_std': returns.std(),
            'vf1_mean': vf1.mean(),
            'vf2_mean': vf2.mean(),
            'vf_twin_std': (vf1 - vf2).std(),
        }
    

    ## Training stage, can use centralized state
    # Therefore, better to not just rejection sample individual actions and join them with IGM
    # The stitched up rejection sampling action also likely falls out of the support of the joint actions in dataset
    # Use joint actions, scored by joint critic E[Z_joint(s,a)] 
    def joint_critic_dcfm_loss(self, batch, grad_params, rng):
        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']
        rng, n1_rng, n2_rng, t1_rng, t2_rng, action_rng = jax.random.split(rng, 6)

        states = batch['states']
        next_states = batch['next_states']
        actions_oh = jax.nn.one_hot(batch['actions'], A)
        joint_actions = actions_oh.reshape(batch_size, K * A)

        next_obs_with_id = batch_concat_agent_id_to_obs(batch['next_observations'])
        next_joint_actions = self._select_next_joint_action_teacher(
            next_obs_with_id, next_states, batch['next_legals'], action_rng)
            

        times1 = jax.random.uniform(t1_rng, (batch_size, 1))
        times2 = jax.random.uniform(t2_rng, (batch_size, 1))
        noises1 = jax.random.normal(n1_rng, (batch_size, 1))
        noises2 = jax.random.normal(n2_rng, (batch_size, 1))

        # Each twin integrates its OWN target flow to its OWN time, then matches that
        # target's velocity (paired j->j). No shared aggregated velocity target -> no
        # force pulling the twins together.
        znext1 = self.compute_flow_returns(
            noises1, next_states, next_joint_actions, end_times=times1,
            flow_network_name='target_joint_critic_flow1')
        znext2 = self.compute_flow_returns(
            noises2, next_states, next_joint_actions, end_times=times2,
            flow_network_name='target_joint_critic_flow2')

        r_ = jnp.expand_dims(batch['rewards'], -1)
        m_ = jnp.expand_dims(batch['masks'], -1)
        noisy_returns1 = r_ + self.config['discount'] * m_ * znext1
        noisy_returns2 = r_ + self.config['discount'] * m_ * znext2

        vf1 = self.network.select('joint_critic_flow1')(
            noisy_returns1, times1, states, joint_actions, params=grad_params)
        vf2 = self.network.select('joint_critic_flow2')(
            noisy_returns2, times2, states, joint_actions, params=grad_params)

        target_vf1 = jax.lax.stop_gradient(self.network.select('target_joint_critic_flow1')(
            znext1, times1, next_states, next_joint_actions))
        target_vf2 = jax.lax.stop_gradient(self.network.select('target_joint_critic_flow2')(
            znext2, times2, next_states, next_joint_actions))

        dcfm_loss = ((vf1 - target_vf1) ** 2 + (vf2 - target_vf2) ** 2).mean()

        # --- Shape assertions ---
        assert znext1.shape == (batch_size, 1)
        assert noisy_returns1.shape == (batch_size, 1)
        assert vf1.shape == (batch_size, 1) and vf2.shape == (batch_size, 1)
        assert target_vf1.shape == (batch_size, 1)

        return dcfm_loss, {
            'dcfm_loss': dcfm_loss,
            'noisy_next_returns_mean': ((znext1 + znext2) / 2).mean(),
            'noisy_returns_mean': ((noisy_returns1 + noisy_returns2) / 2).mean(),
            'target_vf1_mean': target_vf1.mean(),
            'target_vf2_mean': target_vf2.mean(),
            'vf1_mean': vf1.mean(),
            'vf2_mean': vf2.mean(),
            'vf_twin_std': (vf1 - vf2).std(),
        }
    
    def q_head_factorization_loss(self, batch, grad_params, rng):
    ## Distill V_joint = E[Z_joint(s,a)] into a QMIX-factored value, trained at
    ## BOTH the dataset action AND several BC-sampled candidate actions, so the
    ## per-agent heads can RANK off-dataset (but in-support) actions — makes
    ## rejection sampling more effective.

        # Originally, Q-head network is only trained on one joint action from the offline dataset (given any state)
        # Rejection sampling is ranking samples from BC flow -> will be difficult to identify truth against one sample point
        # Sample actions from BC Flow

        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']
        rng, q_rng, cand_rng = jax.random.split(rng, 3)

        states = batch['states']
        obs_with_id = batch_concat_agent_id_to_obs(batch['observations'])     # (B, K, obs_dim + K)
        actions_oh = jax.nn.one_hot(batch['actions'], A)

        return_min = self.config['min_reward'] / (1 - self.config['discount'])
        return_max = self.config['max_reward'] / (1 - self.config['discount'])

        def teacher_V(a_oh, key):
            ## Expectation of joint return distribution E[Z_joint(s,a)]; using 4 samples for some variance reduction
            joint_a = a_oh.reshape(batch_size, K * A)
            L = self.config['teacher_mc_samples']
            eps = jax.random.normal(key, (L, batch_size, 1))   # L independent draws

            def one_step(e):
                z1 = e + self.network.select('target_joint_critic_flow1')(
                    e, jnp.zeros_like(e), states, joint_a)
                z2 = e + self.network.select('target_joint_critic_flow2')(
                    e, jnp.zeros_like(e), states, joint_a)
                if self.config['clip_flow_returns']:
                    z1 = jnp.clip(z1, return_min, return_max)
                    z2 = jnp.clip(z2, return_min, return_max)
                return z1, z2

            z1, z2 = jax.vmap(one_step)(eps)          # each (L, batch_size, 1)
            z1 = z1.mean(axis=0)                       # average over MC draws -> (B, 1)
            z2 = z2.mean(axis=0)
            V = jnp.minimum(z1, z2) if self.config['q_agg'] == 'min' else (z1 + z2) / 2
            return jax.lax.stop_gradient(V)
        
        def student_Q(a_oh):
            ## factorizes into per-agent q_i, with local observations; Q-MIX
            q_pa = self.network.select('q_heads')(obs_with_id, a_oh, params=grad_params)
            return self.network.select('mixer')(q_pa, states, params=grad_params)


        # Dataset action
        V_data = teacher_V(actions_oh, q_rng)
        Q_data = student_Q(actions_oh)
        loss_data = ((Q_data - V_data) ** 2).mean()

        # BC-sampled candidate actions
        N_q = self.config['num_q_candidates']
        cand_keys = jax.random.split(cand_rng, N_q)
        _, cand_oh = self._sample_candidates(
            obs_with_id, batch['legals'], cand_rng, N_q)        # (N_q, B, K, A)
        V_cand = jax.vmap(teacher_V)(cand_oh, cand_keys)          # (N_q, B, 1)
        Q_cand = jax.vmap(student_Q)(cand_oh)                    # (N_q, B, 1)
        loss_cand = ((Q_cand - V_cand) ** 2).mean()

        loss = loss_data + loss_cand

        q_per_agent = self.network.select('q_heads')(
            obs_with_id, actions_oh, params=grad_params)         # (B, K)
        
        # Per-agent q over the candidate set: (N_q, B, K)
        q_pa_cand = jax.vmap(
            lambda a: self.network.select('q_heads')(obs_with_id, a, params=grad_params)
        )(cand_oh)

        # METRIC 1: does each agent's head actually separate the candidates?
        # std over the N_q candidates, per agent, then averaged. This is the
        # execution-time ranking signal. If it collapses toward 0, argmax is noise.
        q_cand_spread_per_agent = q_pa_cand.std(axis=0).mean()          # scalar

        # METRIC 2a: agreement between per-agent stitched argmax and the teacher's
        # preferred WHOLE candidate. Fraction of (B) where they coincide.
        teacher_best = jnp.argmax(V_cand.squeeze(-1), axis=0)          # (B,) best whole candidate by teacher
        stitched_best = jnp.argmax(q_pa_cand, axis=0)                  # (B, K) per-agent pick
        # "does every agent's stitched pick land on the teacher's whole-candidate choice?"
        stitched_matches_teacher = (stitched_best == teacher_best[:, None]).all(axis=-1).mean()

        # METRIC 2b: student-mixer whole-candidate choice vs teacher whole-candidate choice
        student_best = jnp.argmax(Q_cand.squeeze(-1), axis=0)          # (B,)
        student_matches_teacher = (student_best == teacher_best).mean()
        
        assert q_per_agent.shape == (batch_size, K)
        assert V_data.shape == (batch_size, 1) and Q_data.shape == (batch_size, 1)

        return loss, {
            'q_head_loss': loss,
            'q_head_loss_data': loss_data,
            'q_head_loss_cand': loss_cand,
            'v_teacher_mean': V_data.mean(),
            'q_tot_factor_mean': Q_data.mean(),
            'q_per_agent_mean': q_per_agent.mean(),
            'q_per_agent_std_across_agents': q_per_agent.std(axis=-1).mean(),
            'v_cand_std': V_cand.std(),   # teacher spread across candidates = ranking signal
            'q_cand_spread_per_agent': q_cand_spread_per_agent,
            'stitched_matches_teacher': stitched_matches_teacher,
            'student_matches_teacher': student_matches_teacher,
        }
    

    def _integrate_actor_flow(self, noises, observations, flow_network_name):
        num_steps = self.config['num_flow_steps']
        step_size = 1.0 / num_steps
        times_shape = (*noises.shape[:-1], 1)   # (B, K, 1)

        def step_fn(carry, i):
            x = carry
            t = jnp.full(times_shape, i * step_size, dtype=x.dtype)
            t_embed = self._time_sin_embed(t)
            vf = self.network.select(flow_network_name)(observations, x, t_embed)
            return x + step_size * vf, None

        final, _ = jax.lax.scan(step_fn, noises, jnp.arange(num_steps))
        return final


    def actor_loss(self, batch, grad_params, rng):

        # BC Flow 
        # Rejection Sampling: initiate 16 noises, push through BC flow policy to get actions
        # Actions are then scored per-agent, taken the argmax. Combined IGM style
        ## But should the scoring be on the joint? We trained a joint critic?
        # Is it also computationally too expensive to do the joint? 
        # All combinations is A^K (A is action space, K is # of agents) - not scalable for SMAVc2 benchmarks with 20 agents
        

        # Q-guidance: take out for now as we are doing offline RL and not offline-to-online tuning.
        #  It is a mechansim that pushes the actor towards
        #  the action that critic scores the highest. 

        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']

        rng, bc_noise_rng, bc_time_rng = jax.random.split(rng, 3)


        obs_with_id = batch_concat_agent_id_to_obs(batch['observations'])     # (B, K, obs_dim + K)
        actions_oh = jax.nn.one_hot(batch['actions'], A)                      # (B, K, A)

        # 1. BC flow loss — train multi-step actor to imitate dataset
        bc_noises = jax.random.normal(bc_noise_rng, (batch_size, K, A))       # ε in action space
        bc_times = jax.random.uniform(bc_time_rng, (batch_size, K, 1))        # t per agent
        bc_noisy_actions = bc_times * actions_oh + (1 - bc_times) * bc_noises
        bc_target_vf = actions_oh - bc_noises                                 # (B, K, A)
        
        bc_t_embed = self._time_sin_embed(bc_times)
        bc_vf = self.network.select('actor_bc_flow')(
            obs_with_id, bc_noisy_actions, bc_t_embed, params=grad_params)      # (B, K, A)
        assert bc_vf.shape == (batch_size, K, A)

        bc_loss = ((bc_vf - bc_target_vf) ** 2).mean()
        actor_loss = bc_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
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

        kfreq = int(config.get('t_embed_frequencies', 8))
        ex_times_per_agent = jnp.zeros((1, num_agents, 2 * kfreq), dtype = jnp.float32)

        num_agents = ex_obs.shape[1]
        ob_dims    = ex_obs.shape[-1]
        state_dim  = ex_states.shape[-1]
        action_dim = ex_legals.shape[-1]
        # populate config immediately so any config[...] read below is valid
        config['ob_dims']    = ob_dims
        config['action_dim'] = action_dim
        config['num_agents'] = num_agents
        config['state_dim']  = state_dim
        config['min_reward'] = min_reward
        config['max_reward'] = max_reward

        ex_q_vals = jnp.zeros((ex_states.shape[0], num_agents))  # (B, K) for mixer init

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
        
        mixer_def = MixingNetwork(
            embedding_dim=config['mixer_embed_dim'],
            hypernet_hidden_dim=config['mixer_hypernet_hidden_dim'],
            init_scale=config['mixer_init_scale'],
        )

        actor_bc_flow_def = ActorVectorField(
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
            mixer=(mixer_def, (ex_q_vals, ex_states)),
            actor_bc_flow=(actor_bc_flow_def, (ex_actor_obs, ex_actions_per_agent_oh, ex_times_per_agent)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network_params['modules_target_joint_critic_flow1'] = network_params['modules_joint_critic_flow1']
        network_params['modules_target_joint_critic_flow2'] = network_params['modules_joint_critic_flow2']

        network = TrainState.create(network_def, network_params, tx=network_tx)

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

            ret_agg='min',  # Aggregation method for return values.
            q_agg='min',  # Aggregation method for Q values.
            clip_flow_actions=False,  # Whether to clip the intermediate flow actions.
            clip_flow_returns=True,  # Whether to clip flow returns.
            confidence_weight_temp=0.3,  # Temperature for the confidence weights.
            dcfm_lambda=1.0,  # Distributional conditional flow matching loss coefficient.
            bcfm_lambda=1.0,  # Bootstrapped conditional flow matching loss coefficient.
            alpha=10.0,  # Flow distillation coefficient.
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            num_flow_steps=10,  # Number of flow steps.
            mixer_embed_dim=32,            # QMIX mixing embedding dim
            mixer_hypernet_hidden_dim=64,  # hypernetwork hidden width
            mixer_init_scale=1.0,          # orthogonal init scale for mixer/hypernet
            num_candidates = 16, # Number of candidate actions for rejection sampling
            num_q_candidates = 4, # Number of sampled candidates to train q-heads 
            t_embed_frequencies = 8, # Number of frequencies for sinusoidal time embedding
            teacher_mc_samples = 4, # Number of MC samples for teacher V estimation
            num_target_candidates = 8,
            target_select_full_ode = False,  # Whether to use full ODE integration for target selection
        )
    )
    return config
    

