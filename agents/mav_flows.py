import copy
from functools import partial
from logging import config
from typing import Any, Dict, Sequence

from traitlets import This

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


## 8/31 - WEIGHTED PROJECTION (WQMIX)
#   q_head_factorization_loss IS the QMIX projection operator:
#       Pi_mono Qhat* = argmin_{q in Q_mono} sum_a w(s,a) (q(s,a) - Qhat*(s,a))^2
#   with Qhat* = E[Z_joint] (the unrestricted joint flow critic) and w == 1 (uniform).
#   WQMIX's thesis: uniform w is the bug -- the monotonic class flattens the peak and
#   moves the argmax. The fix is to weight the projection toward high-value actions.
#
#   Departure from the paper: WQMIX uses binary weights {1, alpha} because its Q_tot is
#   only consumed by argmax. OURS is consumed by sample_actions_rejection, which RANKS
#   num_candidates actions -- so we need ordinal accuracy across the whole candidate set,
#   not just at the top. Binary weights would starve ~15/16 candidates of supervision and
#   turn the ranking among them into noise. We therefore use a soft (softmax) weight.
#
#   Four schemes are implemented behind `w_mode`, for ablation:
#       'uniform' -> w == 1. Previous behaviour, exact no-op. DEFAULT.
#       'soft'    -> ours. w = N_q * softmax(standardize(V) / w_temp) over candidates.
#                    w_temp -> inf recovers 'uniform'; w_temp -> 0 recovers hard argmax.
#                    V is standardized per state, so w_temp is in "std devs of candidate
#                    spread" and does not silently sharpen as returns grow. Weights come
#                    from the FROZEN TEACHER V, so there is no feedback loop.
#       'ow'      -> OW-QMIX. w = 1 if V_teacher > Q_student else w_alpha.
#       'cw'      -> CW-QMIX. w = 1 where [a == argmax_a V_teacher] OR [y > V_teacher
#                    at that argmax], else w_alpha. Matches oxwhirl/wqmix's
#                    max_q_learner.py (central action from the UNRESTRICTED critic;
#                    second clause compares the bootstrap target y, per-state).
#                    Fixed 9/03 -- it previously took the argmax from the STUDENT and
#                    compared candidates against the teacher's value at that pick,
#                    which let the student choose where it got supervised.
#   `w_normalize` optionally rescales weights to mean 1 over the candidate axis. It is
#   FALSE by default (paper-faithful absolute weights): renormalizing divides out a
#   constant whenever all candidates share the same binary condition, collapsing
#   'ow'/'cw' back to 'uniform' on those states. Set True only to ablate weighting
#   SHAPE at a matched effective learning rate.


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


def _rank_along_axis0(x):
    """Ranks (0..n-1) of x along axis 0. Ties broken arbitrarily (fine for N_q small)."""
    return jnp.argsort(jnp.argsort(x, axis=0), axis=0).astype(x.dtype)


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
        Returns cand_int (N, B, K) and cand_oh (N, B, K, A).

        NOTE (8/31): actor_bc_flow is per-agent (independent noise, local obs, no
        cross-agent coupling), so these joint candidates are drawn from the PRODUCT OF
        MARGINALS prod_i mu_i(a_i|o_i), not from the joint behaviour policy mu(a|s).
        See Score Decomposition (arXiv 2505.05968) Prop 4.1: 2 coordinated modes become
        2^K spurious modes. Not fixed here -- tracked as a separate workstream.
        """
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
    # WAS (pre-0903): no decorator.
    #   def sample_actions_rejection(self, obs_with_id, legals, rng):
    # This is called once per env step from utils/evaluate_smac.py. Un-jitted, every
    # call re-traces and re-lowers a vmap-over-scan-over-flax-apply graph
    # (num_candidates x num_flow_steps x K). Measured on the 0827 500k runs:
    # 0.613 s/env-step on 3s5z (K=8) and 0.632 s/env-step on 5m_vs_6m (K=5) -- FLAT
    # in problem size, i.e. pure Python dispatch, not FLOPs. That made eval 61% of a
    # 26.8 h wall clock. Same fix, same rationale as mav_flows_per_agent.py:254.
    @jax.jit
    def sample_actions_rejection(self, obs_with_id, legals, rng):
        """Rejection-sample per-agent actions from the BC flow, ranked by q_heads.

        Plain @jax.jit (no static_argnames) is correct here: all three args are
        arrays, and the 'actor_bc_flow'/'q_heads' strings reaching network.select are
        closure constants inlined at trace time, not parameters of this function.
        `config` is a nonpytree_field FrozenDict, so num_candidates / num_flow_steps /
        action_dim land in the treedef and the lax.scan length stays static.

        Two shapes get traced and cached: B=1 from evaluate_smac, and
        B=churn_probe_size from the churn probe in main.py.
        """
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

        def vf_fn(ret, times):
            return self.network.select(flow_network_name)(ret, times, observations, actions)

        def func(carry, i):
            """
            carry: (noisy_returns, noisy_jac_eps_prod)
            i: current step index
            """
            (noisy_returns, noisy_jac_eps_prod) = carry

            times = i * step_size + init_times

            # WAS (pre-0903): the jvp was built unconditionally --
            #   vector_field, jac_eps_prod = jax.jvp(
            #       lambda ret: self.network.select(flow_network_name)(
            #           ret, times, observations, actions),
            #       (noisy_returns, ),
            #       (noisy_jac_eps_prod, ),
            #   )
            #   new_noisy_jac_eps_prod = noisy_jac_eps_prod + step_size * jac_eps_prod
            # No caller in THIS file ever passes return_jac_eps_prod=True (only
            # agents/value_flows.py:45,48 do), so the tangent was computed and thrown
            # away on all four calls per gradient step -- each a num_flow_steps scan at
            # B=batch_size, and the jvp roughly doubles that integration. Only build it
            # when asked. Same pattern as mav_flows_per_agent.py:180-194.
            if return_jac_eps_prod:
                vector_field, jac_eps_prod = jax.jvp(
                    lambda ret: vf_fn(ret, times),
                    (noisy_returns, ),
                    (noisy_jac_eps_prod, ),
                )
                new_noisy_jac_eps_prod = noisy_jac_eps_prod + step_size * jac_eps_prod
            else:
                vector_field = vf_fn(noisy_returns, times)
                new_noisy_jac_eps_prod = noisy_jac_eps_prod

            new_noisy_returns = noisy_returns + step_size * vector_field
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
    def joint_critic_bcfm_loss(self, batch, grad_params, rng,
                               joint_actions, next_joint_actions, bootstrap_target):
        """BCFM half of the Bellman backup. `joint_actions`, `next_joint_actions` and
        `bootstrap_target` (= y) are computed once in total_loss and passed in -- see
        the note there."""

        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']
        # WAS (pre-0903): also split off `action_rng` (its own a') and
        # `target_noise_rng` (its own y) here --
        #   rng, action_rng, target_noise_rng, bcfm_n1_rng, bcfm_n2_rng, t1_rng, t2_rng = \
        #       jax.random.split(rng, 7)
        rng, bcfm_n1_rng, bcfm_n2_rng, t1_rng, t2_rng = jax.random.split(rng, 5)
        # bcfm_noise_rng   -> ε for BCFM noise sample
        # time_rng         -> t for flow interpolation


        states = batch['states']
        next_states = batch['next_states']

        # WAS (pre-0903): built locally, duplicating joint_critic_dcfm_loss --
        #   actions_oh = jax.nn.one_hot(batch['actions'], A)
        #   joint_actions = actions_oh.reshape(batch_size, K * A)
        #   next_obs_with_id = batch_concat_agent_id_to_obs(batch['next_observations'])
        #   next_joint_actions = self._select_next_joint_action_teacher(
        #       next_obs_with_id, next_states, batch['next_legals'], action_rng)

        # WAS (pre-0903): the bootstrap target was built here, and only here --
        #   next_q_noises = jax.random.normal(target_noise_rng, (batch_size, 1))
        #   next_z1 = self.compute_flow_returns(
        #       next_q_noises, next_states, next_joint_actions,
        #       flow_network_name='target_joint_critic_flow1')
        #   next_z2 = self.compute_flow_returns(
        #       next_q_noises, next_states, next_joint_actions,
        #       flow_network_name='target_joint_critic_flow2')
        #   next_returns = jnp.minimum(next_z1, next_z2) if ret_agg == 'min' else ...
        #   next_returns = jax.lax.stop_gradient(next_returns)
        #   returns = r + discount * masks * next_returns
        # It now comes from total_loss, because q_head_factorization_loss needs the
        # SAME y for the paper-faithful CW-QMIX weighting condition and there is no
        # reason to integrate the target ODEs twice for it.
        # Bellman target: G = r + γ * mask * E[Z_tot(s', a')]
        returns = bootstrap_target

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
    def joint_critic_dcfm_loss(self, batch, grad_params, rng,
                               joint_actions, next_joint_actions):
        """DCFM half of the Bellman backup. Shares `next_joint_actions` with
        joint_critic_bcfm_loss -- see the note in total_loss."""
        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']
        # WAS (pre-0903): also split off an `action_rng` here to draw its OWN a' --
        #   rng, n1_rng, n2_rng, t1_rng, t2_rng, action_rng = jax.random.split(rng, 6)
        rng, n1_rng, n2_rng, t1_rng, t2_rng = jax.random.split(rng, 5)

        states = batch['states']
        next_states = batch['next_states']

        # WAS (pre-0903): built locally, duplicating joint_critic_bcfm_loss --
        #   actions_oh = jax.nn.one_hot(batch['actions'], A)
        #   joint_actions = actions_oh.reshape(batch_size, K * A)
        #   next_obs_with_id = batch_concat_agent_id_to_obs(batch['next_observations'])
        #   next_joint_actions = self._select_next_joint_action_teacher(
        #       next_obs_with_id, next_states, batch['next_legals'], action_rng)

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

    def q_head_factorization_loss(self, batch, grad_params, rng, bootstrap_target):
    ## WEIGHTED PROJECTION of V_joint = E[Z_joint(s,a)] onto the QMIX-factored class.
    ##
    ##   Q^mix = argmin_{q in Q_mono}  sum_a nu(a|s) w(s,a) (q(s,a) - Vhat(s,a))^2
    ##
    ## nu = {dataset action} U {N_q BC-flow candidates}; w from `w_temp` (see header note).
    ## Trained at BOTH the dataset action AND the candidates so the per-agent heads can
    ## RANK off-dataset (but in-support) actions -- makes rejection sampling effective.

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
            ## Returns BOTH the per-agent heads (B, K) and the mixed joint value (B, 1).
            # WAS (pre-0903): returned only the mixer output and threw q_pa away, so the
            # diagnostics block below re-issued byte-identical q_heads calls (1 for the
            # dataset action + N_q for the candidates) to recover it. Hand it back
            # instead. XLA CSE may well have been deduplicating these already -- this is
            # for clarity, do not budget a speedup against it.
            #   return self.network.select('mixer')(q_pa, states, params=grad_params)
            q_pa = self.network.select('q_heads')(obs_with_id, a_oh, params=grad_params) # (B, K) - per-agent
            q_tot = self.network.select('mixer')(q_pa, states, params=grad_params) #(B, 1)
            return q_pa, q_tot


        # ---------------- Dataset action (the in-support anchor) ----------------
        # Deliberately UNWEIGHTED: this is the one action where the teacher is not
        # extrapolating, so it always carries full weight. The weighting below triages
        # among extrapolated points only.
        V_data = teacher_V(actions_oh, q_rng)
        # WAS (pre-0903): Q_data = student_Q(actions_oh)
        q_per_agent, Q_data = student_Q(actions_oh)   # (B, K), (B, 1)
        loss_data = ((Q_data - V_data) ** 2).mean()

        # ---------------- BC-sampled candidate actions ----------------
        N_q = self.config['num_q_candidates']
        # WAS (pre-0903): cand_rng seeded BOTH the candidate action noise and the
        # teacher's MC readout noise, so V_cand's estimation error was correlated with
        # which candidate had been drawn --
        #   cand_keys = jax.random.split(cand_rng, N_q)
        #   _, cand_oh = self._sample_candidates(
        #       obs_with_id, batch['legals'], cand_rng, N_q)
        cand_rng, mc_rng = jax.random.split(cand_rng)
        cand_keys = jax.random.split(mc_rng, N_q)
        _, cand_oh = self._sample_candidates(
            obs_with_id, batch['legals'], cand_rng, N_q)        # (N_q, B, K, A)
        V_cand = jax.vmap(teacher_V)(cand_oh, cand_keys)          # (N_q, B, 1)
        # WAS (pre-0903): Q_cand = jax.vmap(student_Q)(cand_oh)
        q_pa_cand, Q_cand = jax.vmap(student_Q)(cand_oh)         # (N_q,B,K), (N_q,B,1)

        V_flat = V_cand.squeeze(-1)                              # (N_q, B)
        Q_flat = Q_cand.squeeze(-1)                              # (N_q, B) -> This value lives in the Q_mix set
                                                                 # Follows conditions of monotonicity
        V_cand_std_per_state = V_flat.std(axis=0)                # (B,)

        w_mode = self.config.get('w_mode', 'uniform')
        w_alpha = self.config.get('w_alpha', 0.1)

        if w_mode == 'uniform':
            # Exact pre-WQMIX behaviour.
            w_raw = jnp.ones_like(V_flat)

        elif w_mode == 'soft':
            # OURS. Standardize per state so w_temp is scale-free ("std devs of candidate
            # spread") and does not sharpen as returns grow during training.
            # w_temp -> inf recovers 'uniform'; w_temp -> 0 recovers hard-argmax ('cw').
            w_temp = self.config['w_temp']
            V_z = (V_flat - V_flat.mean(axis=0, keepdims=True)) / (
                V_cand_std_per_state[None, :] + 1e-6)
            w_raw = jax.nn.softmax(V_z / w_temp, axis=0) * N_q      # (N_q, B)

        elif w_mode == 'ow':
            # OW-QMIX (Rashid et al. 2020), Optimistically-Weighted. Full weight where
            # the UNRESTRICTED teacher exceeds the monotonic student -- i.e. where the
            # projection is currently underestimating and should be corrected.
            w_raw = jnp.where(V_flat > Q_flat, 1.0, w_alpha)        # (N_q, B)

        elif w_mode == 'cw':
            # CW-QMIX (Rashid et al. 2020), Centrally-Weighted. Ported against the
            # reference implementation, oxwhirl/wqmix src/learners/max_q_learner.py:
            #     is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
            #     max_action_qtot = self.target_central_mixer(
            #         central_target_max_agent_qvals[:, :-1], batch["state"][:, :-1])
            #     qtot_larger = targets > max_action_qtot
            #     ws = th.where(is_max_action | qtot_larger, ones, ws)
            # i.e.  w = 1  where  [a == argmax_a Qhat*(s,a)]  OR  [y > Qhat*(s, ahat*)].
            # Qhat* is the UNRESTRICTED central critic, which here is the teacher V.
            #
            # WAS (0831 + 0903-early), NOT faithful on either clause --
            #   star_idx = jnp.argmax(Q_flat, axis=0)      # STUDENT's argmax, not teacher's
            #   star = jax.nn.one_hot(star_idx, N_q).T
            #   V_star_student = V_flat[star_idx, jnp.arange(batch_size)]
            #   better_than_star = V_flat > V_star_student[None, :]   # per-CANDIDATE, no y
            #   w_raw = jnp.where((star > 0) | better_than_star, 1.0, w_alpha)
            # Taking the argmax from the student made the student choose where it got
            # supervised, and clause 2 compared candidates against the teacher's value
            # AT the student's pick rather than against y. Those two departures were
            # entangled: with clause 1 fixed to the teacher, V_flat > max_a V_flat is
            # never true, so the old clause 2 would go vacuous. Both are fixed here.
            #
            # UNAVOIDABLE ADAPTATION: the reference weights the ONE dataset action per
            # transition, so its `is_max_action` is a scalar test. We weight a candidate
            # SET (rejection sampling needs ordinal accuracy across it -- see the header
            # note), so clause 1 becomes "this candidate is the teacher's argmax over the
            # N_q candidates" and the argmax is over the sampled set rather than the full
            # action space. Clause 2 stays per-STATE, exactly as `qtot_larger` is in the
            # reference: when y beats the teacher's best candidate the WHOLE state gets
            # full weight.
            star_idx = jnp.argmax(V_flat, axis=0)                  # (B,) teacher's central action
            star = jax.nn.one_hot(star_idx, N_q).T                 # (N_q, B)
            V_star = V_flat.max(axis=0)                            # (B,) = Qhat*(s, ahat*)
            y = jax.lax.stop_gradient(bootstrap_target).squeeze(-1)  # (B,)
            target_larger = jnp.broadcast_to(
                (y > V_star)[None, :], V_flat.shape)               # (N_q, B), per-state
            w_raw = jnp.where((star > 0) | target_larger, 1.0, w_alpha)

        else:
            raise ValueError(f'unknown w_mode: {w_mode!r}')

        # Renormalize to mean 1 over the candidate axis. This matters for a FAIR
        # ablation: with alpha=0.1 the raw binary weights average ~0.2-0.5, so an
        # un-normalized 'ow'/'cw' run would have a 2-5x smaller effective learning rate
        # than 'soft' and you would be comparing step sizes, not weighting schemes.
        # Set w_normalize=False for paper-faithful absolute weights.
        if self.config.get('w_normalize', True):
            w_cand = w_raw * N_q / (w_raw.sum(axis=0, keepdims=True) + 1e-12)
        else:
            w_cand = w_raw
        w_cand = jax.lax.stop_gradient(w_cand)
        w_frac_full = (w_raw >= 1.0 - 1e-6).mean()   # binary modes: fraction at full weight

        loss_cand = (w_cand[..., None] * (Q_cand - V_cand) ** 2).mean()

        loss = loss_data + loss_cand

        # ---------------------------------------------------------------- #
        #                          DIAGNOSTICS                             #
        # ---------------------------------------------------------------- #
        # WAS (pre-0903): both of these re-ran q_heads on inputs student_Q had already
        # evaluated. They now come back from student_Q above.
        #   q_per_agent = self.network.select('q_heads')(
        #       obs_with_id, actions_oh, params=grad_params)         # (B, K)
        #   # Per-agent q over the candidate set: (N_q, B, K)
        #   q_pa_cand = jax.vmap(
        #       lambda a: self.network.select('q_heads')(obs_with_id, a, params=grad_params)
        #   )(cand_oh)
        # q_per_agent : (B, K)      -- from student_Q(actions_oh)
        # q_pa_cand   : (N_q, B, K) -- from vmap(student_Q)(cand_oh)

        # --- Effective sample size of the weights: N_q = uniform, 1 = collapsed to
        # hard-argmax (CW-QMIX). Tells you what the weighting is ACTUALLY doing.
        # Kish ESS = (sum w)^2 / sum w^2, computed on SELF-NORMALIZED weights so it is
        # scale-invariant and stays in [1, N_q] whether or not w_normalize is on. The
        # old form divided by a hardcoded N_q, which only holds when w has mean 1 --
        # with w_normalize=False and 'ow' weights of ~0.1 it read 334 for N_q=4.
        ess = ((jnp.sum(w_cand, axis=0) ** 2)
               / (jnp.sum(w_cand ** 2, axis=0) + 1e-12)).mean()

        # --- Argmax agreement (existing metrics, kept) ---
        teacher_best = jnp.argmax(V_flat, axis=0)                # (N_q,B) -> (B,)
        stitched_best = jnp.argmax(q_pa_cand, axis=0)            # (B, K) per-agent pick
        stitched_matches_teacher = (stitched_best == teacher_best[:, None]).all(axis=-1).mean()
        student_best = jnp.argmax(Q_flat, axis=0)                # (B,)
        student_matches_teacher = (student_best == teacher_best).mean()

        # REMOVED 8/31: norm_resid_cand / norm_resid_data and student_regret /
        # stitched_regret. All four were batch means of per-state RATIOS whose
        # denominators (per-state candidate var/std) get arbitrarily small when the
        # BC flow returns near-duplicate candidates -- measured on the 0827 500k
        # checkpoint, min per-state var 4.5e-5 against a 1e-6 floor, i.e. ~2e4
        # amplification from a single state. Observed swinging three orders of
        # magnitude between consecutive log points. stitched_regret was additionally
        # biased: V_star is an argmax over MC-noisy values (optimistic) while
        # V_stitched came from an independent draw. Ordinal metrics below are
        # bounded and carry the same signal.

        # --- Spearman rank correlation between student and teacher over candidates.
        # This is the ordinal signal rejection sampling actually consumes. Crude at
        # N_q=4 but monotone in what we care about.
        rV = _rank_along_axis0(V_flat)
        rQ = _rank_along_axis0(Q_flat)
        rV = rV - rV.mean(axis=0, keepdims=True)
        rQ = rQ - rQ.mean(axis=0, keepdims=True)
        spearman = ((rV * rQ).sum(axis=0)
                    / (jnp.sqrt((rV ** 2).sum(axis=0) * (rQ ** 2).sum(axis=0)) + 1e-6)).mean()

        # does each agent's head actually separate the candidates? If this collapses
        # toward 0, the execution-time argmax is noise.
        q_cand_spread_per_agent = q_pa_cand.std(axis=0).mean()

        assert q_per_agent.shape == (batch_size, K)
        assert V_data.shape == (batch_size, 1) and Q_data.shape == (batch_size, 1)
        assert w_cand.shape == (N_q, batch_size)

        return loss, {
            'q_head_loss': loss,
            'q_head_loss_data': loss_data,
            'q_head_loss_cand': loss_cand,
            'v_teacher_mean': V_data.mean(),
            'q_tot_factor_mean': Q_data.mean(),
            'q_per_agent_mean': q_per_agent.mean(),
            'q_per_agent_std_across_agents': q_per_agent.std(axis=-1).mean(),
            'v_cand_std': V_cand.std(),   # teacher spread across candidates = ranking signal
            'v_cand_std_per_state': V_cand_std_per_state.mean(),
            'q_cand_spread_per_agent': q_cand_spread_per_agent,
            # --- WQMIX weighting ---
            'w_ess': ess,                     # N_q = uniform, 1 = hard argmax
            'w_max': w_cand.max(axis=0).mean(),
            'w_min': w_cand.min(axis=0).mean(),
            'w_frac_full': w_frac_full,       # binary modes: fraction at full weight
            # --- ranking quality (what actually matters) ---
            'spearman_q_vs_v': spearman,
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

        # NOTE (8/31): this loss is CORRECT and is being correctly minimized. The issue is
        # architectural, not in the objective: ActorVectorField processes the K axis
        # independently, so the minimizer is v*_i = E[a_i - eps_i | x_i, t, o_i] -- the
        # conditioning set excludes o_j and x_j. Each MARGINAL mu_i(a_i|o_i) is exact; the
        # sampler is their PRODUCT. bc_loss therefore carries no information about joint
        # sample quality and can descend forever while candidates are uncoordinated.

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
        # WAS (pre-0903):
        #   rng, q_head_rng, bcfm_rng, dcfm_rng, actor_rng = jax.random.split(rng, 5)
        #   ... and joint_critic_bcfm_loss / joint_critic_dcfm_loss EACH called
        #   _select_next_joint_action_teacher on the identical batch with a different
        #   key. That is the single most expensive op in the step (N=8 candidates x
        #   num_flow_steps Euler steps x B x K actor-BC-flow evals) and it ran twice.
        #
        # Two reasons to draw a' ONCE and share it, beyond the ~30% step-time saving:
        #
        #   1. a' is not a random variable the objective integrates over. It is an
        #      ESTIMATOR of the greedy successor action -- the multi-agent stand-in for
        #      argmax_{a'} Q(s',a'), approximated by an argmax over BC-flow candidates.
        #      Its randomness is estimation error, not part of the target distribution.
        #      BCFM (mean-matching) and DCFM (distributional) are two views of ONE
        #      Bellman operator T: Z(s,a) <- r + gamma*m*Z_target(s',a'). Feeding them
        #      different a' means the critic is pushed toward the average of T^{a'_1}
        #      and T^{a'_2}, two operators differing by exactly the candidate-selection
        #      noise -- on most states the two argmaxes disagree.
        #   2. a' is an argmax over MC-noisy teacher values (teacher_mc_samples draws),
        #      so it carries the usual optimism of a max over noisy estimates. Two
        #      independent optimistic draws summed into one gradient pays that bias
        #      twice per transition per step, working against ret_agg/q_agg='min'.
        #
        # Note this does NOT touch the deliberate per-twin decorrelation inside each
        # loss (noises1/noises2, times1/times2) -- the twins already shared a single
        # next_joint_actions within each loss, so one draw per transition was already
        # the design intent; only the split ACROSS the two losses is removed.
        rng, q_head_rng, bcfm_rng, dcfm_rng, actor_rng, action_rng, y_rng = \
            jax.random.split(rng, 7)

        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']
        actions_oh = jax.nn.one_hot(batch['actions'], A)
        joint_actions = actions_oh.reshape(batch_size, K * A)

        next_obs_with_id = batch_concat_agent_id_to_obs(batch['next_observations'])
        next_joint_actions = self._select_next_joint_action_teacher(
            next_obs_with_id, batch['next_states'], batch['next_legals'], action_rng)

        # Bootstrap target y = r + gamma * mask * E[Z_target(s', a')], built ONCE.
        # Hoisted out of joint_critic_bcfm_loss (0903) because the paper-faithful
        # CW-QMIX condition in q_head_factorization_loss compares against y, and
        # integrating the two target ODEs a second time for it would be waste.
        next_states = batch['next_states']
        next_q_noises = jax.random.normal(y_rng, (batch_size, 1))
        next_z1 = self.compute_flow_returns(
            next_q_noises, next_states, next_joint_actions,
            flow_network_name='target_joint_critic_flow1')
        next_z2 = self.compute_flow_returns(
            next_q_noises, next_states, next_joint_actions,
            flow_network_name='target_joint_critic_flow2')
        next_returns = (jnp.minimum(next_z1, next_z2)
                        if self.config['ret_agg'] == 'min'
                        else (next_z1 + next_z2) / 2)
        next_returns = jax.lax.stop_gradient(next_returns)
        bootstrap_target = (jnp.expand_dims(batch['rewards'], axis=-1)
                            + self.config['discount']
                            * jnp.expand_dims(batch['masks'], axis=-1)
                            * next_returns)                       # (B, 1)

        q_head_loss, q_head_info = self.q_head_factorization_loss(
            batch, grad_params, q_head_rng, bootstrap_target)
        bcfm_loss, bcfm_info = self.joint_critic_bcfm_loss(
            batch, grad_params, bcfm_rng, joint_actions, next_joint_actions,
            bootstrap_target)
        dcfm_loss, dcfm_info = self.joint_critic_dcfm_loss(
            batch, grad_params, dcfm_rng, joint_actions, next_joint_actions)
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
            
            num_candidates = 4, # Number of candidate actions for rejection sampling
            num_q_candidates = 4, # Number of sampled candidates to train q-heads
            num_target_candidates = 4,
            teacher_mc_samples = 4, # Number of samples for teacher V estimation. 
                                    # This should be monotone, higher means more 

            t_embed_frequencies = 8, # Number of frequencies for sinusoidal time embedding
            target_select_full_ode = False,  # Whether to use full ODE integration for target selection

            w_mode = 'uniform', # or "soft" | "ow" | "cw"
            w_temp = 1.0,
            w_alpha = 0.1,
            w_normalize = False,
        )
    )
    return config
