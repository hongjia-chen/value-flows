import copy
from functools import partial
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
#       Train joint distributional critic -> Then factorize to q values per agent using QMIX
#       (unweighted; the WQMIX candidate weights of 0831-0917 were removed 10/19)
#       
#           Distill the flow critic into student flows 
# 2. Actor Architecture: per-agent BC policy flow
# 3. Action Discretization and Representation
#       One-hot encoding of discrete actions for SMAC
#       Distilled BC Flow transports \eps to logits in R^9
#       Argmax(logits) to get discrete actions
#       Legal-action masking: -inf on illegal logits
# 3. Q-guidance: Softmax (relaxed) to allow Q-loss gradients to flow to actors (?)



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
    """QMIX monotonic mixer, no time axis. q_vals (B,K), states (B,state_dim) -> (B,1).

    monotone=False drops the |w| (T1 arm A3, diagnostic only): IGM no longer holds, so
    the per-agent argmax deployed by sample_actions_rejection is invalid for that arm.
    """
    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float
    monotone: bool = True

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
        if self.monotone:
            w_1, w_2 = jnp.abs(w_1), jnp.abs(w_2)
        w_1 = w_1.reshape(B, K, self.embedding_dim)            # (B, K, E)
        b_1 = b_1.reshape(B, 1, self.embedding_dim)            # (B, 1, E)
        w_2 = w_2.reshape(B, self.embedding_dim, 1)            # (B, E, 1)
        b_2 = b_2.reshape(B, 1, 1)                             # (B, 1, 1)

        hidden = nn.elu(jnp.matmul(q_vals[:, None, :], w_1) + b_1)  # (B,1,E)
        q_tot = jnp.matmul(hidden, w_2) + b_2                       # (B,1,1)
        return q_tot.reshape(B, 1)


def _rank_along_axis0(x):
    """Ranks (0..n-1) of x along axis 0. Ties broken arbitrarily (fine for N_q small)."""
    return jnp.argsort(jnp.argsort(x, axis=0), axis=0).astype(x.dtype)


def aggregate_twins(x1, x2, method):
    """Combine the two twin critics' outputs: 'min' (pessimistic) or 'mean'."""
    if method == 'min':
        return jnp.minimum(x1, x2)
    if method == 'mean':
        return (x1 + x2) / 2
    raise ValueError(f'unknown twin aggregation: {method!r}')


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

    def _teacher_readout(self, states, a_oh, key):
        """The q_head teacher's decision readout of the joint action a_oh (B, K, A): the
        one-step E[Z | eps] = eps + v(eps, 0) of both TARGET twins, clipped to the return
        bounds, averaged over L = teacher_mc_samples draws of eps from `key`, then combined
        by q_agg. Pass one key for every option compared (CRN). Returns (V, V1, V2), (B, 1)."""
        B = states.shape[0]
        K, A = self.config['num_agents'], self.config['action_dim']
        joint_a = a_oh.reshape(B, K * A)
        L = self.config['teacher_mc_samples']
        eps = jax.random.normal(key, (L, B, 1))   # L independent draws
        return_min = self.config['min_reward'] / (1 - self.config['discount'])
        return_max = self.config['max_reward'] / (1 - self.config['discount'])

        def one_step(e):
            z1 = e + self.network.select('target_joint_critic_flow1')(
                e, jnp.zeros_like(e), states, joint_a)
            z2 = e + self.network.select('target_joint_critic_flow2')(
                e, jnp.zeros_like(e), states, joint_a)
            if self.config['clip_flow_returns']:
                z1 = jnp.clip(z1, return_min, return_max)
                z2 = jnp.clip(z2, return_min, return_max)
            return z1, z2

        z1, z2 = jax.vmap(one_step)(eps)          # each (L, B, 1)
        z1 = z1.mean(axis=0)                       # average over MC draws -> (B, 1)
        z2 = z2.mean(axis=0)
        return aggregate_twins(z1, z2, self.config['q_agg']), z1, z2

    def _ode_readout(self, states, a_oh, key):
        """Full-ODE mean of Z(s, a): L = teacher_mc_samples endpoints of the 10-step ODE
        from both TARGET twins, combined by ret_agg -- the readout the bootstrap target y
        draws from (total_loss). (B, 1)."""
        B = states.shape[0]
        K, A = self.config['num_agents'], self.config['action_dim']
        joint_a = a_oh.reshape(B, K * A)
        eps = jax.random.normal(key, (self.config['teacher_mc_samples'], B, 1))

        def one(e):
            z1 = self.compute_flow_returns(e, states, joint_a,
                                           flow_network_name='target_joint_critic_flow1')
            z2 = self.compute_flow_returns(e, states, joint_a,
                                           flow_network_name='target_joint_critic_flow2')
            return aggregate_twins(z1, z2, self.config['ret_agg'])
        return jax.vmap(one)(eps).mean(0)

    @jax.jit
    def eval_step(self, obs_with_id, legals, states, rng):
        """One eval decision: each eval arm's joint action, plus the readouts the on-policy
        diagnostics need (utils/evaluate_smac.py). B = 1 at eval.

        The pool and 'deployed' are exactly sample_actions_rejection(obs_with_id, legals,
        rng), so the deployed arm plays the same actions whether or not it is diagnosed.
        Arms: 'deployed' (per-agent argmax q_i), 'bc' (candidate 0 == one BC draw, the
        no-selection floor) and 'teacher' (argmax of the joint critic over the same N
        candidates; it reads the global state, so it is a centralised reference, not a
        deployable policy). V* are the teacher readout, one eps set for every option (CRN,
        key fold_in(rng, 1)); V_ode is the full-ODE mean (key fold_in(rng, 2)).
        """
        N = self.config['num_candidates']
        A = self.config['action_dim']
        B = obs_with_id.shape[0]
        cand_int, cand_oh = self._sample_candidates(obs_with_id, legals, rng, N)
        q = jax.vmap(lambda a: self.network.select('q_heads')(obs_with_id, a))(cand_oh)  # (N, B, K)
        deployed = jnp.take_along_axis(cand_int, jnp.argmax(q, axis=0)[None], axis=0).squeeze(0)

        k_teach, k_ode = jax.random.fold_in(rng, 1), jax.random.fold_in(rng, 2)
        V, V1, V2 = jax.vmap(lambda a: self._teacher_readout(states, a, k_teach))(cand_oh)
        V_ode = jax.vmap(lambda a: self._ode_readout(states, a, k_ode))(cand_oh)
        dep_oh = jax.nn.one_hot(deployed, A)
        V_dep = self._teacher_readout(states, dep_oh, k_teach)[0]
        V_dep_ode = self._ode_readout(states, dep_oh, k_ode)
        teacher = cand_int[jnp.argmax(V[..., 0], axis=0), jnp.arange(B)]
        return dict(cand=cand_int, deployed=deployed, bc=cand_int[0], teacher=teacher,
                    V=V[..., 0], V1=V1[..., 0], V2=V2[..., 0], V_ode=V_ode[..., 0],
                    V_dep=V_dep[:, 0], V_dep_ode=V_dep_ode[:, 0])

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
                return aggregate_twins(z1, z2, self.config['ret_agg'])
            return jax.vmap(one)(eps).mean(0)                 # (B, 1)

        # keys = jax.random.split(score_rng, N)
        v_cand = jax.vmap(score, in_axes=(0, None))(cand_oh, score_rng).squeeze(-1)   # (N, B)
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
        """Compute returns from the return flow model using the Euler method.

        clip_flow_returns clips only the t=1 endpoint, which is a return sample and so
        lies in [min_reward, max_reward] / (1 - discount). A partial integration (an
        explicit end_times, i.e. DCFM's x_t = t*Z + (1-t)*eps) is returned unclipped:
        x_t is not a return and has no such bound.
        """
        integrates_to_endpoint = end_times is None
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

            return (new_noisy_returns, new_noisy_jac_eps_prod), None

        # Use lax.scan to do the iteration
        (noisy_returns, noisy_jac_eps_prod), _ = jax.lax.scan(
            func, (noisy_returns, noisy_jac_eps_prod), jnp.arange(self.config['num_flow_steps']))

        if self.config['clip_flow_returns'] and integrates_to_endpoint:
            noisy_returns = jnp.clip(
                noisy_returns,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )

        if return_jac_eps_prod:
            return noisy_returns, noisy_jac_eps_prod
        else:
            return noisy_returns

    ## Full ODE integration instead of one-step
    # One-step conditional mean E[Z | ε], ε is noise at t = 0

        # Check total_loss() for complete implementation.
        # Drawing a' once and sharing it, beyond the ~30% step-time saving:
        #
        # a' is not a random variable the objective integrates over. It is an
        #      ESTIMATOR of the greedy successor action -- the multi-agent stand-in for
        #      argmax_{a'} Q(s',a'), approximated by an argmax over BC-flow candidates.
        #      Its randomness is estimation error, not part of the target distribution.
        #      BCFM (mean-matching) and DCFM (distributional) are two views of ONE
        #      Bellman operator T: Z(s,a) <- r + gamma*m*Z_target(s',a').


    def joint_critic_bcfm_loss(self, batch, grad_params, noises, times,
                               joint_actions, next_joint_actions, bootstrap_target):
        """BCFM half of the Bellman backup. `noises`, `times`, `joint_actions`,
        `next_joint_actions` and `bootstrap_target` (= y) are computed once in total_loss
        and passed in -- see the note there. `noises` is the eps that drew y.
        `next_joint_actions` is unused here: s' and a' reach BCFM only through y."""

        batch_size = batch['actions'].shape[0]
        states = batch['states']
        returns = bootstrap_target

        # Value Flows noise sharing (value_flows.py:64-88; paper App C.1 fn 4): each path
        # runs from eps to the return that eps produced, and both twins regress at that
        # same point. Before 10/19 each twin drew its own (eps, t), independent of y's eps.
        noisy_returns = times * returns + (1 - times) * noises
        target_vf = returns - noises

        vf1 = self.network.select('joint_critic_flow1')(
            noisy_returns, times, states, joint_actions, params=grad_params)
        vf2 = self.network.select('joint_critic_flow2')(
            noisy_returns, times, states, joint_actions, params=grad_params)

        bcfm_loss = ((vf1 - target_vf) ** 2 + (vf2 - target_vf) ** 2).mean()

        assert returns.shape == (batch_size, 1)
        assert noisy_returns.shape == (batch_size, 1)
        assert vf1.shape == (batch_size, 1) and vf2.shape == (batch_size, 1)

        # vf_twin_std: both twins regress at the same point since 10/19, so this is a real
        # twin disagreement (in velocity units). Before 10/19 it sat near sqrt(2) by
        # construction. Removed 10/19: returns_std, vf1_mean, vf2_mean.
        return bcfm_loss, {
            'bcfm_loss': bcfm_loss,
            'returns_mean': returns.mean(),   # mean bootstrap target y
            'vf_twin_std': (vf1 - vf2).std(),
        }


    ## Training stage, can use centralized state
    # Therefore, better to not just rejection sample individual actions and join them with IGM
    # The stitched up rejection sampling action also likely falls out of the support of the joint actions in dataset
    # Use joint actions, scored by joint critic E[Z_joint(s,a)]
    def joint_critic_dcfm_loss(self, batch, grad_params, noises, times,
                               joint_actions, next_joint_actions):
        """DCFM half of the Bellman backup. Shares `noises`, `times` and
        `next_joint_actions` with joint_critic_bcfm_loss -- see the note in total_loss."""
        batch_size = batch['actions'].shape[0]
        states = batch['states']
        next_states = batch['next_states']

        # DCFM target, as in Value Flows (value_flows.py:92-123) except for the gamma*m
        # factor on the target velocity (10/21, below): the shared (eps, t) is
        # integrated to t through BOTH target flows and combined by dcfm_agg, and the
        # target velocity is dcfm_agg over both target fields at that point. Both online
        # twins regress at the same point. Before 10/19 each twin drew its own (eps, t).
        dcfm_agg = self.config['dcfm_agg']
        target_names = ('target_joint_critic_flow1', 'target_joint_critic_flow2')
        znext = aggregate_twins(*[
            self.compute_flow_returns(noises, next_states, next_joint_actions,
                                      end_times=times, flow_network_name=n)
            for n in target_names], dcfm_agg)
        target_vf_next = aggregate_twins(*[
            self.network.select(n)(znext, times, next_states, next_joint_actions)
            for n in target_names], dcfm_agg)

        r_ = jnp.expand_dims(batch['rewards'], -1)
        m_ = jnp.expand_dims(batch['masks'], -1)
        noisy_returns = r_ + self.config['discount'] * m_ * znext
        # The regression point r + gamma*m*znext moves at gamma*m times the target field's
        # velocity (chain rule), so that is the target. At a terminal (m=0) the point is
        # parked at r and the target is 0, which agrees with BCFM's point mass y = r.
        # WAS (to 10/21, and value_flows.py:115-123): the bare target field, no gamma*m.
        # At terminals, where next_states = states (envs/smac_utils.py:205), that asked
        # the field to move like "keep playing from s" at the point where it should stop.
        target_vf = jax.lax.stop_gradient(self.config['discount'] * m_ * target_vf_next)

        vf1 = self.network.select('joint_critic_flow1')(
            noisy_returns, times, states, joint_actions, params=grad_params)
        vf2 = self.network.select('joint_critic_flow2')(
            noisy_returns, times, states, joint_actions, params=grad_params)

        dcfm_loss = ((vf1 - target_vf) ** 2 + (vf2 - target_vf) ** 2).mean()

        assert znext.shape == (batch_size, 1)
        assert noisy_returns.shape == (batch_size, 1)
        assert vf1.shape == (batch_size, 1) and vf2.shape == (batch_size, 1)
        assert target_vf.shape == (batch_size, 1)

        # Removed 10/19: noisy_next_returns_mean, noisy_returns_mean, target_vf_mean,
        # vf1_mean, vf2_mean -- means of intermediate points and velocities at a random t,
        # with no reading as a value.
        return dcfm_loss, {
            'dcfm_loss': dcfm_loss,
            'vf_twin_std': (vf1 - vf2).std(),
        }

    def q_head_factorization_loss(self, batch, grad_params, rng):

        # Originally, Q-head network is only trained on one joint action from the offline dataset (given any state)
        # Rejection sampling is ranking samples from BC flow -> can be off-dataset joint actions -> Getting 4 samples from BC flow

        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']
        # The middle key was q_rng, V_data's own draws, until 10/17 (see V_data below). It
        # is still split off so cand_rng, and with it every candidate draw, is unchanged.
        rng, _, cand_rng = jax.random.split(rng, 3)

        states = batch['states']
        obs_with_id = batch_concat_agent_id_to_obs(batch['observations'])     # (B, K, obs_dim + K)
        actions_oh = jax.nn.one_hot(batch['actions'], A)

        def teacher_V(a_oh, key):
            ## Expectation of joint return distribution E[Z_joint(s,a)]; using 4 samples for some variance reduction
            ## Returns (V, V1, V2): q_agg of the twins, and each twin's own readout (diagnostics).
            return jax.lax.stop_gradient(self._teacher_readout(states, a_oh, key))

        def student_Q(a_oh):
            ## factorizes into per-agent q_i, with local observations; Q-MIX
            ## Returns BOTH the per-agent heads (B, K) and the mixed joint value (B, 1).
            q_pa = self.network.select('q_heads')(obs_with_id, a_oh, params=grad_params) # (B, K) - per-agent
            q_tot = self.network.select('mixer')(q_pa, states, params=grad_params) #(B, 1)
            return q_pa, q_tot


        # ---------------- Dataset action (the in-support anchor) ----------------
        # The one action where the teacher is not extrapolating.
        # 0903: Split cand_rng into two keys, one for candidate sampling and one for teacher evaluation?
        cand_rng, mc_rng = jax.random.split(cand_rng)
        # The dataset action is scored with the candidates' L draws (CRN), so V_data and
        # V_cand differ only through the actions: a candidate equal to the dataset action
        # gets exactly V_data.
        # WAS (pre-10/17): its own L draws, so every V_data - V_cand contrast carried
        # independent MC noise --
        #   V_data = teacher_V(actions_oh, q_rng)
        V_data = teacher_V(actions_oh, mc_rng)[0]
        q_per_agent, Q_data = student_Q(actions_oh)   # (B, K), (B, 1)
        loss_data = ((Q_data - V_data) ** 2).mean()

        # ---------------- BC-sampled candidate actions ----------------
        N_q = self.config['num_q_candidates']
        # cand_keys = jax.random.split(mc_rng, N_q)
        _, cand_oh = self._sample_candidates(obs_with_id, batch['legals'], cand_rng, N_q)        # (N_q, B, K, A)
        ## Not splitting keys by each sample
        V_cand, V1_cand, V2_cand = jax.vmap(teacher_V, in_axes=(0, None))(cand_oh, mc_rng)  # (N_q, B, 1) each
        q_pa_cand, Q_cand = jax.vmap(student_Q)(cand_oh)         # (N_q,B,K), (N_q,B,1)

        V_flat = V_cand.squeeze(-1)                              # (N_q, B)
        Q_flat = Q_cand.squeeze(-1)                              # (N_q, B) -> This value lives in the Q_mix set
                                                                 # Follows conditions of monotonicity

        loss_cand = ((Q_cand - V_cand) ** 2).mean()

        loss = loss_data + loss_cand

        # T1 (extraction_tests_spec.md): within-state fit. Uniform MSE spends ~99% of its
        # weight on per-state levels, while every argmax downstream reads only the
        # within-state deviations. loss_within == 1 - R^2_within on the batch, so
        # within_lambda re-weights the objective toward the decision-relevant part.
        # Identical candidates give dV = dQ = 0 exactly, so they drop out on their own.
        dV = V_flat - V_flat.mean(axis=0, keepdims=True)         # (N_q, B), teacher stop-grad
        dQ = Q_flat - Q_flat.mean(axis=0, keepdims=True)         # (N_q, B)
        dV_ss = jax.lax.stop_gradient((dV ** 2).sum())
        loss_within = ((dQ - dV) ** 2).sum() / (dV_ss + 1e-6)
        within_lambda = self.config.get('within_lambda', 0.0)
        if within_lambda:            # static; the default graph is untouched
            loss = loss + within_lambda * loss_within

        # Level / within-state split over N_q + 1 options, the dataset action being option
        # 0 and scored with the candidates' teacher draws (row 7). Per state,
        #   mean_n (Q_n - V_n)^2 = (Qbar - Vbar)^2 + mean_n (dQ_n - dV_n)^2,
        # since dQ and dV each sum to 0. q_loss='decomposed' trains on
        # q_level_lambda * level + within in place of the MSE above; q_level_lambda=1 is
        # the uniform MSE over the N_q + 1 options (the MSE above weights the dataset
        # action N_q times a candidate). Only the within part reaches a deployed argmax.
        # Unlike T1's loss_within this is not normalised, and it includes the data action.
        V_opt = jnp.concatenate([V_data[None], V_cand], axis=0).squeeze(-1)   # (N_q+1, B)
        Q_opt = jnp.concatenate([Q_data[None], Q_cand], axis=0).squeeze(-1)   # (N_q+1, B)
        V_bar, Q_bar = V_opt.mean(axis=0), Q_opt.mean(axis=0)                  # (B,)
        loss_level = ((Q_bar - V_bar) ** 2).mean()
        loss_within_opt = (((Q_opt - Q_bar) - (V_opt - V_bar)) ** 2).mean()
        q_loss = self.config.get('q_loss', 'mse')
        if q_loss == 'decomposed':
            if within_lambda:
                raise ValueError("q_loss='decomposed' replaces the MSE: use it with "
                                 "within_lambda=0")
            loss = self.config.get('q_level_lambda', 0.0) * loss_level + loss_within_opt
        elif q_loss != 'mse':
            raise ValueError(f'unknown q_loss: {q_loss!r}')

        # ---------------------------------------------------------------- #
        #                          DIAGNOSTICS                             #
        # ---------------------------------------------------------------- #
        # Rebuilt 10/19. The value chain behind every deployed action is
        #   pool (N BC candidates) -> teacher (joint critic's pick) -> Q_tot -> stitched,
        # and each group below measures one link on this batch's dataset states:
        #   pool/     is there anything to choose between?
        #   teacher/  what does the critic claim choosing is worth, and how sure is it?
        #   extract/  how much of the teacher's ranking survives into Q_tot and the
        #             per-agent (stitched, deployed) pick?
        # Rules, all fixed from the pre-10/19 metrics:
        #   - Compare ACTION ids, never candidate indices: duplicate candidates are common
        #     (29% of 5m6m states have all N identical), and index comparisons gave them
        #     free agreement (spearman_q_vs_v, *_matches_teacher).
        #   - Agreement and spread are averaged over DISTINCT states only (>= 2 different
        #     joint candidates); everywhere else every pick is the same action.
        #   - Top-1 agreement is chance-corrected against picking a uniformly random
        #     candidate: 0 = no better than one BC draw, 1 = always the teacher's pick.

        f32 = jnp.float32
        cand_act = jnp.argmax(cand_oh, axis=-1)                                  # (N_q,B,K)
        same = (cand_act[:, None] == cand_act[None]).all(-1)                     # (N_q,N_q,B)
        n_distinct = (1.0 / same.sum(1)).sum(0)                                  # (B,)
        dmask = (n_distinct > 1.5).astype(f32)                                   # distinct states
        dmean = lambda x: (x * dmask).sum() / (dmask.sum() + 1e-6)
        rows = jnp.arange(batch_size)
        pick = lambda scores: cand_act[jnp.argmax(scores, axis=0), rows]         # (B,K) actions

        # Teacher: its pick, each twin's pick, the gain it sees, its level vs the data.
        V1_flat, V2_flat = V1_cand.squeeze(-1), V2_cand.squeeze(-1)
        teach_act = pick(V_flat)
        V_best, V_mean = V_flat.max(0), V_flat.mean(0)
        same_joint = lambda x, y: (x == y).all(-1).astype(f32)                   # (..., B)

        # Stitched (deployed) and mixer picks, scored by the teacher with the same draws (CRN).
        stitched_best = jnp.argmax(q_pa_cand, axis=0)                            # (B, K)
        stitch_act = jnp.take_along_axis(cand_act, stitched_best[None], 0).squeeze(0)  # (B,K)
        V_stitch = teacher_V(jax.nn.one_hot(stitch_act, A), mc_rng)[0].squeeze(-1)
        student_best = jnp.argmax(Q_flat, axis=0)                                # (B,)
        V_mix = V_flat[student_best, rows]
        gain = (V_best - V_mean).sum() + 1e-6
        # Chance of a uniformly random candidate being the teacher's pick, per state and
        # per agent; top-1 numbers are reported against it.
        null_joint = same_joint(cand_act, teach_act[None]).mean(0)               # (B,)
        null_agent = (cand_act == teach_act[None]).astype(f32).mean(0)           # (B,K)

        def chance_corrected(hit, null):
            h, n = dmean(hit), dmean(null)
            return (h - n) / (1.0 - n + 1e-6)

        info = {
            'q_head_loss': loss,
            'q_head_loss_data': loss_data,
            'q_head_loss_cand': loss_cand,
            'loss_level': loss_level,
            'loss_within_opt': loss_within_opt,
            # --- pool: is there anything to choose between? ---
            'pool/frac_distinct': dmask.mean(),
            'pool/n_distinct': n_distinct.mean(),
            'pool/agent_data_in_pool': (cand_act == batch['actions'][None]).any(0).mean(),
            # --- teacher: what the critic claims choosing is worth, and how sure it is ---
            'teacher/v_data_mean': V_data.mean(),
            'teacher/gain': dmean(V_best - V_mean),
            'teacher/gain_vs_data': (V_best - V_data[:, 0]).mean(),
            # Chance-corrected like top1_*: twin 2's pick vs twin 1's, against a random candidate.
            'teacher/twin_top1_agree': chance_corrected(
                same_joint(pick(V2_flat), pick(V1_flat)),
                same_joint(cand_act, pick(V1_flat)[None]).mean(0)),
            # --- extract: teacher -> Q_tot (mix) -> per-agent argmax (stitch) ---
            'extract/r2_within': 1.0 - jax.lax.stop_gradient(loss_within),
            'extract/rho_within': (dQ * dV).sum() / (jnp.sqrt((dQ ** 2).sum() * dV_ss) + 1e-6),
            'extract/top1_mix': chance_corrected(same_joint(pick(Q_flat), teach_act), null_joint),
            'extract/top1_stitch': chance_corrected(same_joint(stitch_act, teach_act), null_joint),
            'extract/agent_agree': chance_corrected((stitch_act == teach_act).astype(f32).mean(-1),
                                                    null_agent.mean(-1)),
            'extract/capture_mix': (V_mix - V_mean).sum() / gain,
            'extract/capture_stitch': (V_stitch - V_mean).sum() / gain,
            'extract/in_pool': dmean((cand_act == stitch_act[None]).all(-1).any(0).astype(f32)),
        }
        if 'mc_returns' in batch:
            # The behaviour policy's discounted return-to-go from s (main.py adds it).
            G = batch['mc_returns']
            v = V_data[:, 0]
            info['teacher/value_bias_mc'] = (v - G).mean()
            info['teacher/value_corr_mc'] = (((v - v.mean()) * (G - G.mean())).mean()
                                             / (v.std() * G.std() + 1e-6))

        assert q_per_agent.shape == (batch_size, K)
        assert V_data.shape == (batch_size, 1) and Q_data.shape == (batch_size, 1)

        return loss, info


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
            monotone=config.get('mixer_monotone', True),
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
        # The 3rd and 4th keys drew BCFM's and DCFM's own (eps, t) until 10/19. They are
        # still split off, so every other draw matches earlier runs step for step.
        t_rng, q_head_rng, _, _, actor_rng, action_rng, y_rng = jax.random.split(rng, 7)

        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']
        actions_oh = jax.nn.one_hot(batch['actions'], A)
        joint_actions = actions_oh.reshape(batch_size, K * A)

        next_obs_with_id = batch_concat_agent_id_to_obs(batch['next_observations'])
        next_joint_actions = self._select_next_joint_action_teacher(
            next_obs_with_id, batch['next_states'], batch['next_legals'], action_rng)

        # Bootstrap target y = r + gamma * mask * E[Z_target(s', a')], built once.

        next_states = batch['next_states']
        next_q_noises = jax.random.normal(y_rng, (batch_size, 1))
        next_z1 = self.compute_flow_returns(
            next_q_noises, next_states, next_joint_actions,
            flow_network_name='target_joint_critic_flow1')
        next_z2 = self.compute_flow_returns(
            next_q_noises, next_states, next_joint_actions,
            flow_network_name='target_joint_critic_flow2')
        next_returns = aggregate_twins(next_z1, next_z2, self.config['ret_agg'])
        next_returns = jax.lax.stop_gradient(next_returns)
        bootstrap_target = (jnp.expand_dims(batch['rewards'], axis=-1)
                            + self.config['discount']
                            * jnp.expand_dims(batch['masks'], axis=-1)
                            * next_returns)                       # (B, 1)

        # Value Flows noise sharing (value_flows.py:64-123; paper App C.1 fn 4): y's eps and
        # one t are where BCFM and DCFM start, for both twins.
        times = jax.random.uniform(t_rng, (batch_size, 1))

        q_head_loss, q_head_info = self.q_head_factorization_loss(
            batch, grad_params, q_head_rng)
        bcfm_loss, bcfm_info = self.joint_critic_bcfm_loss(
            batch, grad_params, next_q_noises, times, joint_actions, next_joint_actions,
            bootstrap_target)
        dcfm_loss, dcfm_info = self.joint_critic_dcfm_loss(
            batch, grad_params, next_q_noises, times, joint_actions, next_joint_actions)
        a_loss, actor_info = self.actor_loss(
            batch, grad_params, actor_rng)

        # Combine with lambdas
        critic_loss = (self.config['bcfm_lambda'] * bcfm_loss
                    + self.config['dcfm_lambda'] * dcfm_loss)
        total = critic_loss + q_head_loss + a_loss

        info = {'total_loss': total, 'critic_loss': critic_loss}
        # Diagnostics that already carry a group ('pool/', 'teacher/', 'extract/') keep it.
        info.update({k if '/' in k else f'q_head/{k}': v for k, v in q_head_info.items()})
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

        # grad/norm is an L1 sum of per-leaf L2 norms over every module at once, so it
        # mixes the critic, q_head and actor losses. Per-module norms were tried 10/19 and
        # dropped: any extra reduction on the gradients changes XLA's fusion and moves the
        # mixer update by ~1 ulp, which breaks the bit-identity checks.
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

            ret_agg='mean',  # BCFM bootstrap target y, and a' scoring.
            dcfm_agg='mean',  # DCFM target point and velocity.
            q_agg='mean',  # q_head teacher V, i.e. what the deployed policy ranks by.
            clip_flow_actions=False,  # Whether to clip the intermediate flow actions.
            # T1 (extraction_tests_spec.md). Both defaults leave training unchanged.
            within_lambda=0.0,  # weight of the within-state (centred) q_head loss, = 1 - R^2_within
            mixer_monotone=True,  # False drops the mixer's |w| (diagnostic; breaks IGM)
            # q_head loss: 'mse' (dataset action + candidates, as since 0903) or 'decomposed'
            # (within-state MSE over the dataset action + candidates, + q_level_lambda * level).
            q_loss='mse',
            q_level_lambda=0.0,  # weight of the across-state level term under q_loss='decomposed'
            clip_flow_returns=True,  # Clip t=1 flow returns (ODE endpoint, one-step teacher V) to the return bounds.
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
        )
    )
    return config
