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

## MAV-Flow, per-agent critic variant.
#
# Difference from the joint variant:
#   joint:      Z_joint(s, a_joint) via flow  ->  distill into QMIX-factored q_i
#   per-agent:  Z_i(o_i, a_i) via SHARED flow ->  mix q_i with QMIX into Q_tot
#
# There is no per-agent reward in SMAC, so there is no per-agent return sample,
# so CFM has no regression target at the agent level. The critic is instead
# trained by SAMPLE MATCHING at the joint level:
#
#   eps ~ N(0,1) SHARED across agents
#   Z_tot(eps) = Mix_phi( Z_1(eps), ..., Z_K(eps), s )
#   G(eps)     = r + gamma * m * Z_tot^target(eps) at (s', a')
#   loss       = Huber( Z_tot(eps), sg[G(eps)] )
#
# Shared eps is load-bearing. In 1-D, ODE trajectories cannot cross, so each
# eps -> Z_i(eps) is monotone; the QMIX mixer is monotone by the |w| construction;
# the composition is therefore a valid 1-D transport map and eps is a shared
# quantile level. With independent eps_i the composition is not a transport map
# and the loss is ill-posed.
#
# Consequence: gradients must backprop through the ODE integrator. The one-step
# trick eps + v(eps, 0) = E[Z|eps] holds only because CFM regresses v to a
# conditional expectation; with no CFM loss it has no meaning here and is gone.
#
# Pessimism: twin flows AND twin mixers, min taken per-eps on the whole Q_tot
# (min OUTSIDE the mixer). Min inside would give Mix(min_j q^j) <= min_j Mix(q^j)
# by monotonicity -- strictly more pessimistic, compounding in K, and incoherent
# in the sense that no single critic endorses the resulting utility vector.


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

    use_bias=False drops b_1(s) and b_2(s). Since elu(0) = 0, this makes
    q = 0  =>  q_tot = 0, which removes the action-independent V(s) path that
    lets the mixer satisfy the TD target while the utilities collapse to zero.
    """
    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float
    use_bias: bool = True

    @nn.compact
    def __call__(self, q_vals, states):
        B, K = q_vals.shape

        w_1 = HyperNetwork(self.hypernet_hidden_dim, self.embedding_dim * K,
                           self.init_scale)(states)
        w_2 = HyperNetwork(self.hypernet_hidden_dim, self.embedding_dim,
                           self.init_scale)(states)

        w_1 = jnp.abs(w_1).reshape(B, K, self.embedding_dim)   # (B, K, E)
        w_2 = jnp.abs(w_2).reshape(B, self.embedding_dim, 1)   # (B, E, 1)

        hidden = jnp.matmul(q_vals[:, None, :], w_1)           # (B, 1, E)
        if self.use_bias:
            b_1 = nn.Dense(self.embedding_dim,
                           kernel_init=orthogonal(self.init_scale),
                           bias_init=constant(0.0))(states)
            hidden = hidden + b_1.reshape(B, 1, self.embedding_dim)
        hidden = nn.elu(hidden)

        q_tot = jnp.matmul(hidden, w_2)                        # (B, 1, 1)
        if self.use_bias:
            b_2 = HyperNetwork(self.embedding_dim, 1, self.init_scale)(states)
            q_tot = q_tot + b_2.reshape(B, 1, 1)

        return q_tot.reshape(B, 1)


class MAVFlowPerAgentAgent(flax.struct.PyTreeNode):
    """MAV-Flow, per-agent distributional critic mixed by QMIX."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # -------------------- Sinusoidal embedding (actor only) -------------------- #
    def _time_sin_embed(self, ts):
        kfreq = int(self.config.get('t_embed_frequencies', 8))
        freqs = jnp.asarray([2 ** i for i in range(kfreq)], dtype=ts.dtype) * jnp.pi
        ang = ts * freqs
        return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)

    # -------------------- Actor sampling (unchanged) -------------------- #
    def _integrate_actor_flow(self, noises, observations, flow_network_name):
        num_steps = self.config['num_flow_steps']
        step_size = 1.0 / num_steps
        times_shape = (*noises.shape[:-1], 1)

        def step_fn(carry, i):
            x = carry
            t = jnp.full(times_shape, i * step_size, dtype=x.dtype)
            t_embed = self._time_sin_embed(t)
            vf = self.network.select(flow_network_name)(observations, x, t_embed)
            return x + step_size * vf, None

        final, _ = jax.lax.scan(step_fn, noises, jnp.arange(num_steps))
        return final

    def _sample_candidates(self, obs_with_id, legals, rng, num_candidates):
        """N candidate actions per agent from the multi-step BC flow.
        Returns cand_int (N, B, K) and cand_oh (N, B, K, A)."""
        B, K = obs_with_id.shape[:2]
        A = self.config['action_dim']
        noises = jax.random.normal(rng, (num_candidates, B, K, A))
        logits = jax.vmap(
            lambda eps: self._integrate_actor_flow(eps, obs_with_id, 'actor_bc_flow')
        )(noises)
        masked = jnp.where(legals[None] > 0, logits, -1e9)
        cand_int = jnp.argmax(masked, axis=-1)
        return cand_int, jax.nn.one_hot(cand_int, A)

    # -------------------- Critic flow integration -------------------- #
    def compute_flow_returns(
        self,
        noises,
        observations,
        actions,
        init_times=None,
        end_times=None,
        flow_network_name=None,
        return_jac_eps_prod=False,
        params=None,
        num_steps=None,
    ):
        """Euler integration of the return flow. Works on any leading shape;
        here inputs are (B, K, 1) so the agent axis broadcasts through the MLP."""
        noisy_returns = noises
        noisy_jac_eps_prod = jnp.ones_like(noises)
        if init_times is None:
            init_times = jnp.zeros((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        if end_times is None:
            end_times = jnp.ones((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        if num_steps is None:
            num_steps = self.config['critic_flow_steps']
        step_size = (end_times - init_times) / num_steps

        def vf_fn(ret, times):
            return self.network.select(flow_network_name)(
                ret, times, observations, actions, params=params)

        def func(carry, i):
            (ret, jac) = carry
            times = i * step_size + init_times

            # Only build the JVP when it is actually requested. The original
            # computed it unconditionally, which is dead work the backward pass
            # through this scan should not have to reason about.
            if return_jac_eps_prod:
                vector_field, jac_dot = jax.jvp(
                    lambda r: vf_fn(r, times), (ret,), (jac,))
                new_jac = jac + step_size * jac_dot
            else:
                vector_field = vf_fn(ret, times)
                new_jac = jac

            new_ret = ret + step_size * vector_field
            return (new_ret, new_jac), None

        (noisy_returns, noisy_jac_eps_prod), _ = jax.lax.scan(
            func, (noisy_returns, noisy_jac_eps_prod), jnp.arange(num_steps))

        if return_jac_eps_prod:
            return noisy_returns, noisy_jac_eps_prod
        return noisy_returns

    # -------------------- Per-agent Z and mixing -------------------- #
    def _agent_z(self, eps, obs_with_id, actions_oh, flow_network_name, params=None):
        """Shared eps (B, 1) -> per-agent returns Z_i (B, K, 1).

        broadcast_to, NOT tile: the same eps must reach every agent. Independent
        per-agent noise would make Z_1 ... Z_K conditionally independent by
        construction and destroy the transport-map property of the composition.
        """
        B, K = obs_with_id.shape[:2]
        eps_bk = jnp.broadcast_to(eps[:, None, :], (B, K, 1))
        return self.compute_flow_returns(
            eps_bk, obs_with_id, actions_oh,
            flow_network_name=flow_network_name, params=params)

    def _z_tot(self, eps, obs_with_id, states, actions_oh, twin,
               target=False, params=None):
        """Returns (z_tot (B,1), z_i (B,K)) for one twin at one eps."""
        pre = 'target_' if target else ''
        z_i = self._agent_z(
            eps, obs_with_id, actions_oh,
            f'{pre}agent_critic_flow{twin}', params=params).squeeze(-1)   # (B, K)
        z_tot = self.network.select(f'{pre}mixer{twin}')(z_i, states, params=params)
        return z_tot, z_i

    def _z_tot_pessimistic(self, eps, obs_with_id, states, actions_oh, target=True):
        """Per-eps min over twin (mixer, flow) pairs. Min OUTSIDE the mixer.
        Min of two monotone maps is monotone, so the result is still a valid
        transport map and can be used as a distributional bootstrap target."""
        z1, _ = self._z_tot(eps, obs_with_id, states, actions_oh, 1, target=target)
        z2, _ = self._z_tot(eps, obs_with_id, states, actions_oh, 2, target=target)
        if self.config['ret_agg'] == 'min':
            return jnp.minimum(z1, z2)
        return 0.5 * (z1 + z2)

    def _agent_q(self, obs_with_id, actions_oh, rng, target=False):
        """Per-agent Q_i = E_eps[Z_i], twin-aggregated. Returns (B, K).
        Used for execution-time ranking only -- no mixer involved."""
        pre = 'target_' if target else ''
        L = self.config['critic_mc_samples']
        B = obs_with_id.shape[0]
        eps = jax.random.normal(rng, (L, B, 1))

        def one(e):
            z1 = self._agent_z(e, obs_with_id, actions_oh, f'{pre}agent_critic_flow1')
            z2 = self._agent_z(e, obs_with_id, actions_oh, f'{pre}agent_critic_flow2')
            if self.config['q_agg'] == 'min':
                return jnp.minimum(z1, z2)
            return 0.5 * (z1 + z2)

        return jax.vmap(one)(eps).mean(0).squeeze(-1)   # (B, K)

    # -------------------- Decentralized execution -------------------- #
    @jax.jit
    def sample_actions_rejection(self, obs_with_id, legals, rng):
        """Signature identical to the joint agent so evaluate_smac.py is unchanged.

        jit matters here: this is now 16 candidates x critic_flow_steps x
        critic_mc_samples scan steps at B=1, called once per env step.
        """
        N = self.config['num_candidates']
        rng, cand_rng, q_rng = jax.random.split(rng, 3)
        cand_int, cand_oh = self._sample_candidates(obs_with_id, legals, cand_rng, N)
        keys = jax.random.split(q_rng, N)
        q = jax.vmap(lambda a, k: self._agent_q(obs_with_id, a, k))(cand_oh, keys)  # (N,B,K)
        best = jnp.argmax(q, axis=0)                                               # (B,K)
        return jnp.take_along_axis(cand_int, best[None], axis=0).squeeze(0)

    # -------------------- Target action selection -------------------- #
    def _select_next_joint_action_teacher(self, next_obs_with_id, next_states,
                                          next_legals, rng):
        """Coordinated whole-candidate selection, scored by the pessimistic
        target Q_tot. Returns (B, K, A). No one-step shortcut: that identity
        depended on a CFM loss which no longer exists."""
        N = self.config['num_target_candidates']
        B = next_states.shape[0]
        L = self.config['critic_mc_samples']
        rng, cand_rng, score_rng = jax.random.split(rng, 3)

        _, cand_oh = self._sample_candidates(
            next_obs_with_id, next_legals, cand_rng, N)        # (N, B, K, A)

        def score(a_oh, key):
            eps = jax.random.normal(key, (L, B, 1))
            zs = jax.vmap(
                lambda e: self._z_tot_pessimistic(
                    e, next_obs_with_id, next_states, a_oh, target=True)
            )(eps)                                             # (L, B, 1)
            return zs.mean(0)                                  # (B, 1)

        keys = jax.random.split(score_rng, N)
        v_cand = jax.vmap(score)(cand_oh, keys).squeeze(-1)     # (N, B)
        best = jnp.argmax(v_cand, axis=0)                      # (B,)
        return jax.lax.stop_gradient(cand_oh[best, jnp.arange(B)])   # (B, K, A)

    # -------------------- Distributional TD loss -------------------- #
    def agent_critic_dist_td_loss(self, batch, grad_params, rng):
        """Distributional TD via all-pairs quantile Huber at the joint level.

        Why all-pairs and not paired-eps: with the prediction and target matched
        at the SAME eps, slice i of the loss reduces to
            ( Z^(i)(s,a) - [r + gamma*m*Z^(i)_targ(s',a')] )^2
        which is an ordinary scalar TD regression on Z^(i) with no coupling to
        any other slice. All M slices are then the same Bellman equation with the
        same unique solution -- the mean value function -- so every slice
        converges to the same function and the spread is zero by construction.
        Empirically this collapsed z_i_std_across_eps from 0.99 to 1e-4 within
        200 steps on both 3m Good and 3m Poor.

        All-pairs breaks the degeneracy: slice i is pulled toward quantile tau_i
        of the target mixture over j, so different tau_i see different targets.

        eps is a DETERMINISTIC grid, not a random draw. Since eps -> Z_i(eps) is
        monotone (1-D ODE trajectories cannot cross), z_pred[i] IS the tau_i
        quantile with no sorting required. Sharing the grid between prediction
        and target is fine -- it was the pairing in the loss that collapsed
        things, not the shared draw.
        """
        B = batch['actions'].shape[0]
        A = self.config['action_dim']
        K = self.config['num_agents']
        M = self.config['num_dist_samples']
        rng, act_rng = jax.random.split(rng, 2)

        states = batch['states']
        next_states = batch['next_states']
        obs_with_id = batch_concat_agent_id_to_obs(batch['observations'])
        next_obs_with_id = batch_concat_agent_id_to_obs(batch['next_observations'])
        actions_oh = jax.nn.one_hot(batch['actions'], A)                  # (B, K, A)

        next_actions_oh = self._select_next_joint_action_teacher(
            next_obs_with_id, next_states, batch['next_legals'], act_rng)  # (B, K, A)

        r_ = jnp.expand_dims(batch['rewards'], -1)                        # (B, 1)
        m_ = jnp.expand_dims(batch['masks'], -1)                          # (B, 1)

        # Fixed quantile levels and the eps that realize them.
        tau = (jnp.arange(M, dtype=jnp.float32) + 0.5) / M                # (M,)
        eps_grid = jax.scipy.stats.norm.ppf(tau)[:, None, None]           # (M, 1, 1)
        eps = jnp.broadcast_to(eps_grid, (M, B, 1))                       # (M, B, 1)

        # --- target: pessimistic (per-eps min over twins) at (s', a') ---
        z_next = jax.vmap(
            lambda e: self._z_tot_pessimistic(
                e, next_obs_with_id, next_states, next_actions_oh, target=True)
        )(eps)                                                            # (M, B, 1)
        G = jax.lax.stop_gradient(
            r_[None] + self.config['discount'] * m_[None] * z_next)       # (M, B, 1)

        kappa = self.config['huber_kappa']

        def twin_loss(twin):
            def pred(e):
                return self._z_tot(e, obs_with_id, states, actions_oh, twin,
                                   target=False, params=grad_params)
            z_pred, z_i = jax.vmap(pred)(eps)                             # (M,B,1), (M,B,K)

            # u[i, j, b] = G[j, b] - z_pred[i, b]
            u = G[None, :, :, 0] - z_pred[:, None, :, 0]                  # (M, M, B)
            w = jnp.abs(tau[:, None, None] - (u < 0).astype(u.dtype))
            loss = (w * optax.huber_loss(u, 0.0, delta=kappa)).mean()
            return loss, z_pred, z_i

        loss1, z_pred1, z_i1 = twin_loss(1)
        loss2, z_pred2, z_i2 = twin_loss(2)
        loss = loss1 + loss2

        q_i = 0.5 * (z_i1.mean(0) + z_i2.mean(0))                         # (B, K)

        assert z_pred1.shape == (M, B, 1)
        assert G.shape == (M, B, 1)
        assert z_i1.shape == (M, B, K)

        return loss, {
            'td_loss': loss,
            'z_tot_mean': 0.5 * (z_pred1.mean() + z_pred2.mean()),
            'target_mean': G.mean(),
            # The metric that matters. Compare against target_std_across_eps:
            # if the target has spread and the prediction does not, the loss is
            # failing to transmit distributional structure.
            'z_tot_std_across_eps': z_pred1.std(axis=0).mean(),
            'target_std_across_eps': G.std(axis=0).mean(),
            'z_i_std_across_eps': z_i1.std(axis=0).mean(),
            # Disambiguates the two collapse modes:
            #   across_agents -> the agent-id channel died
            #   across_batch  -> the flow stopped depending on (o_i, a_i)
            'q_i_mean': q_i.mean(),
            'q_i_std_across_agents': q_i.std(axis=-1).mean(),
            'q_i_std_across_batch': q_i.std(axis=0).mean(),
            'twin_gap': (z_pred1 - z_pred2).mean(),
            'twin_absgap': jnp.abs(z_pred1 - z_pred2).mean(),
            'twin_std': (z_pred1 - z_pred2).std(),
        }

    # -------------------- Actor (unchanged) -------------------- #
    def actor_loss(self, batch, grad_params, rng):
        batch_size = batch['actions'].shape[0]
        K = self.config['num_agents']
        A = self.config['action_dim']

        rng, bc_noise_rng, bc_time_rng = jax.random.split(rng, 3)

        obs_with_id = batch_concat_agent_id_to_obs(batch['observations'])
        actions_oh = jax.nn.one_hot(batch['actions'], A)

        bc_noises = jax.random.normal(bc_noise_rng, (batch_size, K, A))
        bc_times = jax.random.uniform(bc_time_rng, (batch_size, K, 1))
        bc_noisy_actions = bc_times * actions_oh + (1 - bc_times) * bc_noises
        bc_target_vf = actions_oh - bc_noises

        bc_t_embed = self._time_sin_embed(bc_times)
        bc_vf = self.network.select('actor_bc_flow')(
            obs_with_id, bc_noisy_actions, bc_t_embed, params=grad_params)
        assert bc_vf.shape == (batch_size, K, A)

        bc_loss = ((bc_vf - bc_target_vf) ** 2).mean()
        actor_loss = bc_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
        }

    # -------------------- Construction -------------------- #
    @classmethod
    def create(cls, seed: int, example_batch, config):
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

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['num_agents'] = num_agents
        config['state_dim'] = state_dim
        config['min_reward'] = min_reward
        config['max_reward'] = max_reward

        ex_actions_per_agent_oh = jax.nn.one_hot(ex_actions, action_dim)   # (1, K, A)
        ex_obs_with_id = batch_concat_agent_id_to_obs(ex_obs)              # (1, K, ob+K)

        # Critic flow now lives on the agent axis.
        ex_returns_agent = jnp.zeros((1, num_agents, 1), dtype=jnp.float32)
        ex_times_agent = jnp.zeros((1, num_agents, 1), dtype=jnp.float32)
        ex_q_vals = jnp.zeros((ex_states.shape[0], num_agents), dtype=jnp.float32)

        kfreq = int(config.get('t_embed_frequencies', 8))
        ex_times_per_agent = jnp.zeros((1, num_agents, 2 * kfreq), dtype=jnp.float32)

        def make_flow():
            return ValueVectorField(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['value_layer_norm'],
                num_ensembles=1,
                encoder=None,
            )

        def make_mixer():
            return MixingNetwork(
                embedding_dim=config['mixer_embed_dim'],
                hypernet_hidden_dim=config['mixer_hypernet_hidden_dim'],
                init_scale=config['mixer_init_scale'],
                use_bias=config['mixer_use_bias'],
            )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=None,
        )

        flow_args = (ex_returns_agent, ex_times_agent,
                     ex_obs_with_id, ex_actions_per_agent_oh)
        mixer_args = (ex_q_vals, ex_states)

        network_info = dict(
            agent_critic_flow1=(make_flow(), flow_args),
            agent_critic_flow2=(make_flow(), flow_args),
            target_agent_critic_flow1=(make_flow(), flow_args),
            target_agent_critic_flow2=(make_flow(), flow_args),
            mixer1=(make_mixer(), mixer_args),
            mixer2=(make_mixer(), mixer_args),
            target_mixer1=(make_mixer(), mixer_args),
            target_mixer2=(make_mixer(), mixer_args),
            actor_bc_flow=(actor_bc_flow_def,
                           (ex_obs_with_id, ex_actions_per_agent_oh, ex_times_per_agent)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']

        for m in ('agent_critic_flow1', 'agent_critic_flow2', 'mixer1', 'mixer2'):
            network_params[f'modules_target_{m}'] = network_params[f'modules_{m}']

        network = TrainState.create(network_def, network_params, tx=network_tx)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))

    # -------------------- Optimization -------------------- #
    def total_loss(self, batch, grad_params, rng):
        rng, td_rng, actor_rng = jax.random.split(rng, 3)

        td_loss, td_info = self.agent_critic_dist_td_loss(batch, grad_params, td_rng)
        a_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)

        total = td_loss + a_loss

        info = {'total_loss': total, 'critic_loss': td_loss}
        info.update({f'critic/{k}': v for k, v in td_info.items()})
        info.update({f'actor/{k}': v for k, v in actor_info.items()})
        return total, info

    def target_update(self, network, module_name):
        """Polyak EMA: target <- tau * main + (1 - tau) * target.

        Reads network.params (post-gradient), not self.network.params, so the
        target tracks the parameters that actually exist after this step.
        """
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        for m in ('agent_critic_flow1', 'agent_critic_flow2', 'mixer1', 'mixer2'):
            self.target_update(new_network, m)

        return self.replace(network=new_network, rng=new_rng), info


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='mav_flow_per_agent',
            ob_dims=ml_collections.config_dict.placeholder(int),
            action_dim=ml_collections.config_dict.placeholder(int),
            num_agents=ml_collections.config_dict.placeholder(int),
            state_dim=ml_collections.config_dict.placeholder(int),
            min_reward=ml_collections.config_dict.placeholder(float),
            max_reward=ml_collections.config_dict.placeholder(float),
            encoder=ml_collections.config_dict.placeholder(str),

            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_layer_norm=True,
            discount=0.99,
            tau=0.005,

            ret_agg='min',            # twin aggregation for the bootstrap target
            q_agg='min',              # twin aggregation for execution-time q_i
            clip_flow_actions=False,
            clip_flow_returns=False,
            clip_straight_through = True,

            num_flow_steps=10,        # ACTOR integration steps
            critic_flow_steps=10,     # CRITIC integration steps -- decoupled, since
                                      # this one sits in the backward pass. Drop to
                                      # 4 if memory or step time bites.
            num_dist_samples=8,       # M: shared-eps draws per transition
            critic_mc_samples=4,      # L: eps draws for mean readouts
            huber_kappa=1.0,

            mixer_embed_dim=32,
            mixer_hypernet_hidden_dim=64,
            mixer_init_scale=1.0,
            mixer_use_bias=True,     # closes the action-independent V(s) path

            num_candidates=16,        # execution-time rejection sampling
            num_target_candidates=8,  # bootstrap action selection
            t_embed_frequencies=8,
        )
    )
    return config