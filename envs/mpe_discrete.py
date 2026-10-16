"""Discrete-action MPE Simple Spread for the discrete MARL harness (envs/marl_harness.py).

Uses og_marl's bundled particle env, i.e. the SAME scenario code as the OMAR vaults
(continuous, envs/mpe_omar.py), with `discrete_action=True`:
  action 0 = no-op, 1 = +x, 2 = -x, 3 = +y, 4 = -y   (MultiAgentEnv._set_action)

Upstream per-agent reward (og_marl simple_spread.py, "modified by ling"):
  r_i = sum_l min(1 / min_j d(j, l), 10)  -  5 * #collisions(i)
The coverage term is shared but the collision term is per agent, so the local rewards
differ whenever agents collide. The team reward is upstream's own shared-reward
aggregate, `np.sum(reward_n)` (MultiAgentEnv.step), via MARLAdapter.team_reward.
NOTE this is K times the per-agent scale the OMAR datasets are usually reported in.

No absorbing state: every episode is truncated at the horizon (25, as in OMAR).
"""

import itertools

import numpy as np
from og_marl.custom_environments.multiagent_particle_envs.multiagent.environment import (
    MultiAgentEnv,
)
from og_marl.custom_environments.multiagent_particle_envs.multiagent.simple_spread import (
    Scenario,
)

from envs.marl_harness import MARLAdapter

_ACTION_TO_DIR = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)


class SpreadDiscrete(MARLAdapter):
    family = 'mpedisc'

    def __init__(self, preset, n_agents, horizon):
        super().__init__(preset, horizon, n_agents, num_actions=5)
        self._scenario = Scenario()
        # make_world() calls reset_world(), which draws from global np.random.
        global_state = np.random.get_state()
        np.random.seed(0)
        try:
            world = self._scenario.make_world(num_agents=n_agents)
        finally:
            np.random.set_state(global_state)
        self.env = MultiAgentEnv(
            world, self._scenario.reset_world, self._scenario.reward,
            self._scenario.observation, discrete_action=True)
        self.world = world
        self._obs = None

    def spec(self):
        return dict(family='mpedisc', kind='spread', preset=self.preset,
                    n_agents=self.num_agents, horizon=self.horizon)

    def _reset_inner(self, seed):
        self._obs = np.stack(self.env.reset())

    def _step_inner(self, acts):
        one_hot = [np.eye(self.num_actions)[a] for a in acts]
        obs, rew, _, _ = self.env.step(one_hot)
        self._obs = np.stack(obs)
        return np.asarray(rew, dtype=np.float64), None    # upstream never ends an episode

    def _observe(self):
        return self._obs

    def _global_state(self):
        agents = self.world.agents
        return np.concatenate(
            [a.state.p_pos for a in agents] + [a.state.p_vel for a in agents]
            + [l.state.p_pos for l in self.world.landmarks]).astype(np.float32)

    def _is_absorbing(self):
        return False

    def behaviour_policies(self):
        return ('random', 'local', 'central')

    @staticmethod
    def _steer(delta, vel):
        """Discrete action toward `delta`, damped by current velocity."""
        target = delta - 0.5 * vel
        if np.max(np.abs(target)) < 0.05:
            return 0
        return int(np.argmax(_ACTION_TO_DIR[1:] @ target)) + 1

    def scripted_actions(self, policy, rng):
        agents, landmarks = self.world.agents, self.world.landmarks
        pos = np.stack([a.state.p_pos for a in agents])
        vel = np.stack([a.state.p_vel for a in agents])
        lm = np.stack([l.state.p_pos for l in landmarks])
        dist = np.linalg.norm(pos[:, None, :] - lm[None, :, :], axis=-1)   # (K, L)
        if policy == 'local':
            # Each agent heads for its nearest landmark: no coordination, so agents
            # pile onto the same landmark.
            target = np.argmin(dist, axis=1)
        elif policy == 'central':
            # Joint assignment minimising total distance: a coupled joint action.
            best = min(itertools.permutations(range(len(landmarks)), self.num_agents),
                       key=lambda p: dist[np.arange(self.num_agents), list(p)].sum())
            target = np.asarray(best)
        else:
            return super().scripted_actions(policy, rng)
        return np.array([self._steer(lm[target[i]] - pos[i], vel[i])
                         for i in range(self.num_agents)], dtype=np.int64)


def make_env(spec):
    if spec['kind'] != 'spread':
        raise ValueError(f'Unknown MPE kind {spec["kind"]!r}.')
    return SpreadDiscrete(spec['preset'], spec['n_agents'], spec['horizon'])


PRESETS = {
    'spread-3': lambda: SpreadDiscrete('spread-3', n_agents=3, horizon=25),
    'spread-5': lambda: SpreadDiscrete('spread-5', n_agents=5, horizon=25),
}
