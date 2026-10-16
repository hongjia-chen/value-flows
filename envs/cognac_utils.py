"""COGNAC adapters for the discrete MARL harness (see envs/marl_harness.py).

Upstream code is the pinned, vendored copy in envs/_vendor/cognac (PATCHES.md lists
every edit; none touch dynamics or rewards). Upstream quirks handled HERE:

  * Initial states / BinaryConsensus dynamics use GLOBAL np.random, and
    BinaryConsensus ignores reset(seed). -> the base class runs every upstream call
    inside a private RandomState seeded from the episode seed.
  * Firefighting's reset observation reads `last_visited_house` from the PREVIOUS
    episode. -> reset to the identity (action 0) before every upstream reset.
  * Firefighting ends after `max_steps + 2` steps. -> upstream max_steps = horizon - 2,
    so `horizon` is the real episode length (asserted every step by the base class).
  * SysAdmin / BinaryConsensus flag the time limit as a termination. -> the base class
    decides terminated from `_is_absorbing()`; a time limit is a truncation.
  * Firefighting ignores `is_shared_reward`. -> never used; team reward is computed by
    the base class as the sum of local rewards, which is upstream's own
    BaseReward._get_global_reward.

Observations are re-encoded to fixed-width float vectors (one-hot categoricals; own
entry first; zero padding for smaller neighbourhoods) so one shared network serves all
agents. Firefighting observations also carry the agent's own previous action, which the
agent knows and which determines the house its 1-bit flame reading refers to.
"""

import numpy as np

from envs._vendor.cognac.env import (
    BinaryConsensusNetworkEnvironment,
    GridFireFightingGraphEnvironment,
    RowFireFightingGraphEnvironment,
    SysAdminNetworkEnvironment,
)
from envs._vendor.cognac.utils import graph_utils
from envs.marl_harness import MARLAdapter


def _one_hot(idx, n):
    return np.eye(n, dtype=np.float32)[np.asarray(idx, dtype=np.int64)]


def _seeded_graph(fn, seed, *args, **kwargs):
    """Upstream graph generators draw from global np.random; isolate and seed them."""
    global_state = np.random.get_state()
    np.random.seed(seed)
    try:
        return np.asarray(fn(*args, **kwargs), dtype=np.float64)
    finally:
        np.random.set_state(global_state)


def _neighbourhood_order(masks):
    """Per agent: own index first, then visible neighbours in index order."""
    return [[i] + [int(j) for j in np.nonzero(masks[i])[0] if j != i]
            for i in range(masks.shape[0])]


# =============================================================================
# Firefighting
# =============================================================================
class _FirefightingBase(MARLAdapter):
    family = 'cognac'

    def _observe(self):
        K = self.num_agents
        bits = np.array([float(self._obs_raw[a][0]) for a in self._upstream_agents],
                        dtype=np.float32).reshape(K, 1)
        return np.concatenate([bits, _one_hot(self._last_actions, self.num_actions)], axis=1)

    def _global_state(self):
        fire = np.asarray(self.env.state, dtype=np.float32).ravel() / self.max_fire_level
        return np.concatenate([fire, _one_hot(self._last_actions, self.num_actions).ravel()])

    def _is_absorbing(self):
        return bool(np.all(np.asarray(self.env.state) == 0))

    def _success(self):
        return self._is_absorbing()

    def _reset_inner(self, seed):
        self.env.last_visited_house = {a: self._identity_house(a) for a in self._upstream_agents}
        self._obs_raw, _ = self.env.reset(seed=seed)
        self._last_actions = np.zeros(self.num_agents, dtype=np.int64)

    def _step_inner(self, acts):
        obs, rew, term, trunc, _ = self.env.step(
            {a: int(acts[i]) for i, a in enumerate(self._upstream_agents)})
        self._obs_raw = obs
        self._last_actions = np.asarray(acts, dtype=np.int64).copy()
        local = np.array([rew[a] for a in self._upstream_agents], dtype=np.float64)
        return local, any(term.values()) or any(trunc.values())

    def behaviour_policies(self):
        return ('random', 'local', 'central')

    def scripted_actions(self, policy, rng):
        if policy == 'local':
            # Decentralised, observation-only: keep fighting where flames were seen,
            # otherwise move to the next house this agent can reach.
            obs = self._observe()
            flame = obs[:, 0] > 0.5
            last = np.argmax(obs[:, 1:], axis=1)
            return np.where(flame, last, (last + 1) % self.num_actions)
        if policy == 'central':
            return self._central_assignment()
        return super().scripted_actions(policy, rng)


class FirefightingRow(_FirefightingBase):
    """Row of n+1 houses; agent i fights house i (action 0) or house i+1 (action 1)."""

    def __init__(self, preset, n_agents, max_fire_level, horizon):
        if horizon < 3:
            raise ValueError('Firefighting needs horizon >= 3.')
        super().__init__(preset, horizon, n_agents, num_actions=2)
        self.max_fire_level = int(max_fire_level)
        self.env = RowFireFightingGraphEnvironment(
            n=n_agents, max_steps=horizon - 2, max_fire_level=max_fire_level)
        self._upstream_agents = list(self.env.possible_agents)
        self._last_actions = np.zeros(n_agents, dtype=np.int64)
        self._obs_raw = {a: np.zeros(1) for a in self._upstream_agents}

    @staticmethod
    def _identity_house(agent):
        return agent

    def spec(self):
        return dict(family='cognac', kind='ffrow', preset=self.preset, n_agents=self.num_agents,
                    max_fire_level=self.max_fire_level, horizon=self.horizon)

    def _central_assignment(self):
        """Hottest house first; send BOTH adjacent free agents (two agents zero a house)."""
        state = np.asarray(self.env.state)
        K = self.num_agents
        acts = np.full(K, -1, dtype=np.int64)
        for h in np.argsort(-state, kind='stable'):
            if state[h] <= 0:
                break
            for agent, act in ((h, 0), (h - 1, 1)):
                if 0 <= agent < K and acts[agent] < 0:
                    acts[agent] = act
        for i in np.nonzero(acts < 0)[0]:
            acts[i] = 1 if state[i + 1] > state[i] else 0
        return acts


class FirefightingGrid(_FirefightingBase):
    """(h+1)x(w+1) houses; agent (r, c) fights one of the 4 houses at (r+dr, c+dc)."""

    _OFFSETS = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}   # upstream _act_id_to_house_pos_dict

    def __init__(self, preset, n_width, n_height, max_fire_level, horizon):
        if horizon < 3:
            raise ValueError('Firefighting needs horizon >= 3.')
        super().__init__(preset, horizon, n_width * n_height, num_actions=4)
        self.n_width, self.n_height = int(n_width), int(n_height)
        self.max_fire_level = int(max_fire_level)
        self.env = GridFireFightingGraphEnvironment(
            n_width=n_width, n_height=n_height, max_steps=horizon - 2,
            max_fire_level=max_fire_level)
        self._upstream_agents = list(self.env.possible_agents)
        self._last_actions = np.zeros(self.num_agents, dtype=np.int64)
        self._obs_raw = {a: np.zeros(1) for a in self._upstream_agents}

    @staticmethod
    def _identity_house(agent):
        return tuple(agent)

    def spec(self):
        return dict(family='cognac', kind='ffgrid', preset=self.preset, n_width=self.n_width,
                    n_height=self.n_height, max_fire_level=self.max_fire_level,
                    horizon=self.horizon)

    def _central_assignment(self):
        state = np.asarray(self.env.state)
        index = {a: i for i, a in enumerate(self._upstream_agents)}
        acts = np.full(self.num_agents, -1, dtype=np.int64)
        order = np.argsort(-state.ravel(), kind='stable')
        for flat in order:
            hr, hc = np.unravel_index(flat, state.shape)
            if state[hr, hc] <= 0:
                break
            assigned = 0
            for act, (dr, dc) in self._OFFSETS.items():
                agent = (hr - dr, hc - dc)
                if agent in index and acts[index[agent]] < 0 and assigned < 2:
                    acts[index[agent]] = act
                    assigned += 1
        for i in np.nonzero(acts < 0)[0]:
            r, c = self._upstream_agents[i]
            levels = [state[r + dr, c + dc] for dr, dc in self._OFFSETS.values()]
            acts[i] = int(np.argmax(levels))
        return acts


# =============================================================================
# SysAdmin
# =============================================================================
class SysAdmin(MARLAdapter):
    """Machine status {0 good, 1 faulty, 2 dead} x job {0 idle, 1 loaded, 2 done}.

    Action 0 = do nothing, 1 = reboot. Local reward 1 when a non-dead machine's job is
    done. No absorbing state: every episode runs to the horizon (truncated).
    """

    family = 'cognac'

    def __init__(self, preset, adjacency, horizon, show_neighborhood_state):
        A = np.asarray(adjacency, dtype=np.float64)
        super().__init__(preset, horizon, A.shape[0], num_actions=2)
        self.adjacency = A
        self.show_neighborhood_state = bool(show_neighborhood_state)
        self.env = SysAdminNetworkEnvironment(
            adjacency_matrix=A, max_steps=horizon,
            show_neighborhood_state=self.show_neighborhood_state)
        self._order = _neighbourhood_order(self.env.neighboring_masks)
        self._slots = max(len(o) for o in self._order)
        self._neighbours = (A != 0) | (A.T != 0)

    def spec(self):
        return dict(family='cognac', kind='sysadmin', preset=self.preset,
                    adjacency=self.adjacency.tolist(), horizon=self.horizon,
                    show_neighborhood_state=self.show_neighborhood_state)

    def _reset_inner(self, seed):
        self.env.reset(seed=seed)

    def _step_inner(self, acts):
        _, rew, term, trunc, _ = self.env.step({i: int(acts[i]) for i in range(self.num_agents)})
        local = np.array([rew[i] for i in range(self.num_agents)], dtype=np.float64)
        return local, any(term.values()) or any(trunc.values())

    def _machine_codes(self):
        s = np.asarray(self.env._state, dtype=np.int64)
        return np.concatenate([_one_hot(s[:, 0], 3), _one_hot(s[:, 1], 3)], axis=1)   # (K, 6)

    def _observe(self):
        codes = self._machine_codes()
        obs = np.zeros((self.num_agents, self._slots * 6), dtype=np.float32)
        for i, order in enumerate(self._order):
            obs[i, :len(order) * 6] = codes[order].ravel()
        return obs

    def _global_state(self):
        progress = np.asarray(self.env.task_progress_vect, dtype=np.float32)
        return np.concatenate([self._machine_codes().ravel(), progress])

    def _is_absorbing(self):
        return False

    def behaviour_policies(self):
        return ('random', 'local', 'central')

    def scripted_actions(self, policy, rng):
        s = np.asarray(self.env._state, dtype=np.int64)
        status, job = s[:, 0], s[:, 1]
        # Local: reboot dead machines, and faulty ones that have no job to lose.
        local = (status == 2) | ((status == 1) & (job != 1))
        if policy == 'local':
            return local.astype(np.int64)
        if policy == 'central':
            # Also reboot a faulty LOADED machine when it is about to infect healthy
            # neighbours and its job is not nearly finished. Needs global state.
            progress = np.asarray(self.env.task_progress_vect)
            healthy_nbr = (self._neighbours & (status == 0)[None, :]).any(axis=1)
            stop_spread = (status == 1) & (job == 1) & healthy_nbr & (progress < 0.5)
            return (local | stop_spread).astype(np.int64)
        return super().scripted_actions(policy, rng)


# =============================================================================
# BinaryConsensus
# =============================================================================
class BinaryConsensus(MARLAdapter):
    """Binary votes on an influence graph. Action 1 pushes the agent to flip its vote,
    0 to keep it; neighbours' actions perturb the outcome. Absorbing at consensus."""

    family = 'cognac'

    def __init__(self, preset, adjacency, horizon, show_neighborhood_state):
        A = np.asarray(adjacency, dtype=np.float64)
        super().__init__(preset, horizon, A.shape[0], num_actions=2)
        self.adjacency = A
        self.show_neighborhood_state = bool(show_neighborhood_state)
        self.env = BinaryConsensusNetworkEnvironment(
            adjacency_matrix=A, max_steps=horizon,
            show_neighborhood_state=self.show_neighborhood_state)
        if self.env.reward.target is not None or type(self.env.reward).__name__ != 'FactoredRewardModel':
            raise RuntimeError('Expected upstream FactoredRewardModel with target=None.')
        self._order = _neighbourhood_order(self.env.neighboring_masks)
        self._slots = max(len(o) for o in self._order)

    def spec(self):
        return dict(family='cognac', kind='consensus', preset=self.preset,
                    adjacency=self.adjacency.tolist(), horizon=self.horizon,
                    show_neighborhood_state=self.show_neighborhood_state)

    def _votes(self):
        return np.asarray(self.env._state, dtype=np.int64)

    def _reset_inner(self, seed):
        self.env.reset(seed=seed)

    def _step_inner(self, acts):
        _, rew, term, trunc, _ = self.env.step({i: int(acts[i]) for i in range(self.num_agents)})
        local = np.array([rew[i] for i in range(self.num_agents)], dtype=np.float64)
        return local, any(term.values()) or any(trunc.values())

    def _observe(self):
        codes = _one_hot(self._votes(), 2)
        obs = np.zeros((self.num_agents, self._slots * 2), dtype=np.float32)
        for i, order in enumerate(self._order):
            obs[i, :len(order) * 2] = codes[order].ravel()
        return obs

    def _global_state(self):
        return self._votes().astype(np.float32)

    def _is_absorbing(self):
        v = self._votes()
        return bool(np.all(v == 0) or np.all(v == 1))

    def _success(self):
        return self._is_absorbing()

    def behaviour_policies(self):
        return ('random', 'local', 'central')

    def scripted_actions(self, policy, rng):
        v = self._votes()
        if policy == 'local':
            acts = np.zeros(self.num_agents, dtype=np.int64)
            for i, order in enumerate(self._order):
                ones = v[order].sum()
                zeros = len(order) - ones
                if ones != zeros:
                    acts[i] = int(v[i] != (1 if ones > zeros else 0))
            return acts
        if policy == 'central':
            target = 1 if v.sum() * 2 >= self.num_agents else 0
            return (v != target).astype(np.int64)
        return super().scripted_actions(policy, rng)


# =============================================================================
# Presets and spec rebuild
# =============================================================================
def make_env(spec):
    kind = spec['kind']
    if kind == 'ffrow':
        return FirefightingRow(spec['preset'], spec['n_agents'], spec['max_fire_level'], spec['horizon'])
    if kind == 'ffgrid':
        return FirefightingGrid(spec['preset'], spec['n_width'], spec['n_height'],
                                spec['max_fire_level'], spec['horizon'])
    if kind == 'sysadmin':
        return SysAdmin(spec['preset'], spec['adjacency'], spec['horizon'],
                        spec['show_neighborhood_state'])
    if kind == 'consensus':
        return BinaryConsensus(spec['preset'], spec['adjacency'], spec['horizon'],
                               spec['show_neighborhood_state'])
    raise ValueError(f'Unknown COGNAC kind {kind!r}.')


def _band(n, seed):
    return _seeded_graph(graph_utils.generate_band_adjacency_matrix, seed, n, 1)


def _random(n, seed):
    return _seeded_graph(graph_utils.generate_adjacency_matrix, seed, n)


# Fire levels: COGNAC's default of 100 puts the per-step team reward near -100*n and
# breaks the greedy 'central' heuristic. Measured on 40 episodes each (eps=0):
#   row  fire 3 : random already 100% success in ~10 steps -> nothing to learn
#   row  fire 10: random R=-141 succ=.75 | local -110 / 1.0 | central -77 / 1.0
#   grid fire 3 : random R=-218 succ=.03 | local -198 / .10 | central -77 / .95
# (upstream grid draws initial levels from [0, max) but row from [0, max]).
# Graph presets fix the graph seed: the graph is part of the task, and the dataset
# JSON stores the adjacency itself, so eval rebuilds the identical graph.
PRESETS = {
    'ffrow-8': lambda: FirefightingRow('ffrow-8', n_agents=8, max_fire_level=10, horizon=50),
    'ffrow-16': lambda: FirefightingRow('ffrow-16', n_agents=16, max_fire_level=10, horizon=50),
    'ffgrid-3x3': lambda: FirefightingGrid('ffgrid-3x3', n_width=3, n_height=3,
                                           max_fire_level=3, horizon=50),
    'sysadmin-band-10': lambda: SysAdmin('sysadmin-band-10', _band(10, 0), horizon=100,
                                         show_neighborhood_state=False),
    'sysadmin-rand-10': lambda: SysAdmin('sysadmin-rand-10', _random(10, 0), horizon=100,
                                         show_neighborhood_state=False),
    'consensus-band-10': lambda: BinaryConsensus('consensus-band-10', _band(10, 0), horizon=100,
                                                 show_neighborhood_state=True),
    'consensus-rand-10': lambda: BinaryConsensus('consensus-rand-10', _random(10, 0), horizon=100,
                                                 show_neighborhood_state=True),
}
