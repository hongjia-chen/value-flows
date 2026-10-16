"""Shared harness for discrete cooperative MARL benchmarks WITHOUT published datasets.

COGNAC (Firefighting / SysAdmin / BinaryConsensus) and discrete MPE ship no offline
data, so this module defines one contract that BOTH the dataset generator
(`generate_dataset.py`) and the evaluator (`utils/evaluate_marl.py`) go through:

  env_name = '<family>_<preset>_<uid>'   e.g. cognac_ffrow-8_central-eps0.2
             family in {'cognac', 'mpedisc'}; preset and uid contain no '_'.

The env API mirrors og_marl's SMAC wrapper, so MAV runs exactly as on SMAC:

  reset(seed) -> obs {agent: (O,)}, info {'legals': {agent: (A,)}, 'state': (S,)}
  step(acts)  -> obs, rewards {agent: team}, terminals {agent}, truncations {agent}, info

REWARD CONTRACT. Every agent receives the SAME scalar team reward, defined as the
env's own shared-reward aggregate (`team_reward()`, the sum of local rewards for every
env here). It is replicated per agent exactly like og_marl's SMAC vaults, so the
loader's `rewards[:, 0]` and the evaluator's `mean(rewards.values())` are the same
number by construction. Per-agent local rewards are never used for training.

TERMINATION CONTRACT. `terminated` comes from an explicit absorbing-state predicate
(`_is_absorbing`), never from the upstream flags, because COGNAC sets
terminations=True at the time limit for SysAdmin and BinaryConsensus. A time limit is
ALWAYS a truncation (mask = 1). The upstream end-of-episode signal is still checked
against ours on every step and any disagreement raises.

SEEDING CONTRACT. reset(seed) makes the whole episode a pure function of `seed` and
the joint actions: upstream code that draws from the GLOBAL np.random runs inside a
private RandomState, and the global stream is restored afterwards.
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from utils.vault_paths import get_source_vault_dir

HARNESS_VERSION = 1
FAMILIES = ('cognac', 'mpedisc')

# Keys returned by get_dataset(); identical to envs/smac_utils.get_dataset so
# MAVFlowAgent.create() and Dataset.sample() see the same structure as on SMAC.
TRANSITION_KEYS = (
    'observations', 'next_observations', 'actions', 'legals', 'next_legals',
    'states', 'next_states', 'rewards', 'terminals', 'truncations', 'masks',
)

# Eval episodes use seeds from here up; the generator refuses seeds in this range,
# so evaluation never replays a training episode's initial conditions.
EVAL_SEED_BASE = 1_000_000_000


class MARLAdapter:
    """Base class. Subclasses implement the underscore hooks; do not override reset/step."""

    family = None

    def __init__(self, preset, horizon, num_agents, num_actions):
        self.preset = preset
        self.horizon = int(horizon)
        self.num_agents = int(num_agents)
        self.num_actions = int(num_actions)
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.t = 0
        self._np_state = None
        self._needs_reset = True

    # ---------------------------------------------------------------- hooks
    def _reset_inner(self, seed):
        raise NotImplementedError

    def _step_inner(self, actions):
        """Advance upstream env. Return (local_rewards (K,), upstream_episode_ended)."""
        raise NotImplementedError

    def _observe(self):
        """(K, obs_dim) float32."""
        raise NotImplementedError

    def _global_state(self):
        """(state_dim - 1,) float32; the normalised timestep is appended by the base."""
        raise NotImplementedError

    def _is_absorbing(self):
        raise NotImplementedError

    def _success(self):
        """Task success flag at episode end, or None when the task has no such notion."""
        return None

    def team_reward(self, local_rewards):
        return float(np.sum(local_rewards))

    def scripted_actions(self, policy, rng):
        """Joint action (K,) int for a named behaviour policy. 'random' is handled here."""
        raise ValueError(f'{type(self).__name__} has no behaviour policy {policy!r}.')

    def behaviour_policies(self):
        return ('random',)

    def legals(self):
        return np.ones((self.num_agents, self.num_actions), dtype=np.float32)

    def spec(self):
        """JSON-serialisable constructor args; make_env(spec) must rebuild this env."""
        raise NotImplementedError

    # ---------------------------------------------------------------- RNG plumbing
    def _with_env_rng(self, fn, *args, **kwargs):
        global_state = np.random.get_state()
        np.random.set_state(self._np_state)
        try:
            out = fn(*args, **kwargs)
            self._np_state = np.random.get_state()
            return out
        finally:
            np.random.set_state(global_state)

    # ---------------------------------------------------------------- public API
    def _ensure_initialised(self):
        # Dims need a live upstream state. Every episode starts with reset(seed), so a
        # throwaway reset here cannot leak into any rollout.
        if self._np_state is None:
            self.reset(seed=0)

    @property
    def obs_dim(self):
        self._ensure_initialised()
        return int(self._observe().shape[-1])

    @property
    def state_dim(self):
        self._ensure_initialised()
        return int(self._global_state().shape[-1]) + 1

    def get_state(self):
        t_frac = np.array([self.t / self.horizon], dtype=np.float32)
        return np.concatenate([self._global_state().astype(np.float32), t_frac])

    def _obs_dict(self):
        obs = self._observe().astype(np.float32)
        assert obs.shape[0] == self.num_agents, obs.shape
        return {ag: obs[i] for i, ag in enumerate(self.agents)}

    def _info(self):
        legals = self.legals()
        return {
            'legals': {ag: legals[i] for i, ag in enumerate(self.agents)},
            'state': self.get_state(),
        }

    def reset(self, seed):
        if seed is None:
            raise ValueError('reset() needs an explicit seed; unseeded episodes cannot be paired.')
        seed = int(seed)
        self._np_state = np.random.RandomState(seed % (2 ** 32)).get_state()
        self.t = 0
        self._with_env_rng(self._reset_inner, seed)
        self._needs_reset = False
        return self._obs_dict(), self._info()

    def step(self, actions):
        if self._needs_reset:
            raise RuntimeError('step() called on a finished episode; call reset(seed) first.')
        if isinstance(actions, dict):
            acts = np.array([int(actions[ag]) for ag in self.agents], dtype=np.int64)
        else:
            acts = np.asarray(actions, dtype=np.int64).reshape(self.num_agents)
        legals = self.legals()
        if np.any(acts < 0) or np.any(acts >= self.num_actions) or \
                np.any(legals[np.arange(self.num_agents), acts] == 0):
            raise ValueError(f'Illegal joint action {acts.tolist()}.')

        local, upstream_ended = self._with_env_rng(self._step_inner, acts)
        self.t += 1
        team = self.team_reward(local)
        terminated = bool(self._is_absorbing())
        truncated = (not terminated) and self.t >= self.horizon
        if upstream_ended is not None and bool(upstream_ended) != (terminated or truncated):
            raise RuntimeError(
                f'{self.preset}: upstream episode end ({upstream_ended}) disagrees with the '
                f'harness (terminated={terminated}, truncated={truncated}) at t={self.t}.')
        if terminated or truncated:
            self._needs_reset = True

        info = self._info()
        info['team_reward'] = team
        info['local_rewards'] = np.asarray(local, dtype=np.float32)
        if terminated or truncated:
            info['success'] = self._success()
        return (
            self._obs_dict(),
            {ag: team for ag in self.agents},
            {ag: terminated for ag in self.agents},
            {ag: truncated for ag in self.agents},
            info,
        )

    def behaviour_actions(self, policy, epsilon, rng, joint_epsilon=False):
        """Scripted joint action with epsilon-uniform exploration.

        joint_epsilon=False: each agent independently explores w.p. epsilon (the
        og_marl/QMIX recorder's scheme). True: the whole team explores together, which
        couples the agents' marginals even when the base policy is decentralised.
        """
        if policy == 'random':
            return rng.integers(0, self.num_actions, size=self.num_agents)
        acts = np.asarray(self.scripted_actions(policy, rng), dtype=np.int64).copy()
        if epsilon > 0:
            rand = rng.integers(0, self.num_actions, size=self.num_agents)
            if joint_epsilon:
                if rng.random() < epsilon:
                    acts = rand
            else:
                flip = rng.random(self.num_agents) < epsilon
                acts = np.where(flip, rand, acts)
        return acts


# ---------------------------------------------------------------------- presets
def _presets():
    from envs import cognac_utils, mpe_discrete

    presets = {}
    presets.update({k: ('cognac', v) for k, v in cognac_utils.PRESETS.items()})
    presets.update({k: ('mpedisc', v) for k, v in mpe_discrete.PRESETS.items()})
    return presets


def list_presets():
    return {name: fam for name, (fam, _) in _presets().items()}


def is_harness_env_name(env_name):
    return env_name.split('_', 1)[0] in FAMILIES


def parse_env_name(env_name):
    """'<family>_<preset>_<uid>' -> (family, preset, uid)."""
    parts = env_name.split('_')
    if len(parts) != 3 or parts[0] not in FAMILIES:
        raise ValueError(
            f'Bad harness env name {env_name!r}: expected <family>_<preset>_<uid> with '
            f'family in {FAMILIES} and no "_" inside preset or uid.')
    family, preset, uid = parts
    presets = _presets()
    if preset not in presets:
        raise ValueError(f'Unknown preset {preset!r}. Known: {sorted(presets)}')
    if presets[preset][0] != family:
        raise ValueError(f'Preset {preset!r} belongs to family {presets[preset][0]!r}, not {family!r}.')
    return family, preset, uid


def make_env_from_preset(preset):
    family, builder = _presets()[preset]
    return builder()


def make_env_from_spec(spec):
    """Rebuild an env from the spec saved next to a dataset (graphs included)."""
    from envs import cognac_utils, mpe_discrete

    module = {'cognac': cognac_utils, 'mpedisc': mpe_discrete}[spec['family']]
    return module.make_env(spec)


# ---------------------------------------------------------------------- dataset I/O
def dataset_paths(env_name):
    family, preset, uid = parse_env_name(env_name)
    root = get_source_vault_dir('generated') / family / preset
    return root / f'{uid}.npz', root / f'{uid}.json'


def _git_head():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def save_dataset(env_name, arrays, metadata, overwrite=False):
    npz_path, json_path = dataset_paths(env_name)
    if not overwrite and (npz_path.exists() or json_path.exists()):
        raise FileExistsError(f'{npz_path} exists; pass --overwrite to replace it.')
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, harness_version=HARNESS_VERSION, git_head=_git_head(),
                    env_name=env_name)
    tmp = npz_path.with_name(npz_path.name + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz_path)
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return npz_path, json_path


def load_metadata(env_name):
    _, json_path = dataset_paths(env_name)
    if not json_path.exists():
        raise FileNotFoundError(
            f'No generated dataset for {env_name} ({json_path}). Create it with:\n'
            f'  python generate_dataset.py --env_name={env_name} --behaviour=<policy:eps:weight,...>')
    with open(json_path) as f:
        return json.load(f)


def get_dataset(env_name):
    """Transition dict with exactly TRANSITION_KEYS, as smac_utils.get_dataset returns."""
    npz_path, _ = dataset_paths(env_name)
    meta = load_metadata(env_name)
    if meta.get('harness_version') != HARNESS_VERSION:
        raise ValueError(f'{npz_path} was written by harness v{meta.get("harness_version")}, '
                         f'this is v{HARNESS_VERSION}; regenerate it.')
    with np.load(npz_path) as data:
        out = {k: np.asarray(data[k]) for k in TRANSITION_KEYS}
    for k in ('observations', 'next_observations', 'legals', 'next_legals',
              'states', 'next_states', 'rewards', 'terminals', 'truncations', 'masks'):
        out[k] = out[k].astype(np.float32)
    out['actions'] = out['actions'].astype(np.int32)
    return out


def make_eval_env(env_name):
    """Eval env rebuilt from the dataset's saved spec, so graphs/params cannot drift."""
    meta = load_metadata(env_name)
    env = make_env_from_spec(meta['env_spec'])
    if env.spec() != meta['env_spec']:
        raise RuntimeError(f'Rebuilt env spec differs from the dataset spec for {env_name}.')
    return env, meta
