"""Tests for the COGNAC / discrete-MPE harness (envs/marl_harness.py).

Run:  python -m tests.test_marl_harness

The core guarantee under test is the REWARD ROUND TRIP: a dataset written by
generate_dataset.py, read back through the same loader main.py uses, and the evaluator
replaying the same policy on the same seeds must agree on every episode's return.
"""

import os
import subprocess
import sys
import tempfile

import jax
import numpy as np

from envs import cognac_utils, marl_harness
from utils.evaluate_marl import evaluate_marl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_PRESETS = sorted(marl_harness.list_presets())


def _family(preset):
    return marl_harness.list_presets()[preset]


def _run_episode(env, seed, policy='central', eps=0.0, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    obs, info = env.reset(seed=seed)
    traj = [(obs, info['state'])]
    total, flags = 0.0, None
    while True:
        a = env.behaviour_actions(policy, eps, rng)
        obs, rewards, te, tr, info = env.step(a)
        total += info['team_reward']
        traj.append((obs, info['state']))
        if all(te.values()) or all(tr.values()):
            flags = (all(te.values()), all(tr.values()))
            return total, traj, flags, info


class _ScriptedAgent:
    """Stands in for MAVFlowAgent: replays an env behaviour policy (eps=0)."""

    def __init__(self, env, policy):
        self.env, self.policy = env, policy

    def sample_actions_rejection(self, obs_with_id, legals, rng):
        del obs_with_id, legals, rng
        acts = self.env.behaviour_actions(self.policy, 0.0, np.random.default_rng(0))
        return acts[None]


def test_name_parsing():
    assert marl_harness.parse_env_name('cognac_ffrow-8_central-eps0.2') == ('cognac', 'ffrow-8', 'central-eps0.2')
    assert marl_harness.parse_env_name('mpedisc_spread-3_mix') == ('mpedisc', 'spread-3', 'mix')
    for bad in ('cognac_ffrow-8', 'cognac_ffrow_8_x', 'mpedisc_ffrow-8_x', 'cognac_nope_x'):
        try:
            marl_harness.parse_env_name(bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad} should not parse')
    assert marl_harness.is_harness_env_name('cognac_ffrow-8_x')
    assert not marl_harness.is_harness_env_name('smac_3m_good')
    assert not marl_harness.is_harness_env_name('mpe_simple_spread_Expert')


def test_seeded_episodes_are_reproducible_and_leave_global_rng_alone():
    for preset in ALL_PRESETS:
        np.random.seed(123)
        expected = np.random.RandomState(123).rand()
        env_a = marl_harness.make_env_from_preset(preset)
        env_b = marl_harness.make_env_from_preset(preset)
        # Play a different episode on env_b first: the next episode must not depend on it
        # (catches Firefighting's last_visited_house leak across resets).
        _run_episode(env_b, seed=99, policy='random', rng_seed=5)
        r_a, traj_a, _, _ = _run_episode(env_a, seed=7, policy='random', rng_seed=1)
        r_b, traj_b, _, _ = _run_episode(env_b, seed=7, policy='random', rng_seed=1)
        assert r_a == r_b, preset
        assert len(traj_a) == len(traj_b), preset
        for (oa, sa), (ob, sb) in zip(traj_a, traj_b):
            assert np.array_equal(sa, sb), preset
            assert all(np.array_equal(oa[k], ob[k]) for k in oa), preset
        assert np.random.rand() == expected, f'{preset} consumed the global numpy RNG'
        r_c, _, _, _ = _run_episode(env_a, seed=8, policy='random', rng_seed=1)
        assert r_c != r_a, f'{preset}: different seeds gave identical episodes'


def test_termination_contract():
    for preset in ALL_PRESETS:
        env = marl_harness.make_env_from_preset(preset)
        for seed in range(15):
            _, traj, (terminated, truncated), info = _run_episode(env, seed, policy='random', rng_seed=seed)
            steps = len(traj) - 1
            assert terminated != truncated, preset
            assert steps <= env.horizon, (preset, steps)
            if truncated:
                assert steps == env.horizon, (preset, steps)
            if terminated:
                assert env._is_absorbing(), preset
            if preset.startswith(('sysadmin', 'spread')):
                assert truncated and not terminated, f'{preset} has no absorbing state'
            if info.get('success') is not None:
                assert info['success'] == terminated, preset


def test_team_reward_matches_independent_recomputation():
    # Firefighting: upstream local reward is -fire level at the house each agent
    # visited, read AFTER the transition. Recompute it from env internals.
    env = marl_harness.make_env_from_preset('ffrow-8')
    rng = np.random.default_rng(0)
    for seed in range(5):
        env.reset(seed=seed)
        while True:
            a = env.behaviour_actions('random', 0.0, rng)
            _, rewards, te, tr, info = env.step(a)
            houses = [i + int(a[i]) for i in range(env.num_agents)]
            assert info['team_reward'] == -float(np.sum(env.env.state[houses]))
            assert all(r == info['team_reward'] for r in rewards.values())
            if all(te.values()) or all(tr.values()):
                break
    # Every env: team reward is the upstream sum of local rewards.
    for preset in ALL_PRESETS:
        env = marl_harness.make_env_from_preset(preset)
        env.reset(seed=3)
        for _ in range(5):
            _, _, te, tr, info = env.step(env.behaviour_actions('random', 0.0, rng))
            assert np.isclose(info['team_reward'], info['local_rewards'].astype(np.float64).sum())
            if all(te.values()) or all(tr.values()):
                break


def test_spec_round_trip_rebuilds_identical_env():
    for preset in ALL_PRESETS:
        env = marl_harness.make_env_from_preset(preset)
        import json
        spec = json.loads(json.dumps(env.spec()))
        rebuilt = marl_harness.make_env_from_spec(spec)
        assert rebuilt.spec() == spec, preset
        r1, _, _, _ = _run_episode(env, seed=11, policy='random', rng_seed=2)
        r2, _, _, _ = _run_episode(rebuilt, seed=11, policy='random', rng_seed=2)
        assert r1 == r2, preset


def test_vendored_cognac_imports_without_pettingzoo_or_networkx():
    from envs._vendor.cognac import _compat
    assert hasattr(_compat, 'ParallelEnv')
    assert cognac_utils.PRESETS


def _generate(tmp, env_name, behaviour, n):
    env = dict(os.environ, MARL_VAULT_ROOT=tmp, JAX_PLATFORMS='cpu')
    out = subprocess.run(
        [sys.executable, 'generate_dataset.py', f'--env_name={env_name}',
         f'--behaviour={behaviour}', f'--num_transitions={n}', '--data_seed=0'],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]


def test_reward_round_trip_generator_loader_evaluator():
    """Dataset rewards (as main.py loads them) == evaluator returns on the same seeds."""
    old = os.environ.get('MARL_VAULT_ROOT')
    with tempfile.TemporaryDirectory() as tmp:
        try:
            for preset in ALL_PRESETS:
                env_name = f'{_family(preset)}_{preset}_test'
                _generate(tmp, env_name, 'central:0.0:1.0', n=120)
                os.environ['MARL_VAULT_ROOT'] = tmp
                data = marl_harness.get_dataset(env_name)
                meta = marl_harness.load_metadata(env_name)

                assert set(data) == set(marl_harness.TRANSITION_KEYS)
                T, K, O = data['observations'].shape
                assert data['actions'].shape == (T, K) and data['actions'].dtype == np.int32
                assert data['legals'].shape == (T, K, meta['num_actions'])
                assert data['states'].shape == (T, meta['state_dim'])
                for k in ('rewards', 'terminals', 'truncations', 'masks'):
                    assert data[k].shape == (T,), k
                # masks == 0 exactly at absorbing ends; terminals == 1 at every end.
                term = data['terminals'] > 0
                absorbing = data['masks'] == 0
                assert np.all(term[absorbing]) and np.all(term == (absorbing | (data['truncations'] > 0)))
                # next_* chains inside episodes.
                inner = np.nonzero(~term[:-1])[0]
                assert np.array_equal(data['next_observations'][inner], data['observations'][inner + 1])
                assert np.array_equal(data['next_states'][inner], data['states'][inner + 1])

                ends = np.nonzero(term)[0]
                starts = np.concatenate([[0], ends[:-1] + 1])
                data_returns = np.array([data['rewards'][s:e + 1].astype(np.float64).sum()
                                         for s, e in zip(starts, ends)])

                eval_env, _ = marl_harness.make_eval_env(env_name)
                res = evaluate_marl(_ScriptedAgent(eval_env, 'central'), eval_env,
                                    num_episodes=len(data_returns), seed=0, episode_seed_base=0)
                np.testing.assert_allclose(res['raw_returns'], data_returns, rtol=1e-5, atol=1e-3,
                                           err_msg=preset)
                assert res['episode_seeds'] == list(range(len(data_returns)))
        finally:
            if old is None:
                os.environ.pop('MARL_VAULT_ROOT', None)
            else:
                os.environ['MARL_VAULT_ROOT'] = old


def test_mav_agent_trains_and_evaluates_on_harness_data():
    from agents.mav_flows import MAVFlowAgent, get_config
    from utils.datasets import Dataset

    old = os.environ.get('MARL_VAULT_ROOT')
    with tempfile.TemporaryDirectory() as tmp:
        try:
            env_name = 'cognac_ffrow-8_smoke'
            _generate(tmp, env_name, 'central:0.3:0.5,random:1.0:0.5', n=400)
            os.environ['MARL_VAULT_ROOT'] = tmp
            ds = Dataset.create(**marl_harness.get_dataset(env_name))
            config = get_config()
            config.update(dict(value_hidden_dims=(16,), actor_hidden_dims=(16,), num_flow_steps=2,
                               num_candidates=2, num_target_candidates=2, num_q_candidates=2,
                               teacher_mc_samples=2, batch_size=8))
            example = ds.sample(1)
            example['min_reward'] = float(ds['rewards'].min())
            example['max_reward'] = float(ds['rewards'].max())
            agent = MAVFlowAgent.create(0, example, config)
            agent, info = agent.update(ds.sample(8))
            assert all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree.leaves(info))
            eval_env, _ = marl_harness.make_eval_env(env_name)
            res = evaluate_marl(agent, eval_env, num_episodes=2, seed=0)
            assert np.isfinite(res['mean_return']) and res['success_rate'] is not None
        finally:
            if old is None:
                os.environ.pop('MARL_VAULT_ROOT', None)
            else:
                os.environ['MARL_VAULT_ROOT'] = old


if __name__ == '__main__':
    test_name_parsing()
    test_vendored_cognac_imports_without_pettingzoo_or_networkx()
    test_seeded_episodes_are_reproducible_and_leave_global_rng_alone()
    test_termination_contract()
    test_team_reward_matches_independent_recomputation()
    test_spec_round_trip_rebuilds_identical_env()
    test_reward_round_trip_generator_loader_evaluator()
    test_mav_agent_trains_and_evaluates_on_harness_data()
    print('All MARL harness tests passed.')
