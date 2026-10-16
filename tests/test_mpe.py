import jax
import jax.numpy as jnp
import numpy as np

from agents.mav_flows_continuous import MAVFlowContinuousAgent, get_config
from envs.mpe_omar import MPEOMAR
from envs.mpe_utils import experience_to_dataset, parse_env_name
from utils.evaluate_mpe import evaluate_mpe


def test_mpe_name_parser_preserves_vault_uid_case():
    assert parse_env_name('mpe_simple_spread_Expert') == ('simple_spread', 'Expert')
    assert parse_env_name('mpe_simple_spread_Medium-Replay') == (
        'simple_spread',
        'Medium-Replay',
    )


def test_mpe_experience_conversion_respects_episode_boundaries():
    observations = np.arange(1 * 4 * 3 * 2, dtype=np.float32).reshape(1, 4, 3, 2)
    actions = np.full((1, 4, 3, 2), 2.0, dtype=np.float32)
    rewards = np.repeat(np.arange(4, dtype=np.float32)[None, :, None], 3, axis=2)
    terminals = np.zeros((1, 4, 3), dtype=np.float32)
    terminals[:, 1, :] = 1.0
    truncations = np.zeros_like(terminals)
    truncations[:, 3, :] = 1.0
    experience = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'terminals': terminals,
        'truncations': truncations,
        'infos': {'state': np.arange(1 * 4 * 5, dtype=np.float32).reshape(1, 4, 5)},
    }

    dataset = experience_to_dataset(experience)
    np.testing.assert_array_equal(dataset['next_observations'][0], observations[0, 1])
    np.testing.assert_array_equal(dataset['next_observations'][1], observations[0, 1])
    np.testing.assert_array_equal(dataset['next_observations'][3], observations[0, 3])
    assert dataset['actions'].shape == (4, 3, 2)
    assert np.max(dataset['actions']) == 1.0
    np.testing.assert_array_equal(dataset['terminals'], [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(dataset['masks'], [1.0, 0.0, 1.0, 1.0])


def test_mpe_experience_conversion_flattens_parallel_buffers_without_crossing():
    observations = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 1, 3)
    experience = {
        'observations': observations,
        'actions': np.zeros((2, 2, 1, 2), dtype=np.float32),
        'rewards': np.zeros((2, 2, 1), dtype=np.float32),
        'terminals': np.zeros((2, 2, 1), dtype=np.float32),
        'truncations': np.zeros((2, 2, 1), dtype=np.float32),
        'infos': {},
    }
    dataset = experience_to_dataset(experience)
    assert dataset['observations'].shape == (4, 1, 3)
    np.testing.assert_array_equal(dataset['next_observations'][1], observations[0, 1])
    np.testing.assert_array_equal(dataset['next_observations'][3], observations[1, 1])
    np.testing.assert_array_equal(dataset['terminals'], [0.0, 1.0, 0.0, 1.0])


def _small_agent():
    config = get_config()
    config.update(
        actor_hidden_dims=(16,),
        value_hidden_dims=(16,),
        flow_steps=2,
        batch_size=4,
    )
    example = {
        'observations': jnp.zeros((1, 3, 6), dtype=jnp.float32),
        'next_observations': jnp.zeros((1, 3, 6), dtype=jnp.float32),
        'actions': jnp.zeros((1, 3, 2), dtype=jnp.float32),
        'rewards': jnp.zeros((1,), dtype=jnp.float32),
        'masks': jnp.ones((1,), dtype=jnp.float32),
        'min_reward': -1.0,
        'max_reward': 1.0,
    }
    return MAVFlowContinuousAgent.create(0, example, config)


def test_continuous_agent_updates_and_returns_action_dict():
    agent = _small_agent()
    batch = {
        'observations': jnp.zeros((4, 3, 6), dtype=jnp.float32),
        'next_observations': jnp.ones((4, 3, 6), dtype=jnp.float32),
        'actions': jnp.zeros((4, 3, 2), dtype=jnp.float32),
        'rewards': jnp.zeros((4,), dtype=jnp.float32),
        'masks': jnp.ones((4,), dtype=jnp.float32),
    }
    updated, info = agent.update(batch)
    assert jnp.isfinite(info['total_loss'])
    assert updated.network.step == agent.network.step + 1

    observations = {f'agent_{index}': np.zeros(6, dtype=np.float32) for index in range(3)}
    actions = updated.sample_actions(observations, seed=jax.random.PRNGKey(3))
    assert set(actions) == set(observations)
    assert all(action.shape == (2,) for action in actions.values())
    assert all(jnp.all(jnp.abs(action) <= 1.0) for action in actions.values())


def test_mpe_evaluation_reports_return_and_length():
    class FakeAgent:
        def sample_actions(self, observations, seed=None):
            del seed
            return {name: jnp.zeros(2) for name in observations}

    class FakeEnv:
        def reset(self):
            self.step_count = 0
            return {f'agent_{index}': np.zeros(6) for index in range(3)}, {}

        def step(self, actions):
            self.step_count += 1
            observations = {name: np.zeros(6) for name in actions}
            rewards = {name: 2.0 for name in actions}
            done = self.step_count == 3
            terminals = {name: done for name in actions}
            truncations = {name: False for name in actions}
            return observations, rewards, terminals, truncations, {}

    metrics = evaluate_mpe(FakeAgent(), FakeEnv(), num_episodes=2, seed=0)
    assert metrics['mean_return'] == 6.0
    assert metrics['mean_length'] == 3.0


def test_mpe_wrapper_has_independent_seeded_rng_and_25_step_horizon():
    np.random.seed(99)
    expected_global_draw = np.random.RandomState(99).rand()
    env_a = MPEOMAR('simple_spread', seed=7)
    env_b = MPEOMAR('simple_spread', seed=7)
    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()

    assert all(np.allclose(obs_a[name], obs_b[name]) for name in obs_a)
    assert np.isclose(np.random.rand(), expected_global_draw)

    actions = {name: np.zeros(2, dtype=np.float32) for name in obs_a}
    episode_length = 0
    while True:
        obs_a, _, terminals, truncations, _ = env_a.step(actions)
        episode_length += 1
        if all(terminals.values()) or all(truncations.values()):
            break
    assert episode_length == 25
