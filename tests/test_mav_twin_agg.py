import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.mav_flows import MAVFlowAgent, aggregate_twins, get_config


B, K, A = 8, 3, 5
OBS_DIM, STATE_DIM = 6, 9


def _config(**overrides):
    config = get_config()
    config.update(dict(
        value_hidden_dims=(16,),
        actor_hidden_dims=(16,),
        num_flow_steps=3,
        num_candidates=2,
        num_target_candidates=2,
        num_q_candidates=2,
        teacher_mc_samples=2,
    ))
    config.update(overrides)
    return config


def _data(seed=0):
    rng = np.random.default_rng(seed)

    def rand(*shape):
        return jnp.asarray(rng.standard_normal(shape), dtype=jnp.float32)

    def legals(batch_size):
        values = jnp.asarray((rng.random((batch_size, K, A)) > 0.25).astype(np.float32))
        return values.at[:, :, 0].set(1.0)

    example_batch = dict(
        observations=rand(1, K, OBS_DIM), states=rand(1, STATE_DIM),
        actions=jnp.asarray(rng.integers(0, A, (1, K))), legals=legals(1),
        min_reward=-1.0, max_reward=1.0,
    )
    batch_legals = legals(B)
    batch = dict(
        observations=rand(B, K, OBS_DIM), next_observations=rand(B, K, OBS_DIM),
        states=rand(B, STATE_DIM), next_states=rand(B, STATE_DIM),
        actions=jnp.asarray(rng.integers(0, A, (B, K))),
        legals=batch_legals, next_legals=batch_legals,
        rewards=rand(B), masks=jnp.asarray(rng.integers(0, 2, B), dtype=jnp.float32),
    )
    return example_batch, batch


def _agent(**overrides):
    example_batch, batch = _data()
    return MAVFlowAgent.create(0, example_batch, _config(**overrides)), batch


def _joint(batch):
    return jax.nn.one_hot(batch['actions'], A).reshape(B, K * A)


def _dcfm(agent, batch):
    eps = jax.random.normal(jax.random.PRNGKey(7), (B, 1))
    t = jax.random.uniform(jax.random.PRNGKey(8), (B, 1))
    return agent.joint_critic_dcfm_loss(
        batch, agent.network.params, eps, t, _joint(batch), _joint(batch))


def _with_identical_target_twins(agent):
    params = dict(agent.network.params)
    params['modules_target_joint_critic_flow2'] = params['modules_target_joint_critic_flow1']
    return agent.replace(network=agent.network.replace(params=params))


def test_new_runs_default_to_mean_everywhere():
    # 10/15: MAC-Flow App. E and the Value Flows default.
    config = get_config()
    assert (config['ret_agg'], config['dcfm_agg'], config['q_agg']) == ('mean', 'mean', 'mean')


def test_aggregate_twins():
    x1, x2 = jnp.array([1.0, 4.0]), jnp.array([3.0, 2.0])
    np.testing.assert_array_equal(aggregate_twins(x1, x2, 'min'), [1.0, 2.0])
    np.testing.assert_array_equal(aggregate_twins(x1, x2, 'mean'), [2.0, 3.0])
    with pytest.raises(ValueError):
        aggregate_twins(x1, x2, 'typo')


def test_dcfm_agg_is_moot_when_target_twins_coincide():
    # With identical target twins, min and mean of (z, z) are both z, so dcfm_agg acts
    # only through the twins' disagreement.
    mean, batch = _agent(dcfm_agg='mean')
    mn, _ = _agent(dcfm_agg='min')
    loss_mean, info_mean = _dcfm(_with_identical_target_twins(mean), batch)
    loss_min, info_min = _dcfm(_with_identical_target_twins(mn), batch)
    np.testing.assert_allclose(loss_mean, loss_min, rtol=1e-6)


def _dcfm_reference(agent, batch):
    # The DCFM loss written out: target point and velocity are dcfm_agg over both target
    # twins at the shared (eps, t), and both online twins regress at that point.
    eps = jax.random.normal(jax.random.PRNGKey(7), (B, 1))
    t = jax.random.uniform(jax.random.PRNGKey(8), (B, 1))
    agg, ja, ns = agent.config['dcfm_agg'], _joint(batch), batch['next_states']
    names = ('target_joint_critic_flow1', 'target_joint_critic_flow2')
    znext = aggregate_twins(*[agent.compute_flow_returns(eps, ns, ja, end_times=t, flow_network_name=n)
                              for n in names], agg)
    gm = agent.config['discount'] * batch['masks'][:, None]
    # 10/21: the target velocity is gamma*m times the target field (tests/test_mav_dcfm_mask.py).
    target = gm * aggregate_twins(*[agent.network.select(n)(znext, t, ns, ja) for n in names], agg)
    x = batch['rewards'][:, None] + gm * znext
    return sum(((agent.network.select(n)(x, t, batch['states'], ja) - target) ** 2).mean()
               for n in ('joint_critic_flow1', 'joint_critic_flow2'))


def test_dcfm_loss_uses_the_dcfm_agg_target():
    # WAS (pre-10/19): read the min-vs-mean target off the logged noisy_next_returns_mean,
    # which was dropped from the logs.
    losses = {}
    for agg in ('mean', 'min'):
        agent, batch = _agent(dcfm_agg=agg)
        losses[agg] = float(_dcfm(agent, batch)[0])
        np.testing.assert_allclose(losses[agg], float(_dcfm_reference(agent, batch)), rtol=1e-5)
    assert losses['mean'] != losses['min']


def test_unknown_aggregation_fails_early():
    for key in ('ret_agg', 'dcfm_agg', 'q_agg'):
        agent, batch = _agent(**{key: 'typo'})
        with pytest.raises(ValueError):
            agent.total_loss(batch, agent.network.params, jax.random.PRNGKey(3))


def test_update_runs_and_reports_finite_losses():
    agent, batch = _agent()
    agent, info = agent.update(batch)
    assert all(np.isfinite(np.asarray(v)).all() for v in info.values())
    assert not any(k.startswith('cw/') for k in info)
