"""Generate an offline dataset for a harness env (COGNAC / discrete MPE).

    python generate_dataset.py --env_name=cognac_ffrow-8_central-eps0.2 \
        --behaviour=central:0.2:1.0 --num_transitions=1000000 --data_seed=0

--behaviour is a comma list of policy:epsilon:weight. Each EPISODE draws one component
by weight (an og_marl "Replay"-style mixture when there are several). Policies:
  random   uniform independent per-agent actions (factorised by construction)
  local    decentralised, uses only the agent's own observation
  central  uses the global state to pick a COUPLED joint action (e.g. two firefighters
           doubling up on one house, Hungarian-style landmark assignment in MPE), so
           the joint behaviour policy does NOT factorise across agents.
Epsilon is per-agent independent by default; --joint_epsilon makes the whole team
explore together.

The uid (last '_' field of env_name) is just a name; the behaviour spec, env spec
(graph included), seeds and summary stats are written to <uid>.json next to the data.
Writes $MARL_VAULT_ROOT/generated/<family>/<preset>/<uid>.{npz,json}.
"""

import time

import numpy as np
from absl import app, flags

from envs import marl_harness

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', None, '<family>_<preset>_<uid>', required=True)
flags.DEFINE_string('behaviour', None, 'policy:epsilon:weight[,policy:epsilon:weight...]',
                    required=True)
flags.DEFINE_integer('num_transitions', 1_000_000, 'Stop after the episode that reaches this.')
flags.DEFINE_integer('data_seed', 0, 'Dataset seed (0-99); episode seeds are disjoint from eval.')
flags.DEFINE_bool('joint_epsilon', False, 'Explore as a team instead of per agent.')
flags.DEFINE_bool('overwrite', False, 'Replace an existing dataset with the same name.')

SEED_STRIDE = 10_000_000


def parse_behaviour(spec, env):
    components = []
    for item in spec.split(','):
        try:
            policy, eps, weight = item.split(':')
            eps, weight = float(eps), float(weight)
        except ValueError as e:
            raise ValueError(f'Bad behaviour component {item!r}; expected policy:epsilon:weight.') from e
        if policy not in env.behaviour_policies():
            raise ValueError(f'{env.preset} has no policy {policy!r}; '
                             f'available: {env.behaviour_policies()}')
        if not (0.0 <= eps <= 1.0) or weight <= 0:
            raise ValueError(f'Bad epsilon/weight in {item!r}.')
        components.append(dict(policy=policy, epsilon=eps, weight=weight))
    total = sum(c['weight'] for c in components)
    for c in components:
        c['weight'] /= total
    return components


def rollout(env, component, episode_seed, rng, joint_epsilon):
    obs, info = env.reset(seed=episode_seed)
    steps = []
    while True:
        o = np.stack([obs[ag] for ag in env.agents])
        legals = np.stack([info['legals'][ag] for ag in env.agents])
        state = info['state']
        acts = env.behaviour_actions(component['policy'], component['epsilon'], rng,
                                     joint_epsilon=joint_epsilon)
        obs, rewards, terms, truncs, info = env.step({ag: int(acts[i]) for i, ag in enumerate(env.agents)})
        terminated = all(terms.values())
        truncated = all(truncs.values())
        assert terminated == any(terms.values()) and truncated == any(truncs.values())
        team = info['team_reward']
        assert all(r == team for r in rewards.values()), 'team reward must be replicated'
        steps.append(dict(
            observations=o, actions=acts.astype(np.int32), legals=legals, states=state,
            rewards=team, terminated=terminated, truncated=truncated,
            next_observations=np.stack([obs[ag] for ag in env.agents]),
            next_legals=np.stack([info['legals'][ag] for ag in env.agents]),
            next_states=info['state'],
        ))
        if terminated or truncated:
            return steps, info.get('success')


def main(_):
    family, preset, uid = marl_harness.parse_env_name(FLAGS.env_name)
    if not 0 <= FLAGS.data_seed < 100:
        raise ValueError('--data_seed must be in [0, 100).')
    env = marl_harness.make_env_from_preset(preset)
    components = parse_behaviour(FLAGS.behaviour, env)
    npz_path, _ = marl_harness.dataset_paths(FLAGS.env_name)
    if npz_path.exists() and not FLAGS.overwrite:
        raise FileExistsError(f'{npz_path} exists; pass --overwrite to replace it.')

    print(f'{FLAGS.env_name}: K={env.num_agents} A={env.num_actions} obs={env.obs_dim} '
          f'state={env.state_dim} horizon={env.horizon}')
    print('behaviour:', components)

    buf = {k: [] for k in ('observations', 'actions', 'legals', 'states', 'rewards',
                           'terminated', 'truncated', 'next_observations',
                           'next_legals', 'next_states')}
    ep_returns, ep_lengths, ep_seeds, ep_component, ep_success = [], [], [], [], []
    weights = np.array([c['weight'] for c in components])
    total, ep, t0 = 0, 0, time.time()
    while total < FLAGS.num_transitions:
        episode_seed = FLAGS.data_seed * SEED_STRIDE + ep
        if episode_seed >= marl_harness.EVAL_SEED_BASE or ep >= SEED_STRIDE:
            raise RuntimeError('Ran out of training episode seeds.')
        rng = np.random.default_rng([FLAGS.data_seed, ep])
        c_idx = int(rng.choice(len(components), p=weights))
        steps, success = rollout(env, components[c_idx], episode_seed, rng, FLAGS.joint_epsilon)
        for s in steps:
            for k in buf:
                buf[k].append(s[k])
        ep_returns.append(float(sum(s['rewards'] for s in steps)))
        ep_lengths.append(len(steps))
        ep_seeds.append(episode_seed)
        ep_component.append(c_idx)
        ep_success.append(-1 if success is None else int(success))
        total += len(steps)
        ep += 1
        if ep % 1000 == 0:
            print(f'  {ep} episodes, {total} transitions, {total / (time.time() - t0):.0f} steps/s')

    terminated = np.asarray(buf['terminated'], dtype=np.float32)
    truncated = np.asarray(buf['truncated'], dtype=np.float32)
    arrays = dict(
        observations=np.asarray(buf['observations'], dtype=np.float32),
        next_observations=np.asarray(buf['next_observations'], dtype=np.float32),
        actions=np.asarray(buf['actions'], dtype=np.int32),
        legals=np.asarray(buf['legals'], dtype=np.float32),
        next_legals=np.asarray(buf['next_legals'], dtype=np.float32),
        states=np.asarray(buf['states'], dtype=np.float32),
        next_states=np.asarray(buf['next_states'], dtype=np.float32),
        rewards=np.asarray(buf['rewards'], dtype=np.float32),
        # Same semantics as envs/smac_utils.get_dataset:
        terminals=((terminated + truncated) > 0).astype(np.float32),   # any episode end
        truncations=truncated,
        masks=(1.0 - terminated).astype(np.float32),                     # bootstrap unless absorbing
        episode_returns=np.asarray(ep_returns, dtype=np.float64),
        episode_lengths=np.asarray(ep_lengths, dtype=np.int64),
        episode_seeds=np.asarray(ep_seeds, dtype=np.int64),
        episode_component=np.asarray(ep_component, dtype=np.int64),
        episode_success=np.asarray(ep_success, dtype=np.int64),
    )
    assert not np.any((terminated > 0) & (truncated > 0))

    # Reward bookkeeping must close: per-episode sums of the stored per-step rewards
    # (float32, split at `terminals`) reproduce the float64 episode returns.
    ends = np.nonzero(arrays['terminals'])[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    resummed = np.array([arrays['rewards'][s:e + 1].astype(np.float64).sum() for s, e in zip(starts, ends)])
    assert len(resummed) == len(ep_returns)
    np.testing.assert_allclose(resummed, ep_returns, rtol=1e-5, atol=1e-3)

    success = np.asarray(ep_success)
    per_component = []
    for i, c in enumerate(components):
        sel = arrays['episode_component'] == i
        per_component.append(dict(c, episodes=int(sel.sum()),
                                  mean_return=float(np.mean(arrays['episode_returns'][sel])) if sel.any() else None))
    metadata = dict(
        env_spec=env.spec(),
        behaviour=components,
        joint_epsilon=FLAGS.joint_epsilon,
        data_seed=FLAGS.data_seed,
        num_agents=env.num_agents, num_actions=env.num_actions,
        obs_dim=env.obs_dim, state_dim=env.state_dim, horizon=env.horizon,
        num_transitions=int(total), num_episodes=int(ep),
        mean_return=float(np.mean(ep_returns)), std_return=float(np.std(ep_returns)),
        min_return=float(np.min(ep_returns)), max_return=float(np.max(ep_returns)),
        mean_length=float(np.mean(ep_lengths)),
        success_rate=float(np.mean(success[success >= 0])) if np.any(success >= 0) else None,
        per_component=per_component,
        reward_definition='team reward = sum of upstream local rewards per step, replicated per agent',
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
    )
    npz_path, json_path = marl_harness.save_dataset(FLAGS.env_name, arrays, metadata,
                                                    overwrite=FLAGS.overwrite)
    print(f'wrote {npz_path} ({total} transitions, {ep} episodes)')
    print(f'  return {metadata["mean_return"]:.3f} +- {metadata["std_return"]:.3f}  '
          f'success {metadata["success_rate"]}  length {metadata["mean_length"]:.1f}')


if __name__ == '__main__':
    app.run(main)
