"""SMAC eval via OG-MARL's gym-style API.

Mirrors MAC-Flow's _evaluate() function but adapted for MAV-Flow's tensor-based
sample_actions. Works for both SMACv1 and SMACv2 — OG-MARL abstracts the difference.

Reports mean episodic return, matching MAC-Flow's Table 1 convention.
"""

import numpy as np
import jax
from tqdm import trange

from utils.marl_utils import batch_concat_agent_id_to_obs


def _extract_smac_win(info):
    """Return SMAC's explicit terminal win flag, including nested wrappers."""
    if not isinstance(info, dict):
        return None
    for key in ('battle_won', 'won', 'win'):
        if key in info:
            value = np.asarray(info[key])
            if value.size == 1:
                return bool(value.item())
            return bool(value.all())
    for value in info.values():
        result = _extract_smac_win(value)
        if result is not None:
            return result
    return None


def _get_smac_battles_won(env):
    """Read SMAC's cumulative win counter through OG-MARL wrappers."""
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, 'battles_won'):
            return int(current.battles_won)
        get_stats = getattr(current, 'get_stats', None)
        if callable(get_stats):
            try:
                stats = get_stats()
            except (AttributeError, KeyError, TypeError, ZeroDivisionError):
                stats = None
            if isinstance(stats, dict) and 'battles_won' in stats:
                return int(stats['battles_won'])
        current = getattr(current, '_environment', None)
    return None


def evaluate_smac(agent, env, num_episodes=50, seed=0, verbose=False, arm='deployed',
                  diagnostics=True, return_steps=False, exact_check_states=200):
    """Run num_episodes rollouts of one eval arm.

    Args:
        agent: trained MAVFlowAgent.
        env: OG-MARL env from get_environment(...).
        num_episodes: number of full rollouts.
        seed: RNG seed for the agent's sampling keys (the env is not reseeded).
        verbose: tqdm progress bar.
        arm: which joint action to play (MAVFlowAgent.eval_step):
            'deployed' -- per-agent argmax of q_i over the N BC candidates (the policy);
            'bc'       -- candidate 0, i.e. one BC draw: the no-selection floor;
            'teacher'  -- argmax of the joint critic over the same N candidates. It reads
                          the global state, so it is a centralised reference, not a
                          deployable policy.
        diagnostics: also return on-policy diagnostics (`onpolicy`, see
            onpolicy_metrics) computed on the states this arm visits.
        return_steps: also return the per-step records behind them (`steps`; probes).
        exact_check_states: agents with exact_max (agents/qmix_cql.py, critic='joint'):
            on this many visited states, also find the exhaustive argmax to check how
            close the greedy (coordinate ascent) gets.

    Agents with an `act` method (the scalar-Q baselines in agents/qmix_cql.py) play their
    greedy joint action; only arm='deployed' exists for them, and `onpolicy` holds
    scalar_onpolicy_metrics.

    Returns:
        dict with return statistics, win_rate, win_rate_se, mean_length, raw_returns,
        raw_wins, and `onpolicy` when diagnostics ran.
    """
    use_act = hasattr(agent, 'act')
    if use_act and arm != 'deployed':
        raise ValueError(f'{type(agent).__name__} has only the deployed arm, got {arm!r}')
    use_step = not use_act and hasattr(agent, 'eval_step') and (diagnostics or arm != 'deployed')
    gamma = float(agent.config['discount'])
    exact_left = exact_check_states if (use_act and diagnostics and hasattr(agent, 'exact_max')
                                        and agent.config.get('critic') == 'joint') else 0
    rng = jax.random.PRNGKey(seed)
    returns, disc_returns, lengths, wins = [], [], [], []
    steps = []   # per-step diagnostic records, all episodes
    iterator = trange(num_episodes, desc=f'SMAC eval [{arm}]', dynamic_ncols=True) if verbose else range(num_episodes)

    for _ in iterator:
        wins_before = _get_smac_battles_won(env)
        obs_dict, info = env.reset()
        agent_names = list(obs_dict.keys())   # e.g. ['agent_0', 'agent_1', 'agent_2']

        episode_return = 0.0
        episode_len = 0
        ep_steps, ep_rewards = [], []

        while True:
            # Dict → Tensor: stack per-agent dict → (1, K, obs_dim)
            obs_arr = np.stack([obs_dict[ag] for ag in agent_names], axis=0)[None]
            legals_arr = np.stack(
                [info['legals'][ag] for ag in agent_names], axis=0)[None]

            obs_with_id = batch_concat_agent_id_to_obs(obs_arr)

            # Sample actions
            rng, act_rng = jax.random.split(rng)
            if use_act:
                state = np.asarray(info['state'], dtype=np.float32).reshape(1, -1)
                acts, q = agent.act(obs_with_id, legals_arr, state, act_rng)
                actions_np = np.array(acts[0])
                if diagnostics:
                    rec = dict(v_exec_ode=float(q[0]))
                    if exact_left > 0:
                        a_ex, q_ex = agent.exact_max(state, legals_arr)
                        rec.update(exact_gap=float(q_ex[0]) - float(q[0]),
                                   exact_hit=bool((np.asarray(a_ex[0]) == actions_np).all()))
                        exact_left -= 1
                    ep_steps.append(rec)
            elif use_step:
                if 'state' not in info:
                    raise KeyError(f"arm {arm!r} and the eval diagnostics need info['state'], "
                                   f"which this env does not provide")
                state = np.asarray(info['state'], dtype=np.float32).reshape(1, -1)
                out = jax.device_get(agent.eval_step(obs_with_id, legals_arr, state, act_rng))
                actions_np = np.array(out[arm][0])
                ep_steps.append(_step_record(out, arm))
            else:
                actions_tensor = agent.sample_actions_rejection(obs_with_id, legals_arr, act_rng)
                actions_np = np.asarray(actions_tensor[0])

            # Defensive: ensure every action is legal
            for i, ag in enumerate(agent_names):
                if info['legals'][ag][actions_np[i]] == 0:
                    actions_np[i] = int(np.argmax(info['legals'][ag]))

            # Tensor → Dict
            actions_dict = {ag: int(actions_np[i]) for i, ag in enumerate(agent_names)}

            obs_dict, rewards, terminals, truncations, info = env.step(actions_dict)

            r = float(np.mean(list(rewards.values())))
            episode_return += r
            ep_rewards.append(r)
            episode_len += 1

            done = all(terminals.values()) or all(truncations.values())
            if done:
                break

        returns.append(episode_return)
        disc_returns.append(float(np.sum(np.asarray(ep_rewards) * gamma ** np.arange(len(ep_rewards)))))
        episode_won = _extract_smac_win(info)
        if episode_won is None:
            wins_after = _get_smac_battles_won(env)
            if wins_before is None or wins_after is None:
                raise KeyError(
                    'SMAC exposed neither a terminal win flag nor a cumulative '
                    'battles_won statistic. '
                    f'Available info keys: '
                    f'{tuple(info.keys()) if isinstance(info, dict) else type(info)}')
            episode_won = wins_after > wins_before
        wins.append(episode_won)
        lengths.append(episode_len)
        if ep_steps:
            # Realised discounted return-to-go from every visited state, for calibration.
            g = 0.0
            for rec, r in zip(reversed(ep_steps), reversed(ep_rewards)):
                g = r + gamma * g
                rec['G'] = g
            ep_steps[0]['t0'] = True
            for rec in ep_steps:
                rec['ep'], rec['won'], rec['gamma'] = len(returns) - 1, bool(episode_won), gamma
            steps.extend(ep_steps)

        if verbose:
            iterator.set_postfix({
                'ret': f'{episode_return:.2f}',
                'len': episode_len,
                'mean': f'{np.mean(returns):.2f}',
                'win': int(episode_won),
            })

    win_rate = float(np.mean(wins))
    out = {
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'max_return': float(np.max(returns)),
        'min_return': float(np.min(returns)),
        'win_rate': win_rate,
        'win_rate_se': float(np.sqrt(win_rate * (1 - win_rate) / len(wins))),
        # On SMACv1 the return is graded (damage, kills, +win bonus; 20 = a win) and
        # separates arms where every win rate is 0. Not on SMACv2: its reward scale is
        # frozen at the first episode's teams (pitfall B2).
        'return_se': float(np.std(returns) / np.sqrt(len(returns))),
        # Discounted return from the first state: the units of the critic and of
        # onpolicy/pred_gap_to_teacher. Compare arms on this when checking that metric.
        'mean_discounted_return': float(np.mean(disc_returns)),
        'mean_length': float(np.mean(lengths)),
        'raw_returns': returns,
        'raw_wins': wins,
    }
    if diagnostics and steps:
        out['onpolicy'] = scalar_onpolicy_metrics(steps) if use_act else onpolicy_metrics(steps)
    if return_steps:
        out['steps'] = steps
    return out


def _step_record(out, arm):
    """What onpolicy_metrics needs from one MAVFlowAgent.eval_step output (B = 1)."""
    cand = np.asarray(out['cand'][:, 0])                 # (N, K) candidate joint actions
    V, V_ode = np.asarray(out['V'][:, 0]), np.asarray(out['V_ode'][:, 0])
    best = int(np.argmax(V))
    if arm == 'deployed':
        v_exec, v_exec_ode = float(out['V_dep'][0]), float(out['V_dep_ode'][0])
    else:
        n = 0 if arm == 'bc' else best
        v_exec, v_exec_ode = float(V[n]), float(V_ode[n])
    return dict(cand=cand, V=V, V_ode=V_ode,
                t1=cand[int(np.argmax(out['V1'][:, 0]))], t2=cand[int(np.argmax(out['V2'][:, 0]))],
                teacher=cand[best], deployed=np.asarray(out['deployed'][0]),
                v_dep=float(out['V_dep'][0]), v_exec=v_exec, v_exec_ode=v_exec_ode)


def scalar_onpolicy_metrics(steps):
    """On-policy diagnostics for the scalar-Q baselines (agents/qmix_cql.py):

    value_bias          Q(s, a) of the action played minus the discounted return the
                        episode then collected; > 0 = over-optimistic. Comparable with
                        MAV-Flow's onpolicy/value_bias (same units).
    value_bias_t0       the same at each episode's first state.
    value_corr          correlation of Q with the realised return over visited states.
    realised_return     mean discounted return-to-go over visited states.
    ca_exact_rate       critic='joint': share of checked states where coordinate ascent
                        found the exhaustive argmax's value (within 1e-5).
    ca_gap              mean exhaustive max minus coordinate ascent's value (>= 0).
    """
    G = np.array([s['G'] for s in steps])
    q = np.array([s['v_exec_ode'] for s in steps])
    t0 = np.array([s.get('t0', False) for s in steps])
    m = dict(
        value_bias=float((q - G).mean()),
        value_bias_t0=float((q - G)[t0].mean()) if t0.any() else float('nan'),
        value_corr=float(np.corrcoef(q, G)[0, 1]) if G.std() > 0 and q.std() > 0 else float('nan'),
        realised_return=float(G.mean()),
    )
    gaps = np.array([s['exact_gap'] for s in steps if 'exact_gap' in s])
    if len(gaps):
        m.update(ca_exact_rate=float((gaps <= 1e-5).mean()), ca_gap=float(gaps.mean()))
    return m


def _chance_corrected(hit, null):
    """(mean hit - mean null) / (1 - mean null): 0 = no better than a uniformly random
    candidate (`null` is that chance per state), 1 = always agrees."""
    h, n = float(np.mean(hit)), float(np.mean(null))
    return (h - n) / max(1.0 - n, 1e-6)


def onpolicy_metrics(steps):
    """Diagnostics on the states an eval arm actually visited. Plain meanings:

    frac_distinct       share of visited states where the N BC candidates are not all the
                        same joint action. Only there can any choice matter; the next five
                        are over those states.
    teacher_gain        how much more return the critic expects from the best candidate than
                        from an average one (its claimed value of choosing).
    in_pool             share of states where the deployed pick is one of the N candidates
                        (otherwise the per-agent argmax stitched a new joint action).
    twin_top1_agree     how often the two critics name the same best candidate, chance-
                        corrected: 0 = no more often than a random candidate would match
                        twin 1's pick, 1 = always.
    readout_top1_agree  how often the one-step readout (what picks a') and the full-ODE mean
                        (what y is drawn from) name the same best candidate.
    readout_gap         one-step minus full-ODE value, averaged over candidates and states.
    value_bias          critic value (full-ODE mean) of the executed action minus the
                        discounted return the episode actually collected from there;
                        > 0 = over-optimistic. The one-step readout's bias is this plus
                        readout_gap.
    value_bias_t0       the same at each episode's first state only: the critic's forecast
                        of the episode against what the policy then got.
    realised_return     mean discounted return-to-go over visited states (scale for the
                        biases).
    value_corr          correlation of the full-ODE value with the realised return across
                        visited states: does the critic know which states are good?
    regret              critic value of its own best candidate minus that of the action
                        taken, per visited state (one-step readout, return units).
    pred_gap_to_teacher the critic's prediction of (teacher arm's discounted return - this
                        arm's): regret summed along each episode with discount, averaged
                        over episodes. The critic is the Q-function of the teacher policy
                        (a' = its argmax over BC candidates), so by the performance-
                        difference lemma this equals the true gap when the critic is right
                        about within-state differences. Compare it with the measured gap,
                        evaluation/teacher/discounted_return - this arm's.
    Removed after the 10/19 probe: top1_deployed (chance-corrected joint agreement with the
    critic's pick; uncorrelated with the in-game extraction gap, r = 0.0 over 22 runs) and
    capture_deployed (the same information as regret, as a ratio).
    """
    d = np.array([(s['cand'] != s['cand'][:1]).any() for s in steps])
    D = [s for s, k in zip(steps, d) if k]
    agree = lambda x, y: bool(np.array_equal(x, y))
    m = dict(frac_distinct=float(d.mean()))
    if D:
        m.update(
            teacher_gain=float(np.mean([s['V'].max() - s['V'].mean() for s in D])),
            in_pool=float(np.mean([any(agree(c, s['deployed']) for c in s['cand']) for s in D])),
            twin_top1_agree=_chance_corrected(
                [agree(s['t2'], s['t1']) for s in D],
                [np.mean([agree(c, s['t1']) for c in s['cand']]) for s in D]),
            readout_top1_agree=float(np.mean([agree(s['cand'][int(np.argmax(s['V_ode']))], s['teacher'])
                                              for s in D])),
        )
    G = np.array([s['G'] for s in steps])
    vo = np.array([s['v_exec_ode'] for s in steps])
    t0 = np.array([s.get('t0', False) for s in steps])
    m.update(
        readout_gap=float(np.mean([(s['V'] - s['V_ode']).mean() for s in steps])),
        value_bias=float((vo - G).mean()),
        value_bias_t0=float((vo - G)[t0].mean()) if t0.any() else float('nan'),
        value_corr=float(np.corrcoef(vo, G)[0, 1]) if G.std() > 0 and vo.std() > 0 else float('nan'),
        realised_return=float(G.mean()),
    )
    if all('ep' in s for s in steps):
        regret = np.array([s['V'].max() - s['v_exec'] for s in steps])
        ep = np.array([s['ep'] for s in steps])
        gamma = steps[0].get('gamma', 0.99)
        per_ep = [float(np.sum(regret[ep == e] * gamma ** np.arange((ep == e).sum())))
                  for e in np.unique(ep)]
        m.update(regret=float(regret.mean()), pred_gap_to_teacher=float(np.mean(per_ep)))
    return m
