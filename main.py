import os
 
import json
import random
import time
 
import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
 
from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.marl_utils import batch_concat_agent_id_to_obs  # churn probe

FLAGS = flags.FLAGS

flags.DEFINE_integer('enable_wandb', 1, 'Whether to use wandb.')
flags.DEFINE_string('wandb_run_group', 'ValueFlows', 'Run group.')
flags.DEFINE_string('wandb_mode', 'offline', 'Wandb mode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 100, 'Logging interval.') ## Changed log_interval from 5000 to 100 for test runs.
# 0903: eval used to be 61% of a 500k 3s5z run's wall clock (16.3 h of 26.8 h),
# because sample_actions_rejection was not jitted. That is fixed in
# agents/mav_flows.py, and these two defaults cut the remaining eval work ~17x:
# 500k steps now means 6 evals x 10 episodes instead of 21 x 50.
# CAVEAT for reported numbers: 3s5z win_rate ran 0.02-0.16 in the 0827 run, i.e.
# 1-8 wins out of 50. At 10 episodes win_rate can only read 0.0 or 0.1. Pass
# --eval_episodes=50 --eval_interval=25000 for any run whose curves get published;
# post-jit that costs ~10 min, not ~16 h.
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

# WAS (pre-0903): flags.DEFINE_integer('eval_episodes', 50, ...)
flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')

# Churn probe: number of fixed probe states used to measure greedy-policy churn
# across checkpoints. Set to 0 to disable.
flags.DEFINE_integer('churn_probe_size', 256, 'Fixed probe-batch size for churn metric (0 disables).')

# Which recorder produced the SMAC vault being loaded — this selects the eval env
# and therefore the observation spec, so it must match the vault:
#   'og_marl' : standard og-marl SMACv1 recording (5m_vs_6m, 3s5z_vs_3s6z, 3m, ...)
#   'omiga'   : the OMIGA paper's SMACv1 variant (corridor). Its env appends an
#               agent-id one-hot to each observation and uses per-agent states, so
#               corridor's obs is 346 rather than the 156 the standard env emits.
# main() verifies the choice against the dataset and fails fast on a mismatch.
flags.DEFINE_string('smac_env_source', 'og_marl', "SMAC eval env recorder: 'og_marl' or 'omiga'.")

config_flags.DEFINE_config_file('agent', 'agents/value_flows.py', lock_config=False)


def _parse_smac_env_name(env_name):
    parts = env_name.split('_')
    if parts[0] == 'smac':
        return ('smac_v1', '_'.join(parts[1:-1]))
    elif parts[0] == 'smacv2':
        return ('smac_v2', '_'.join(parts[1:-1]))
    return None


def _parse_mpe_env_name(env_name):
    if not env_name.startswith('mpe_'):
        return None
    from envs.mpe_utils import parse_env_name

    scenario, _ = parse_env_name(env_name)
    return scenario

def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb_run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    if FLAGS.enable_wandb:
        setup_wandb(
            wandb_output_dir=FLAGS.save_dir,
            project='value-flows', group=FLAGS.wandb_run_group, name=exp_name,
            mode=FLAGS.wandb_mode
        )
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets. (Loading train/test datasets using appropriate benchmarks)
    config = FLAGS.agent
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
    if FLAGS.online_steps > 0:
        assert 'visual' not in FLAGS.env_name, 'Online fine-tuning is currently not supported for visual environments.'

    smac_info = _parse_smac_env_name(FLAGS.env_name)
    is_smac = smac_info is not None
    mpe_scenario = _parse_mpe_env_name(FLAGS.env_name)
    is_mpe = mpe_scenario is not None
    if is_mpe and FLAGS.online_steps > 0:
        raise ValueError('MPE currently supports offline training and evaluation only; set --online_steps=0.')
    if is_smac:
        from og_marl.environments import get_environment
        map_source, scenario = smac_info
        print(f'Building SMAC eval env: {map_source} / {scenario} '
              f'(dataset_source={FLAGS.smac_env_source})')
        eval_env = get_environment(FLAGS.smac_env_source, map_source, scenario, seed=FLAGS.seed)

        # The vaults under smac_v1/ were not all recorded by the same pipeline, and
        # the recorder determines the observation spec. Picking the wrong source
        # builds an env whose observations do not match the dataset the agent was
        # sized from, and the run dies at the FIRST eval (not at startup) inside
        # sample_actions_rejection with a ScopeParamShapeError -- i.e. hours in.
        # Fail here instead, with the fix spelled out.
        #
        # NEVER reset() the eval env here. SMAC finishes building max_reward inside
        # init_units() (called from reset) under `if self._episode_count == 0`, and
        # _episode_count only advances when an episode TERMINATES -- so a probe
        # reset that never steps lets the first eval episode's reset add every
        # enemy's health_max+shield_max a SECOND time. That inflates the reward
        # normalizer max_reward/reward_scale_rate for the whole run and silently
        # compresses every eval return (0831: 5m_vs_6m capped at 13.25 instead of
        # 20.00, because max_reward was 800 rather than 530). get_obs_size() is
        # pure arithmetic over map params set in __init__, so it needs no reset and
        # does not boot SC2. The OMIGA env returns a list whose [0] is the total
        # (it folds in the agent-id one-hot); vanilla SMAC returns an int.
        ds_obs_dim = int(np.asarray(train_dataset['observations']).shape[-1])
        obs_size = eval_env._environment.get_obs_size()
        env_obs_dim = int(obs_size[0] if isinstance(obs_size, (list, tuple)) else obs_size)
        if ds_obs_dim != env_obs_dim:
            raise ValueError(
                f'SMAC observation mismatch for {scenario}: the vault provides '
                f'{ds_obs_dim}-dim observations but the '
                f'"{FLAGS.smac_env_source}" env provides {env_obs_dim}. The eval '
                f'env was built by the wrong recorder. Re-run with '
                f'--smac_env_source=omiga (or =og_marl) to match the vault. '
                f'Known: 5m_vs_6m/3s5z_vs_3s6z are og_marl, corridor is omiga.')
        print(f'[OK] SMAC obs parity: dataset={ds_obs_dim} env={env_obs_dim}')
    elif is_mpe:
        from envs.mpe_omar import MPEOMAR

        print(f'Building OMAR MPE eval env: simple_spread / seed {FLAGS.seed}')
        eval_env = MPEOMAR(mpe_scenario, seed=FLAGS.seed)
    
    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets. 
    train_dataset = Dataset.create(**train_dataset)
    replay_buffer = None
    if FLAGS.online_steps > 0:
        if FLAGS.balanced_sampling:
            # RLPD-style online buffer: sample half from the static dataset and half online.
            example_transition = {k: v[0] for k, v in train_dataset.items()}
            replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
        else:
            # Initialize an online replay buffer from the offline dataset.
            replay_buffer = ReplayBuffer.create_from_initial_dataset(
                dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
            )
    # Set p_aug and frame_stack.
    # Short frame_stack give the network an approximate Markov state
    # p_aug is padding the image by a few pixels and crop back the original size
        # Regularization for smalle camera or pose translations
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] in ['rebrac']:
                dataset.return_next_actions = True

    # Create agent.
    example_batch = train_dataset.sample(1)

    assert 'rewards' in train_dataset
    example_batch['min_reward'] = float(train_dataset['rewards'].min())
    example_batch['max_reward'] = float(train_dataset['rewards'].max())
    assert example_batch['min_reward'] <= example_batch['max_reward']

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # --- Churn probe setup (F1 diagnostic) -------------------------------------
    # A fixed set of probe states + a fixed RNG key, frozen for the whole run, so
    # that any change in the greedy action between checkpoints is attributable to
    # PARAMETER drift, not to different states or different sampling noise.
    #   churn_vs_prev   : instantaneous instability (flips since last checkpoint)
    #   churn_vs_anchor : cumulative drift from the first eval checkpoint
    # Healthy learning -> churn_vs_prev decays to a low floor; the F1 gauge-drift
    # pathology -> it plateaus high and churn_vs_anchor rises monotonically.
    # Only meaningful for SMAC (uses sample_actions_rejection); gated accordingly.
    churn = None
    if is_smac and FLAGS.churn_probe_size > 0:
        probe_batch = train_dataset.sample(FLAGS.churn_probe_size)
        churn = dict(
            obs_id=batch_concat_agent_id_to_obs(probe_batch['observations']),
            legals=probe_batch['legals'],
            key=jax.random.PRNGKey(12345),   # FIXED key: flips are pure param drift
            prev=None,
            anchor=None,
        )

    # Train agent.

    ## Two main loops: offline training (i <= FLAGS.offline_steps) 
    #       and online fine-tuning (i > FLAGS.offline_steps).

        # Jitted validation forward pass. total_loss now runs ~112 ten-step ODE
    # integrations; eager dispatch would dominate the training loop.
    val_loss_fn = None
    if val_dataset is not None:
        val_loss_fn = jax.jit(lambda ag, b, k: ag.total_loss(b, ag.network.params, k))

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    
    rng = jax.random.PRNGKey(FLAGS.seed)
    expl_metrics = dict()
    done = True
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            # Offline RL. This is the gradient step (computes its respective loss)
            batch = train_dataset.sample(config['batch_size'])
            if config['agent_name'] in ['rebrac']:
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)
        else:
            # Online fine-tuning.
            rng, expl_rng = jax.random.split(rng)
            
            if done:
                obs, _ = env.reset()
            
            if config['agent_name'] in ['value_flows']:
                action = agent.sample_actions(observations=obs, temperature=1, seed=expl_rng, policy_extraction='rpg')
            else:
                action = agent.sample_actions(observations=obs, temperature=1, seed=expl_rng)
            action = np.array(action)
            
            next_obs, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                # Adjust reward for D4RL antmaze.
                reward = reward - 1.0
            
            replay_buffer.add_transition(
                dict(
                    observations=obs,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_obs,
                )
            )
            obs = next_obs
            
            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            if FLAGS.balanced_sampling:
                # Half-and-half sampling from the training dataset and the replay buffer.
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
            else:
                batch = replay_buffer.sample(config['batch_size'])

            if config['agent_name'] in ['rebrac']:
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = val_loss_fn(agent, val_batch, jax.random.PRNGKey(i))
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            if FLAGS.enable_wandb:
                wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            eval_metrics = {}

            # --- Churn metric (F1 diagnostic) ---
            # Runs the greedy policy on a FIXED probe set with a FIXED key, then
            # diffs the joint actions against the previous checkpoint and the
            # first (anchor) checkpoint. Pure reads; does not touch training state.
            if churn is not None:
                cur_probe = agent.sample_actions_rejection(
                    churn['obs_id'], churn['legals'], churn['key'])   # (probe_size, K)
                if churn['prev'] is not None:
                    eval_metrics['churn/greedy_flip_rate_vs_prev'] = float(
                        (cur_probe != churn['prev']).mean())
                if churn['anchor'] is None:
                    churn['anchor'] = cur_probe   # freeze first eval as the anchor
                else:
                    eval_metrics['churn/greedy_flip_rate_vs_anchor'] = float(
                        (cur_probe != churn['anchor']).mean())
                churn['prev'] = cur_probe

            if is_smac:
                # OG-MARL eval via evaluate_smac (matches MAC-Flow's _evaluate)
                from utils.evaluate_smac import evaluate_smac
                results = evaluate_smac(
                    agent=agent, env=eval_env,
                    num_episodes=FLAGS.eval_episodes, seed=i, verbose=False,
                )
                eval_info = {
                    'mean_episode_return': results['mean_return'],
                    'std_episode_return': results['std_return'],
                    'max_episode_return': results['max_return'],
                    'min_episode_return': results['min_return'],
                    'win_rate': results['win_rate'],
                    'mean_episode_length': results['mean_length'],
                }
                renders = []
            elif is_mpe:
                from utils.evaluate_mpe import evaluate_mpe

                results = evaluate_mpe(
                    agent=agent,
                    env=eval_env,
                    num_episodes=FLAGS.eval_episodes,
                    seed=i,
                )
                eval_info = {
                    'mean_episode_return': results['mean_return'],
                    'std_episode_return': results['std_return'],
                    'max_episode_return': results['max_return'],
                    'min_episode_return': results['min_return'],
                    'mean_episode_length': results['mean_length'],
                }
                renders = []
            else:
                if i > FLAGS.offline_steps and config['agent_name'] in ['value_flows']:
                    eval_kwargs = dict(policy_extraction='rpg')
                else:
                    eval_kwargs = dict()
                eval_info, _, renders = evaluate(
                    agent=agent, env=eval_env,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    **eval_kwargs,
                )
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v
            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video
            if FLAGS.enable_wandb:
                wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
