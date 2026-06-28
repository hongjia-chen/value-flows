

# in a notebook cell with the value-flows env active

#%%
from agents.mav_flows import MAVFlowAgent, get_config
from envs.env_utils import make_env_and_datasets
import jax
import jax.numpy as jnp

# --- Setup ---
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('smac_3m_good')

example_batch = train_dataset.sample(1)
example_batch['min_reward'] = float(train_dataset['rewards'].min())
example_batch['max_reward'] = float(train_dataset['rewards'].max())

config = get_config()
agent = MAVFlowAgent.create(seed=0, example_batch=example_batch, config=config)

print("Agent created successfully\n")

# --- Sanity check 1: param tree structure ---
print("=" * 60)
print("Param tree shapes:")
print("=" * 60)
print(jax.tree.map(lambda x: x.shape, agent.network.params))

# --- Sanity check 2: expected top-level keys ---
print("\n" + "=" * 60)
print("Top-level module key check")
print("=" * 60)

expected_keys = {
    'modules_joint_critic_flow1',
    'modules_joint_critic_flow2',
    'modules_target_joint_critic_flow1',
    'modules_target_joint_critic_flow2',
    'modules_q_heads',
    'modules_actor_bc_flow',
    'modules_actor_onestep_flow',
}
actual_keys = set(agent.network.params.keys())
missing = expected_keys - actual_keys
extra = actual_keys - expected_keys
assert actual_keys == expected_keys, (
    f"Param tree key mismatch:\n  missing: {missing}\n  extra:   {extra}"
)
print(f"All {len(expected_keys)} expected module keys present, no extras.")

# --- Sanity check 3: target params are exact copies of main critic params ---
print("\n" + "=" * 60)
print("Target / main param parity check")
print("=" * 60)

def trees_equal(a, b):
    leaves_a, _ = jax.tree_util.tree_flatten(a)
    leaves_b, _ = jax.tree_util.tree_flatten(b)
    return all(jnp.array_equal(x, y) for x, y in zip(leaves_a, leaves_b))

assert trees_equal(
    agent.network.params['modules_joint_critic_flow1'],
    agent.network.params['modules_target_joint_critic_flow1'],
), "target_joint_critic_flow1 not a copy of joint_critic_flow1"
assert trees_equal(
    agent.network.params['modules_joint_critic_flow2'],
    agent.network.params['modules_target_joint_critic_flow2'],
), "target_joint_critic_flow2 not a copy of joint_critic_flow2"
print("Both target critics correctly initialized as copies of their main critics.")

# --- Sanity check 4: input-dim shape assertions ---
print("\n" + "=" * 60)
print("First-layer input-dim check")
print("=" * 60)

K = agent.config['num_agents']
A = agent.config['action_dim']
obs_dim = agent.config['ob_dims']
state_dim = agent.config['state_dim']

joint_in_expected = state_dim + 1 + 1 + K * A   # state + returns(1) + times(1) + joint_action(K*A)
qhead_in_expected = obs_dim + K + A             # local_obs + agent_id_onehot + per_agent_action_onehot

# jax.tree_util.tree_map_with_path(
#     lambda p, x: print(jax.tree_util.keystr(p), x.shape),
#     agent.network.params['modules_joint_critic_flow1']
# )


joint_kernel = agent.network.params['modules_joint_critic_flow1']['value_net']['Dense_0']['kernel']
qhead_kernel = agent.network.params['modules_q_heads']['value_net']['Dense_0']['kernel']

assert joint_kernel.shape[0] == joint_in_expected, (
    f"Joint critic first-layer input dim wrong: got {joint_kernel.shape[0]}, expected {joint_in_expected}"
)
assert qhead_kernel.shape[0] == qhead_in_expected, (
    f"Q-head first-layer input dim wrong: got {qhead_kernel.shape[0]}, expected {qhead_in_expected}"
)
print(f"Joint critic input dim: {joint_kernel.shape[0]} == {joint_in_expected}  OK")
print(f"Q-head input dim:       {qhead_kernel.shape[0]} == {qhead_in_expected}  OK")

# --- Sanity check 5: config values populated from data ---
print("\n" + "=" * 60)
print("Config inference check")
print("=" * 60)

print(f"num_agents = {K}        (expected 3 for 3m)")
print(f"ob_dims    = {obs_dim}  (expected 30 for 3m)")
print(f"state_dim  = {state_dim}  (expected 48 for 3m)")
print(f"action_dim = {A}        (expected 9 for 3m)")
print(f"min_reward = {agent.config['min_reward']:.4f}")
print(f"max_reward = {agent.config['max_reward']:.4f}")

assert K == 3 and obs_dim == 30 and state_dim == 48 and A == 9, \
    "Shape inference disagrees with known 3m dimensions"
print("All inferred shapes match 3m spec.")

print("\n" + "=" * 60)
print("All sanity checks passed.")
print("=" * 60)

example_batch = train_dataset.sample(8)
example_batch['min_reward'] = float(train_dataset['rewards'].min())
example_batch['max_reward'] = float(train_dataset['rewards'].max())

rng = jax.random.PRNGKey(42)
loss, info = agent.q_head_factorization_loss(
    example_batch, agent.network.params, rng
)
print(f"Loss: {loss:.4f}")
print(f"Info: {info}")
assert jnp.isfinite(loss), "Non-finite loss"
assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"

# Verify gradient flows to q_heads but NOT to joint critic
grads = jax.grad(lambda p: agent.q_head_factorization_loss(example_batch, p, rng)[0])(
    agent.network.params
)
q_head_grad_norm = jnp.sqrt(sum(
    (g ** 2).sum() for g in jax.tree.leaves(grads['modules_q_heads'])
))
critic_grad_norm = jnp.sqrt(sum(
    (g ** 2).sum() for g in jax.tree.leaves(grads['modules_joint_critic_flow1'])
))
print(f"Q-head gradient norm:    {q_head_grad_norm:.6f}  (should be > 0)")
print(f"Joint critic grad norm:  {critic_grad_norm:.6f}  (should be 0)")
assert q_head_grad_norm > 0, "No gradient to Q-heads!"
assert critic_grad_norm == 0.0, "Gradient leaked to joint critic!"
print("All factorization loss checks passed.")

K = agent.config['num_agents']
expected = info['q_per_agent_mean'] * K
actual = info['q_tot_factor_mean']
assert abs(actual - expected) < 1e-4, (
    f"Factor sum direction wrong: q_tot_factor_mean={actual:.4f}, "
    f"expected q_per_agent_mean × K = {expected:.4f}"
)
print(f"Magnitude check: {actual:.4f} ≈ {expected:.4f}  OK")

example_batch = train_dataset.sample(8)
example_batch['min_reward'] = float(train_dataset['rewards'].min())
example_batch['max_reward'] = float(train_dataset['rewards'].max())

rng = jax.random.PRNGKey(123)
loss, info = agent.joint_critic_bcfm_loss(
    example_batch, agent.network.params, rng
)
print(f"BCFM loss: {loss:.4f}")
print(f"Info: {info}")
assert jnp.isfinite(loss), "Non-finite loss"
assert loss.shape == (), f"Expected scalar, got {loss.shape}"

# Gradient flow: main critics yes, target critics no, q_heads no, actor no
grads = jax.grad(lambda p: agent.joint_critic_bcfm_loss(example_batch, p, rng)[0])(
    agent.network.params
)

def norm(subtree):
    return jnp.sqrt(sum((g ** 2).sum() for g in jax.tree.leaves(subtree)))

main_c1_norm = norm(grads['modules_joint_critic_flow1'])
main_c2_norm = norm(grads['modules_joint_critic_flow2'])
tgt_c1_norm  = norm(grads['modules_target_joint_critic_flow1'])
tgt_c2_norm  = norm(grads['modules_target_joint_critic_flow2'])
qhead_norm   = norm(grads['modules_q_heads'])
actor_norm   = norm(grads['modules_actor_bc_flow'])

print(f"Main critic 1 grad:    {main_c1_norm:.4f}  (should be > 0)")
print(f"Main critic 2 grad:    {main_c2_norm:.4f}  (should be > 0)")
print(f"Target critic 1 grad:  {tgt_c1_norm:.4f}  (should be 0)")
print(f"Target critic 2 grad:  {tgt_c2_norm:.4f}  (should be 0)")
print(f"Q-heads grad:          {qhead_norm:.4f}  (should be 0)")
print(f"Actor BC flow grad:    {actor_norm:.4f}  (should be 0)")

assert main_c1_norm > 0 and main_c2_norm > 0, "Main critics not getting gradient"
assert tgt_c1_norm == 0.0 and tgt_c2_norm == 0.0, "Gradient leaked to target critics"
assert qhead_norm == 0.0, "Gradient leaked to Q-heads"
assert actor_norm == 0.0, "Gradient leaked to actor"
print("All BCFM loss checks passed.")

example_batch = train_dataset.sample(8)
example_batch['min_reward'] = float(train_dataset['rewards'].min())
example_batch['max_reward'] = float(train_dataset['rewards'].max())

rng = jax.random.PRNGKey(456)
loss, info = agent.joint_critic_dcfm_loss(
    example_batch, agent.network.params, rng
)
print(f"DCFM loss: {loss:.4f}")
print(f"Info: {info}")
assert jnp.isfinite(loss), "Non-finite loss"
assert loss.shape == (), f"Expected scalar, got {loss.shape}"

# Gradient flow: main critics yes, everything else no
grads = jax.grad(lambda p: agent.joint_critic_dcfm_loss(example_batch, p, rng)[0])(
    agent.network.params
)

def norm(subtree):
    return jnp.sqrt(sum((g ** 2).sum() for g in jax.tree.leaves(subtree)))

main_c1_norm = norm(grads['modules_joint_critic_flow1'])
main_c2_norm = norm(grads['modules_joint_critic_flow2'])
tgt_c1_norm  = norm(grads['modules_target_joint_critic_flow1'])
tgt_c2_norm  = norm(grads['modules_target_joint_critic_flow2'])
qhead_norm   = norm(grads['modules_q_heads'])
actor_norm   = norm(grads['modules_actor_bc_flow'])

print(f"Main critic 1 grad:    {main_c1_norm:.4f}  (should be > 0)")
print(f"Main critic 2 grad:    {main_c2_norm:.4f}  (should be > 0)")
print(f"Target critic 1 grad:  {tgt_c1_norm:.4f}  (should be 0)")
print(f"Target critic 2 grad:  {tgt_c2_norm:.4f}  (should be 0)")
print(f"Q-heads grad:          {qhead_norm:.4f}  (should be 0)")
print(f"Actor BC flow grad:    {actor_norm:.4f}  (should be 0)")

assert main_c1_norm > 0 and main_c2_norm > 0, "Main critics not getting gradient"
assert tgt_c1_norm == 0.0 and tgt_c2_norm == 0.0, "Gradient leaked to target critics"
assert qhead_norm == 0.0, "Gradient leaked to Q-heads"
assert actor_norm == 0.0, "Gradient leaked to actor"

# DCFM magnitude sanity: with clip_flow_returns, noisy_returns should be bounded
return_lo = agent.config['min_reward'] / (1 - agent.config['discount'])
return_hi = agent.config['max_reward'] / (1 - agent.config['discount'])
nnr_mean = info['noisy_next_returns_mean']
assert return_lo - 2.0 <= nnr_mean <= return_hi + 2.0, (
    f"noisy_next_returns_mean={nnr_mean:.4f} far outside [{return_lo:.2f}, {return_hi:.2f}]"
)
print(f"Noisy return bound check: {return_lo:.2f} ≤ {nnr_mean:.4f} ≤ {return_hi:.2f}  OK")

print("=" * 60)
print("Joint DCFM: PASS")
print("=" * 60)

example_batch = train_dataset.sample(8)
example_batch['min_reward'] = float(train_dataset['rewards'].min())
example_batch['max_reward'] = float(train_dataset['rewards'].max())

rng = jax.random.PRNGKey(789)
loss, info = agent.actor_loss(example_batch, agent.network.params, rng)
print(f"Actor loss: {loss:.4f}")
print(f"Info: {info}")
assert jnp.isfinite(loss), "Non-finite actor loss"
assert loss.shape == (), f"Expected scalar, got {loss.shape}"

# Gradient routing
grads = jax.grad(lambda p: agent.actor_loss(example_batch, p, rng)[0])(
    agent.network.params
)

def norm(subtree):
    return jnp.sqrt(sum((g ** 2).sum() for g in jax.tree.leaves(subtree)))

bc_norm      = norm(grads['modules_actor_bc_flow'])
onestep_norm = norm(grads['modules_actor_onestep_flow'])
qhead_norm   = norm(grads['modules_q_heads'])
critic1_norm = norm(grads['modules_joint_critic_flow1'])
critic2_norm = norm(grads['modules_joint_critic_flow2'])
tgt1_norm    = norm(grads['modules_target_joint_critic_flow1'])
tgt2_norm    = norm(grads['modules_target_joint_critic_flow2'])

print(f"actor_bc_flow grad:      {bc_norm:.4f}  (> 0 expected)")
print(f"actor_onestep_flow grad: {onestep_norm:.4f}  (> 0 expected)")
print(f"q_heads grad:            {qhead_norm:.4f}  (== 0 expected)")
print(f"joint_critic_flow1 grad: {critic1_norm:.4f}  (== 0 expected)")
print(f"joint_critic_flow2 grad: {critic2_norm:.4f}  (== 0 expected)")
print(f"target_joint_critic_flow1 grad: {tgt1_norm:.4f}  (== 0 expected)")
print(f"target_joint_critic_flow2 grad: {tgt2_norm:.4f}  (== 0 expected)")

assert bc_norm > 0, "BC flow not getting gradient"
assert onestep_norm > 0, "One-step actor not getting gradient (distill+Q-guide)"
assert qhead_norm == 0.0, "Gradient leaked to Q-heads"
assert critic1_norm == 0.0 and critic2_norm == 0.0, "Gradient leaked to critics"
assert tgt1_norm == 0.0 and tgt2_norm == 0.0, "Gradient leaked to target critics"

# Magnitude check: distill target must come from the multi-step actor, not random
# (if multi-step gradient is zero AND distill loss > 0, target is being produced — good.)
assert info['distill_loss'] > 0, "Distill loss is zero — argmax target wiring may be broken"

# Q-guidance sanity: q_tot should be finite even with random init
assert jnp.isfinite(info['q_tot_mean']), "Q-guidance produced non-finite Q values"

print("=" * 60)
print("Actor loss: PASS")
print("=" * 60)

example_batch = train_dataset.sample(256)   # realistic batch size now
example_batch['min_reward'] = float(train_dataset['rewards'].min())
example_batch['max_reward'] = float(train_dataset['rewards'].max())

# Snapshot pre-update params for change-detection
import copy
pre_params = jax.tree.map(lambda x: x.copy(), agent.network.params)

# Single update
import time
t0 = time.time()
agent, info = agent.update(example_batch)
print(f"First update (compile + run): {time.time() - t0:.2f}s")
print(f"  total_loss: {info['total_loss']:.4f}")
print(f"  critic_loss: {info['critic_loss']:.4f}")
print(f"  q_head_loss: {info['q_head/q_head_loss']:.4f}")
print(f"  actor_loss: {info['actor/actor_loss']:.4f}")

# Run a few more steps to confirm jit cached + values evolve
losses = []
t0 = time.time()
for step in range(10):
    batch = train_dataset.sample(256)
    batch['min_reward'] = float(train_dataset['rewards'].min())
    batch['max_reward'] = float(train_dataset['rewards'].max())
    agent, info = agent.update(batch)
    losses.append(float(info['total_loss']))
print(f"10 cached steps: {time.time() - t0:.2f}s")
print(f"Total loss trajectory: {[f'{l:.2f}' for l in losses]}")

post_params = agent.network.params

# --- Sanity 1: trainable networks changed ---
def params_diff(pre, post):
    return jnp.sqrt(sum(
        ((b - a) ** 2).sum() for a, b in zip(jax.tree.leaves(pre), jax.tree.leaves(post))
    ))

for name in ['joint_critic_flow1', 'joint_critic_flow2', 'q_heads',
             'actor_bc_flow', 'actor_onestep_flow']:
    d = params_diff(pre_params[f'modules_{name}'], post_params[f'modules_{name}'])
    print(f"{name} param Δ: {d:.6f}  (should be > 0)")
    assert d > 0, f"{name} parameters didn't change after update"

# --- Sanity 2: target critics moved slightly (EMA), not synchronously ---
for name in ['joint_critic_flow1', 'joint_critic_flow2']:
    main_d = params_diff(pre_params[f'modules_{name}'],
                         post_params[f'modules_{name}'])
    tgt_d = params_diff(pre_params[f'modules_target_{name}'],
                        post_params[f'modules_target_{name}'])
    print(f"target_{name}: main Δ={main_d:.4f}, target Δ={tgt_d:.6f}, "
          f"ratio={tgt_d/main_d:.4f} (should be small, ~tau)")
    # Target should lag main by roughly tau per step, summed over 11 steps
    # Rough check: ratio should be much less than 1
    assert tgt_d < main_d, f"target_{name} moved as much as main critic"

# --- Sanity 3: losses are finite throughout ---
assert all(jnp.isfinite(l) for l in losses), "Loss went non-finite during training"

# --- Sanity 4: total loss generally trending down ---
# Not a hard requirement at 10 steps, but worth printing
print(f"Loss[0]: {losses[0]:.2f} → Loss[-1]: {losses[-1]:.2f}")

print("=" * 60)
print("Training step: PASS")
print("=" * 60)

# %%
# %%
"""
Agreement check: compare trained MAV-Flow agent's actions against the dataset's
behavior policy actions.

Before running:
  1. ls exp/mav_flow_3m_good_500k/   → find your experiment directory name
  2. Replace <your_exp_name> below with that name
  3. Confirm the largest available step (probably 500000)
"""

from agents.mav_flows import MAVFlowAgent, get_config
from envs.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent
from utils.marl_utils import batch_concat_agent_id_to_obs
from utils.datasets import Dataset
import jax
import jax.numpy as jnp
import numpy as np

# ---- EDIT THESE TWO LINES ----
CHECKPOINT_DIR = 'exp/mav_flow_3m_good_500k/sd000_20260627_161911'
CHECKPOINT_STEP = 500000
# ------------------------------

# --- Load dataset ---
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('smac_3m_good')
train_dataset = Dataset.create(**train_dataset)

# --- Build agent shell and restore trained params ---
example_batch = train_dataset.sample(1)
example_batch['min_reward'] = float(train_dataset['rewards'].min())
example_batch['max_reward'] = float(train_dataset['rewards'].max())
config = get_config()
agent = MAVFlowAgent.create(seed=0, example_batch=example_batch, config=config)
agent = restore_agent(agent, CHECKPOINT_DIR, CHECKPOINT_STEP)
print(f"Restored agent from {CHECKPOINT_DIR} at step {CHECKPOINT_STEP}\n")

# --- Sample a large batch and compare actions ---
N = 10000
batch = train_dataset.sample(N)
obs_with_id = batch_concat_agent_id_to_obs(batch['observations'])
rng = jax.random.PRNGKey(0)
predicted = agent.sample_actions(obs_with_id, batch['legals'], rng)
behavior = batch['actions']

predicted = np.asarray(predicted)
behavior = np.asarray(behavior)

# Overall
agreement = (predicted == behavior).mean()
print(f"Overall agreement: {agreement:.1%}  ({N} transitions × {predicted.shape[1]} agents)\n")

# Per-agent
print("Per-agent agreement:")
K = predicted.shape[1]
for i in range(K):
    a = (predicted[:, i] == behavior[:, i]).mean()
    print(f"  Agent {i}: {a:.1%}")

# Per-action distribution
print("\nAction distribution comparison:")
print(f"  {'Action':<8}{'Predicted':>12}{'Behavior':>12}{'Diff':>10}")
A = 9
for a in range(A):
    pred_freq = (predicted == a).mean()
    beh_freq = (behavior == a).mean()
    diff = pred_freq - beh_freq
    print(f"  {a:<8}{pred_freq:>11.1%}{beh_freq:>12.1%}{diff:>+10.1%}")
# %%
