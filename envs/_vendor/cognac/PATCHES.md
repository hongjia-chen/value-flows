# Vendored COGNAC

Source: https://github.com/yojul/cognac, commit `dd357cc2e7c46f707cca2e766810b19de3017f73`
(2026-01-26). License: Apache-2.0, see `LICENSE.txt`.

## Why vendored instead of `pip install cognac`

Upstream requires Python >= 3.12 and numpy >= 2.1. The `value-flows` conda env is
Python 3.10 / numpy 1.26.4, and upgrading numpy underneath jax, og_marl and SMAC is
not a risk worth taking for four small environment files. Vendoring also pins the
dynamics, so a dataset generated today is replayable against the same code later.

## What was copied

`core/BaseReward.py`, `utils/graph_utils.py`, and `env/{FirefightingGraph,SysAdmin,
BinaryConsensus}`. `MultiCommodityFlow` was NOT copied: its actions are continuous
`Box` spaces whose dimension varies per agent, which the discrete MAV agent cannot use.

## Patches (every patched line is tagged `# VENDOR PATCH`)

None of these change dynamics, observations or rewards.

1. `from pettingzoo import ParallelEnv` -> `from ..._compat import ParallelEnv`.
   pettingzoo is only the base class; `_compat.py` falls back to `object` when
   pettingzoo is not installed.
2. `import networkx as nx` -> `from ..._compat import nx`. networkx is only used for
   rendering. `BinaryConsensusNetworkEnvironment.__init__` builds its rendering graph
   only when networkx is importable. `graph_utils.py` gains
   `from __future__ import annotations` so its `-> nx.DiGraph` annotation does not
   evaluate at import time.
3. `np.clip(x, min=a, max=b)` -> `np.clip(x, a, b)` (two call sites). The keyword
   names only exist in numpy >= 2.1; the positional form is identical.
4. `env/__init__.py` rewritten to drop the MultiCommodityFlow import;
   `utils/__init__.py` left empty (upstream's imports `make_env`, which imports MCF).

To re-verify: clone upstream at the commit above and `diff -r` against this
directory; only the lines tagged `# VENDOR PATCH` (and the two `__init__.py` files)
should differ.

## Upstream behaviours that are NOT patched here

These are handled in `envs/cognac_utils.py` (the adapter), not by editing upstream:

- Initial states and BinaryConsensus dynamics draw from the GLOBAL `np.random`, and
  BinaryConsensus ignores `reset(seed=...)` entirely.
- Firefighting `get_obs()` at reset reads `last_visited_house` left over from the
  previous episode.
- SysAdmin and BinaryConsensus set `terminations=True` at the time limit.
- `is_shared_reward` is ignored by the Firefighting reward.
- Firefighting truncates when `timestep > max_steps`, i.e. after `max_steps + 2`
  steps.
