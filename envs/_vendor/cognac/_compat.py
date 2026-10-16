"""VENDOR PATCH: optional-dependency shims (see PATCHES.md).

pettingzoo is used only for the ParallelEnv base class, and networkx only for
rendering, so neither is required to step the environments.
"""

try:
    from pettingzoo import ParallelEnv  # noqa: F401
except ImportError:  # pragma: no cover - depends on the local environment
    class ParallelEnv:  # noqa: D101
        pass

try:
    import networkx as nx  # noqa: F401
except ImportError:  # pragma: no cover - depends on the local environment
    nx = None
