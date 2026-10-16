"""Reproducible OMAR Simple Spread environment for evaluation."""

from typing import Any

import numpy as np
from og_marl.custom_environments.multiagent_particle_envs.multiagent.environment import (
    MultiAgentEnv,
)
from og_marl.custom_environments.multiagent_particle_envs.multiagent.simple_spread import (
    Scenario,
)


class MPEOMAR:
    """Three-agent continuous Simple Spread with the dataset's 25-step horizon."""

    def __init__(self, scenario: str, seed: int | None = None):
        if scenario != 'simple_spread':
            raise ValueError(f'Only simple_spread is supported, got {scenario!r}.')
        self._rng = np.random.RandomState(1 if seed is None else seed)
        global_state = np.random.get_state()
        np.random.set_state(self._rng.get_state())
        try:
            scenario_builder = Scenario()
            world = scenario_builder.make_world()
            self.environment = MultiAgentEnv(
                world,
                scenario_builder.reset_world,
                scenario_builder.reward,
                scenario_builder.observation,
            )
            self._rng.set_state(np.random.get_state())
        finally:
            np.random.set_state(global_state)
        self.agents = [f'agent_{index}' for index in range(3)]
        self.num_actions = 2
        self.num_agents = len(self.agents)
        self.max_episode_length = 25
        self.t = 0

    def _with_environment_rng(self, function, *args):
        """Run legacy MPE code without consuming NumPy's training RNG stream."""
        global_state = np.random.get_state()
        np.random.set_state(self._rng.get_state())
        try:
            result = function(*args)
            self._rng.set_state(np.random.get_state())
            return result
        finally:
            np.random.set_state(global_state)

    def reset(self):
        observations = self._with_environment_rng(self.environment.reset)
        self.t = 0
        return {
            agent: observations[index].astype(np.float32)
            for index, agent in enumerate(self.agents)
        }, {}

    def step(self, actions):
        action_list = [
            np.clip(np.asarray(actions[agent], dtype=np.float32), -1.0, 1.0)
            for agent in self.agents
        ]
        observations, rewards, dones, info = self._with_environment_rng(
            self.environment.step, action_list
        )
        self.t += 1
        at_time_limit = self.t >= self.max_episode_length

        return (
            {
                agent: observations[index].astype(np.float32)
                for index, agent in enumerate(self.agents)
            },
            {agent: float(rewards[index]) for index, agent in enumerate(self.agents)},
            {agent: bool(dones[index]) for index, agent in enumerate(self.agents)},
            {agent: at_time_limit for agent in self.agents},
            info,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.environment, name)
