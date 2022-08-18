from unittest.mock import patch, Mock

import pytest

from Agent_based.base.agent import Agent
from Agent_based.base.environment import BaseEnvironment


@pytest.fixture()
@patch.multiple(BaseEnvironment, __abstractmethods__=set())
def env():
    return BaseEnvironment()


def test_adds_agents_to_environment(env):
    agents = ["Agent1", "Agent2"]
    env.add_agents(agents)

    for agent in agents:
        assert agent in env.agents


def test_adds_agent_to_environment(env):
    agent = "Agent"

    env.add_agent(agent)

    assert agent in env.agents


def test_execute_action_updates_performance(env):
    agent = Agent()
    env.add_agent(agent)

    prior_performance = agent.performance
    reward = 42
    env.reward = Mock(return_value=reward)

    env.execute_action(agent, "action")

    assert agent.performance == prior_performance + reward
