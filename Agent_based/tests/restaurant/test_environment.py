import pytest

from base.agent import Agent
from restaurant.environment import RestaurantEnvironment, POSSIBLE_NUMBER_OF_GUESTS


@pytest.fixture()
def restaurant_env():
    return RestaurantEnvironment()


def test_is_fully_observable(restaurant_env):
    agent = Agent()
    assert restaurant_env.percept(agent) == restaurant_env.state


def test_state_change_is_valid(restaurant_env):
    restaurant_env.apply_exogenous_change()
    assert restaurant_env.state in POSSIBLE_NUMBER_OF_GUESTS
