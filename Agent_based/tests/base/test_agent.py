from unittest.mock import patch

from Agent_based.base.agent import Agent


def test_uses_fallback_program_when_no_program_is_supplied():
    agent = Agent()
    percept = "Some percept"
    user_action = "'Some user-specified action'"

    with patch('builtins.input', return_value=user_action):
        action = agent.program(percept)

        assert action == eval(user_action)
