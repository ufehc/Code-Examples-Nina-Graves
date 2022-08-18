from unittest.mock import patch, Mock

from Agent_based.base.program import BaseAgentProgram


@patch.multiple(BaseAgentProgram, __abstractmethods__=set())
def test_calls_program():
    program = BaseAgentProgram()
    program.choose_action = Mock(return_value=2)

    assert program(percept="some percept") == 2