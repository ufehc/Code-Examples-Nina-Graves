from restaurant.program import RestaurantAgentProgram


def test_returns_valid_action():
    number_of_guests = 20
    program = RestaurantAgentProgram()

    action = program(percept=number_of_guests)

    assert type(action) == int


def test_is_rational():
    program = RestaurantAgentProgram()

    assert program(percept=0) == 1
    assert program(percept=20) == 2
    assert program(percept=40) == 3
    assert program(percept=60) == 4
