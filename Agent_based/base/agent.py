from collections.abc import Callable
from random import randint


class Agent:

    def __init__(self, program=None, name=None):
        self.name = name if name else "Agent{}".format(randint(0, 1000))
        self.performance = 0

        if program is None or not isinstance(program, Callable):
            class_name = self.__class__.__name__
            print("Can't find a valid program for {} {}, falling back to default.".format(class_name, self.name))

            def program(percept):
                return eval(input('Percept={}; action? '.format(percept)))

        self.program = program

    def __call__(self, percept):
        return self.program(percept)
