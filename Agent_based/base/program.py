from abc import ABC, abstractmethod


class BaseAgentProgram(ABC):

    def __call__(self, percept):
        return self.choose_action(percept)

    @abstractmethod
    def choose_action(self, percept):
        pass
