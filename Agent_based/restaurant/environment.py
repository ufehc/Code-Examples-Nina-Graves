import random

from Agent_based.base.environment import BaseEnvironment
from GLOBAL import POSSIBLE_NUMBER_OF_GUESTS


class RestaurantEnvironment(BaseEnvironment):

    def __init__(self):
        super().__init__()
        self.state = random.choice(POSSIBLE_NUMBER_OF_GUESTS)

    def percept(self, agent):
        """Restaurant environment is fully observable, so percept = state."""
        return self.state

    def reward(self, action):
        """NOTE: Your utility-based agent must NOT use a hard-coded utility matrix.
        We only use it here to conveniently specify the environment's performance measure."""
        utility_matrix = {(1, 0): -100, (1, 20): 50, (1, 40): 50, (1, 60): 50,
                          (2, 0): -200, (2, 20): 100, (2, 40): 250, (2, 60): 250,
                          (3, 0): -300, (3, 20): 0, (3, 40): 300, (3, 60): 450,
                          (4, 0): -400, (4, 20): -100, (4, 40): 200, (4, 60): 500}
        return utility_matrix[(action, self.state)]

    def apply_exogenous_change(self):
        self.state = random.choice(POSSIBLE_NUMBER_OF_GUESTS)