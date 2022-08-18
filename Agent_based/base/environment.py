from abc import ABC, abstractmethod


class BaseEnvironment(ABC):

    def __init__(self):
        self.agents = []

    def add_agents(self, agents):
        self.agents.extend(agents)

    def add_agent(self, agent):
        self.agents.append(agent)

    def step(self):
        actions_by_agents = self.collect_actions()
        self.execute_actions(actions_by_agents)
        self.apply_exogenous_change()
    
    def collect_actions(self):
        actions_by_agents = []
        for agent in self.agents:
            percept = self.percept(agent)
            action = agent(percept)
            actions_by_agents.append((agent, action))
        return actions_by_agents

    def reset(self):
        for agent in self.agents:
            agent.performance = 0

    @abstractmethod
    def percept(self, agent):
        pass

    def execute_actions(self, actions_by_agents):
        for agent, action in actions_by_agents:
            self.execute_action(agent, action)

    def execute_action(self, agent, action):
        agent.performance += self.reward(action)

    @abstractmethod
    def reward(self, action):
        pass

    @abstractmethod
    def apply_exogenous_change(self):
        pass

