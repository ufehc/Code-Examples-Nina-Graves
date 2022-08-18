# Add your experimental setup here. Use the code from fraud_prevention.py as a starting point.
import math

from Agent_based.base.agent import Agent
from environment import FraudPreventionEnvironment
from program import FraudPreventionAgentProgram

import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm


def main():
    number_timesteps = [10, 100, 1000, 10000]
    number_experiments = list(np.arange(1, 101))

    fraud_prevention_environment = FraudPreventionEnvironment()

    agents = [Agent(program=FraudPreventionAgentProgram(utility_function=lambda x: x), name="Agent 1"),
              Agent(program=FraudPreventionAgentProgram(utility_function=lambda x: -100 * math.exp(-0.0015 * x) + 100),
                    name="Agent 2"),
              Agent(program=FraudPreventionAgentProgram(utility_function=lambda x: math.exp(0.005 * x) + 300),
                    name="Agent 3")]

    fraud_prevention_environment.add_agents(agents)

    agents_names = [x.name for x in fraud_prevention_environment.agents]
    product = itertools.product(number_timesteps, agents_names)
    cols = pd.MultiIndex.from_tuples(product, names=["Timesteps", "Agent"])
    results = pd.DataFrame(columns=cols, index=number_experiments)

    for timesteps in tqdm(number_timesteps):
        for experiment in tqdm(number_experiments):
            for _ in range(timesteps):
                fraud_prevention_environment.step()

            for agent in fraud_prevention_environment.agents:
                results.loc[experiment, (timesteps, agent.name)] = agent.performance

            fraud_prevention_environment.reset()

    results.to_csv("Results_Experiment.csv")


if __name__ == "__main__":
    main()
