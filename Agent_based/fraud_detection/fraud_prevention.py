import math


from environment import FraudPreventionEnvironment
from program import FraudPreventionAgentProgram

from Agent_based.base.agent import Agent


def main():

    fraud_prevention_environment = FraudPreventionEnvironment()

    agents = [Agent(program=FraudPreventionAgentProgram(utility_function=lambda x: x), name="Agent 1"),
              Agent(program=FraudPreventionAgentProgram(utility_function=lambda x: -100 * math.exp(-0.0015 * x) + 100),
                    name="Agent 2"),
              Agent(program=FraudPreventionAgentProgram(utility_function=lambda x: math.exp(0.005 * x) + 300),
                    name="Agent 3")]

    fraud_prevention_environment.add_agents(agents)


    lifetime = 100
    for _ in range(lifetime):
        fraud_prevention_environment.step()

    print("The agents showed the following performance over a lifetime of {} timesteps...".format(lifetime))
    for agent in fraud_prevention_environment.agents:
        print("{agent_name}: {performance}".format(agent_name=agent.name, performance=agent.performance))

    fraud_prevention_environment.reset()
    pass


if __name__ == "__main__":
    main()
