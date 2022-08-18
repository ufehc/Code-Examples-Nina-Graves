from Agent_based.base.agent import Agent
from environment import RestaurantEnvironment
from program import RestaurantAgentProgram


def main():
    restaurant_environment = RestaurantEnvironment()

    agents = [Agent(program=RestaurantAgentProgram(), name="Our agent"),
              Agent(program=RestaurantAgentProgram(), name="Your agent")]

    restaurant_environment.add_agents(agents)

    lifetime = 10
    for _ in range(lifetime):
        restaurant_environment.step()

    print("The agents showed the following performance over a lifetime of {} timesteps...".format(lifetime))
    for agent in restaurant_environment.agents:
        print("{agent_name}: {performance}".format(agent_name=agent.name, performance=agent.performance))


if __name__ == "__main__":
    main()
