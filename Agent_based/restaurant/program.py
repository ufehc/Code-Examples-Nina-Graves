from Agent_based.base.program import BaseAgentProgram
from calculation_KPIs_Utility import calculate_profit_with_action, determine_workload_with_action, \
    partial_utility_profit, partial_utility_workload, determine_utility
from GLOBAL import ACTION_SPACE


class RestaurantAgentProgram(BaseAgentProgram):
    """Add your program implementation for a utility-based agent here."""

    def __init__(self):
        self.history = []  # list of tuples (percept, action)
        self.calculated_utilities = {}  # dict containing the tuple of percept and action as key and
        self.percept_action = {}
        self.performance = 0  # is used to keep track of the current state
        # self.plan = [] #As the former actions don't have an influence on the next percept and the environment is
        # creating a plan isn't considered beneficial

    def update_state(self, percept, action, utility):
        """Update current state of the World NOTE: as the former actions don't have an influence on following percepts
        updating the performance measure and keeping track of past actions is sufficient to model the current
        state of the world. As the new percept is independent of former percepts it is not relevant wether the state is
        update before or after the action is chosen. To provide a complete history of all percepts and actions after a
        given number of steps the state is updated after the choice was made in this scenario."""
        self.history.append((percept, action))
        self.performance += utility

    def update_internal_decision_dictionairy(self, utilities, chosen_action, percept):
        """Add information about the utilities and choices for a given percept"""
        for entry in utilities:
            self.calculated_utilities.update({(percept, utilities[entry]): entry})

        self.percept_action.update({percept: chosen_action})
        print(f"Calculated utilieties include: {self.calculated_utilities}")
        print(f"Rules now holds the following strategies: {self.percept_action}")

    def calculate_utility_of_action(self, percept, action):
        """Calculate Performance Indicators, Partial Utilities and Total Ultility"""
        profit = calculate_profit_with_action(percept, action)
        workload = determine_workload_with_action(percept, action)
        utility_profit = partial_utility_profit(profit)
        utility_workload = partial_utility_workload(workload)
        utility = determine_utility(utility_profit, utility_workload)

        return {utility: action}

    def calculate_utility_of_all_actions(self, percept):
        """As the set of actions is small and discrete the chosen approach is to test all possible actions in regard to
        the perception of the state"""
        utilities = {}

        for action in ACTION_SPACE:
            utility = self.calculate_utility_of_action(percept, action)
            utilities.update(utility)

        return utilities

    def choose_action(self, percept) -> int:
        """Choose action by looking up whether optimal action for the given perception was already determined and if
        not calculating the utility of all possible actions"""

        # In this case state of the world is updated after the decision for the action is made as it does
        # as the only parameter relevant for the decision is the percept

        if percept in self.percept_action:
            action = self.percept_action[percept]
            utility = self.calculated_utilities[(percept, action)]
            print(f"I already know what to do if I have {percept} guests: {action} waiters with a utility of {utility}")

        else:
            utilities = self.calculate_utility_of_all_actions(percept)
            print(f"The following utilities have been calculated: {utilities}")
            utility = max(utilities.keys())
            print(f"The maximum utility I identified is {utility}")
            action = utilities[utility]
            print(f"Which means I chose the following action: {action}")

            self.update_internal_decision_dictionairy(utilities, action, percept)


        self.update_state(percept, action, utility)
        print(f"The following performance is anticipated {self.performance}")

        return action