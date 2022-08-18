from Agent_based.base.program import BaseAgentProgram
from GLOBAL import Action, PROBABILITIES_STATES, PERCEPT_PROBABILITIES, TransactionState, COST_MANUAL_REVIEW, \
    TransactionPercept, COST_PROCESS_LEGIT
from functions_decisions_risk import theorem_of_bayes_inverse_conditional_probability, calculate_expected_value

import numpy as np


# additional: attitude towards risk
# save history


class FraudPreventionAgentProgram(BaseAgentProgram):
    """
    Add your program implementation for a decision-theoretic-agent here.
    You can also create additional classes for components of the agent inside this module.
    """

    def __init__(self, utility_function):
        self.utility_function = utility_function
        self.conditional_probabilies_transaction_details_on_state = PERCEPT_PROBABILITIES
        self.probabilities_states = PROBABILITIES_STATES
        self.probabilities_transaction_details = self._calculate_probabilities_of_transaction_details_()
        self.cost_manual_review = COST_MANUAL_REVIEW
        self.cost_process_legit_transaction = COST_PROCESS_LEGIT
        self.fixed_utilities = self._calculate_fixed_utilities_()
        self.conditional_probabilies_state_on_transaction_details = {}  # as the environment is episodic the probabilities for the state given a certain perception can be reused

    def _calculate_probabilities_of_transaction_details_(self):
        """Calculate the probability for the information system showing the different transaction details"""
        #### Required for the denominator to calculate conditional dependencies of the state dependent on the transactions details

        probabilities_transaction_details = {}

        for transaction_details in TransactionPercept:
            tmp = 0
            for state in TransactionState:
                tmp += self.conditional_probabilies_transaction_details_on_state[state][transaction_details.value] * \
                       self.probabilities_states[state]
            probabilities_transaction_details[transaction_details] = tmp

        return probabilities_transaction_details

    def _calculate_fixed_utilities_(self):
        """calculates the utility of the manual review and the processing a legit transaction"""
        ##### Doesn't have to be calculated in every step as the costs of these cases don't change

        utilities = {-self.cost_manual_review: self.utility_function(-self.cost_manual_review),
                     self.cost_process_legit_transaction: self.utility_function(self.cost_process_legit_transaction)}

        return utilities

    def calculate_conditional_probability_for_state_on_transaction_details(self, transaction_details):
        """identify the appropriate probabilities required to calculate the inverse conditional probability for all states.
        Transaction details the message of the information system  relevent to chose the appropriate variables."""

        probability_state_given_transaction_details = {}

        for state in TransactionState:
            probability_transaction_details_given_state = \
                self.conditional_probabilies_transaction_details_on_state[state][transaction_details.value]
            probability_state = self.probabilities_states[state]
            probability_transaction_details = self.probabilities_transaction_details[transaction_details]

            probability_state_given_transaction_details[state] = theorem_of_bayes_inverse_conditional_probability(
                probability_transaction_details_given_state,
                probability_state,
                probability_transaction_details)

        return probability_state_given_transaction_details

    def calculate_expected_utility_per_action(self, transaction_value, transaction_details):
        """Calculates the expected utility for all actions using the conditional probability of the state based
        on the transaction details"""

        utility_transaction_value = self.utility_function(-transaction_value)
        # print(f"The utility of reimbusing the tranbsaction value of {transaction_value} is {utility_transaction_value}")

        utility_outcomes = {Action.BLOCK: [self.fixed_utilities[-self.cost_manual_review],
                                           self.fixed_utilities[-self.cost_manual_review]],
                            Action.PROCESS: [utility_transaction_value,
                                             self.fixed_utilities[self.cost_process_legit_transaction]]}

        probabilities_states = [self.conditional_probabilies_state_on_transaction_details[transaction_details][state]
                                for state in TransactionState]

        expected_utilities = {}
        for action in Action:
            # print(f"expected utility: {action}")
            # print(f"expected utility - outcomes: {np.array(utility_outcomes[action])}")
            # print(f"expected utility - probabilities: {np.array(probabilities_states)}")
            utility = calculate_expected_value(np.array(utility_outcomes[action]), np.array(probabilities_states))
            expected_utilities[utility] = action

        return expected_utilities

    def calculate_utility_of_expected_value_per_action(self, transaction_value, transaction_details):
        """Calculates the expected value and subsequently the utility of the expected value.
        Returns dict with utilitiy of expected values for every action"""

        #### Not required for the agent programme, but created for the purpose of illustration

        outcomes = {Action.BLOCK: [-self.cost_manual_review, -self.cost_manual_review],
                    Action.PROCESS: [-transaction_value, self.cost_process_legit_transaction]}

        probabilities_states = [self.conditional_probabilies_state_on_transaction_details[transaction_details][state]
                                for state in TransactionState]

        utilities_expected_outcome = {}
        for action in Action:
            # print(f"utility of expected value: {action}")
            # print(f"utility of expected value - outcomes: {np.array(outcomes[action])}")
            # print(f"utility of expected value - probabilities: {np.array(probabilities_states)}")
            utility_outcome = self.utility_function(
                calculate_expected_value(np.array(outcomes[action]), np.array(probabilities_states)))
            utilities_expected_outcome[utility_outcome] = action

        return utilities_expected_outcome

    def choose_action(self, percept) -> Action:
        """Divides the perception into the two relevant parts, determines relevant conditional expected values,
        calculates expected utility and selections action. Returns action"""

        transaction_value = float(percept["value"])
        transaction_details = percept["other_transaction_info"]

        #### Determine relevant conditional probabilities for states based on transaction info
        if not transaction_details in self.conditional_probabilies_state_on_transaction_details:
            """As the probabilities of the messages displaying the transaction details don't change the values 
            can be calculated once and can then be stored"""

            # print(f"Probabilities for {transaction_details} hasn't been calculated yet")
            conditional_probabilities = self.calculate_conditional_probability_for_state_on_transaction_details(
                transaction_details)
            self.conditional_probabilies_state_on_transaction_details[transaction_details] = conditional_probabilities
            # print(f"The dictionary of conditional probabilities now contains: {
            # self.conditional_probabilies_state_on_transaction_details}")

        #### Not required
        utilities_expected_value = self.calculate_utility_of_expected_value_per_action(transaction_value,
                                                                                       transaction_details)
        # print(f"The utilities of the expected values are: {utilities_expected_value}")

        #### Calculate expected utilities
        expected_utilities = self.calculate_expected_utility_per_action(transaction_value, transaction_details)

        # print(f"The expected utilities are: {expected_utilities}")

        ### Chose action that maximises the expected utility
        chosen_action = expected_utilities[max([*expected_utilities])]
        # print(f"Therefore, I chose action: {chosen_action}")

        return chosen_action
