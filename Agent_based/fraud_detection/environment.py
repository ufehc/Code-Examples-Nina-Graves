import enum

from scipy.stats import uniform, multinomial, binom
from GLOBAL import TransactionState, FRAUD_PROBABILITY, PERCEPT_PROBABILITIES, Action, COST_MANUAL_REVIEW, \
    TransactionPercept

from Agent_based.base.environment import BaseEnvironment


class FraudPreventionEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.value = uniform.rvs(size=1, scale=1000)
        self.fraud = TransactionState(binom.rvs(n=1, p=FRAUD_PROBABILITY))
        self.fraud_percept = TransactionPercept(self._draw_multinomial())

    def _draw_multinomial(self):
        return list(multinomial.rvs(n=1, p=PERCEPT_PROBABILITIES[self.fraud])).index(1)

    def percept(self, agent):
        return {"value": self.value, "other_transaction_info": self.fraud_percept}

    def reward(self, action):
        if action == Action.BLOCK:
            return -COST_MANUAL_REVIEW
        elif self.fraud == TransactionState.FRAUDULENT:
            return -float(self.value)
        else:
            return 0

    def apply_exogenous_change(self):
        self._retrieve_new_transaction()

    def _retrieve_new_transaction(self):
        self.value = uniform.rvs(size=1, scale=1000)
        self.fraud = TransactionState(binom.rvs(n=1, p=FRAUD_PROBABILITY))
        self.fraud_percept = TransactionPercept(self._draw_multinomial())

