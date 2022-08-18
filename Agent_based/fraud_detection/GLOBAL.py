import enum


class Action(enum.Enum):
    BLOCK = "block transaction and trigger manual review"
    PROCESS = "process transaction"


class TransactionState(enum.Enum):
    FRAUDULENT = 1
    LEGITIMATE = 0


class TransactionPercept(enum.Enum):
    ORDINARY_TIME_AND_LOCATION = 0
    SUSPICIOUS_LOCATION = 1
    SUSPICIOUS_TIME = 2
    SUSPICIOUS_TIME_AND_LOCATION = 3


FRAUD_PROBABILITY = 0.1

PROBABILITIES_STATES = {TransactionState.FRAUDULENT: FRAUD_PROBABILITY,
                        TransactionState.LEGITIMATE: 1 - FRAUD_PROBABILITY}

PERCEPT_PROBABILITIES = {TransactionState.FRAUDULENT: [0.1, 0.2, 0.2, 0.5],
                         TransactionState.LEGITIMATE: [0.8, 0.1, 0.1, 0.0]}

COST_MANUAL_REVIEW = 100
COST_PROCESS_LEGIT = 0
