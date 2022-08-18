from GLOBAL import PROFIT_PER_SERVED_GUEST, CAPACITY_PER_WAITER, WAGE_PER_WAITER


def calculate_profit_with_action(percept, action):
    """Performance Measure: Profit - calculate Profit when action a is taken"""
    sales = PROFIT_PER_SERVED_GUEST * min(CAPACITY_PER_WAITER * action, percept)
    wages = WAGE_PER_WAITER * action

    return sales - wages


def determine_workload_with_action(percept, action):
    """Performance Measure: Workload - calculate Workload when action is taken"""
    if percept >= action * CAPACITY_PER_WAITER:
        workload = "high"
    else:
        workload = "load"

    return workload


def partial_utility_profit(profit):
    """Partial Utility: Utility of profit => In this case Utility = Profit"""
    return profit


def partial_utility_workload(workload):
    """Value when considering workload with action"""
    if workload == "high":
        workload_utility = -150
    else:
        workload_utility = 0

    return workload_utility


def determine_utility(profit, workload):
    """Trade-off of goals. NOTE: Due to mutual preferential independence the partial utilities are summed up"""
    return profit + workload
