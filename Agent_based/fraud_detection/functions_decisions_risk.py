import numpy as np


def theorem_of_bayes_inverse_conditional_probability(inverse_conditional_probability, probability_observation,
                                                     probability_condition):
    """calculates the conditional probability from the inverse conditional probability"""

    return (inverse_conditional_probability * probability_observation) / probability_condition


def calculate_expected_value(data, weights):
    """takes same sized lists of data and weights and returns an expected value"""

    if not len(data) == len(weights):
        print(f"Not the same number of items passed for data and weigths. "
              f"Items data: {len(data)}, "
              f"Items Weights: {len(weights)}")
        return

    multiplied = data * weights
    expected_value = np.sum(multiplied)

    # print(f"I calculated the following expected value {expected_value}")

    return float(expected_value)


def calculate_utilities(utility_function, data):
    """create list with the results of the data used on the passed utility function"""
    result = []

    for date in data:
        result.append(utility_function(date))

    return result
