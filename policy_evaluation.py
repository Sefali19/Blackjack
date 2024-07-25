import numpy as np
from single_run_20 import single_run_20

def policy_evaluation(maxiter=10000):
    """ Implementation of first-visit Monte Carlo prediction """
    V = np.zeros((10, 10, 2))
    returns = np.zeros((10, 10, 2))
    visits = np.zeros((10, 10, 2))
    for _ in range(maxiter):
        episode_states, episode_return = single_run_20()
        visited_states = set()
        for state in episode_states:
            player_sum, dealer_card, useable_ace = state
            if (player_sum, dealer_card, useable_ace) not in visited_states:
                idx = (player_sum - 12, dealer_card - 1, int(useable_ace))
                returns[idx] += episode_return
                visits[idx] += 1
                V[idx] = returns[idx] / visits[idx]
                visited_states.add((player_sum, dealer_card, useable_ace))
    return V
