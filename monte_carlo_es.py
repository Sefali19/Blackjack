import numpy as np
from generate_episode import generate_episode_from_policy
import gym

env = gym.make('Blackjack-v1')


def monte_carlo_es(maxiter=5000000):
    """ Implementation of Monte Carlo ES """
    pi = np.zeros((10, 10, 2), dtype=int)
    Q = np.ones((10, 10, 2, 2)) * 100  # optimistic initialization of Q
    returns = np.zeros((10, 10, 2, 2))
    visits = np.zeros((10, 10, 2, 2))

    for i in range(maxiter):
        if i % 100000 == 0:
            print(f"Iteration: {i}")
            print("Policy (no usable ace):")
            print(pi[:, :, 0])
            print("Policy (usable ace):")
            print(pi[:, :, 1])

        states, actions, rewards = generate_episode_from_policy(env, pi)
        G = 0
        visited = set()
        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            action = actions[t]
            G += rewards[t]
            state_action = (state[0] - 12, state[1] - 1, int(state[2]), action)
            if state_action not in visited:
                returns[state_action] += G
                visits[state_action] += 1
                Q[state_action] = returns[state_action] / visits[state_action]
                pi[state_action[:-1]] = np.argmax(Q[state[0] - 12, state[1] - 1, int(state[2]), :])
                visited.add(state_action)

    print("Optimal policy (no usable ace):")
    print(pi[:, :, 0])
    print("Optimal policy (usable ace):")
    print(pi[:, :, 1])
