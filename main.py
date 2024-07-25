import gym
from single_run_20 import single_run_20
from policy_evaluation import policy_evaluation
from monte_carlo_es import monte_carlo_es
from plot_value_function import plot_value_function
from matplotlib import pyplot as plt

def main():
    env = gym.make('Blackjack-v1')

    single_run_20()
    monte_carlo_es()

    value = policy_evaluation()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 15), subplot_kw={'projection': '3d'})
    axes[0].set_title('Value function without usable ace')
    axes[1].set_title('Value function with usable ace')
    plot_value_function(value, axes[0], axes[1])
    plt.show()

if __name__ == "__main__":
    main()
