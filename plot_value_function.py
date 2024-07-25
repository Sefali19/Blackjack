import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

def plot_value_function(V, ax1, ax2):
    player_sum_range = np.arange(12, 22)
    dealer_card_range = np.arange(1, 11)
    X, Y = np.meshgrid(dealer_card_range, player_sum_range)

    ax1.plot_wireframe(X, Y, V[:, :, 0])
    ax2.plot_wireframe(X, Y, V[:, :, 1])

    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('player sum')
        ax.set_xlabel('dealer showing')
        ax.set_zlabel('state-value')
