import os

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


def bar_plot(
        ax, data, err_data, data2, err_data2, colors=None, total_width=0.8,
        single_width=1, legend=True
    ):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of
        the data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        err = err_data[name]
        err2 = err_data2[name]
        values2 = data2[name]
        for x, y in enumerate(values):
            bar = ax.bar(
                x + x_offset, y, yerr=err[x], width=bar_width * single_width,
                color=colors[i % len(colors)], edgecolor='black', linewidth=.3
            )
            bar = ax.bar(
                x + x_offset, values2[x], yerr=err2[x], width=bar_width *
                single_width, color=colors[i % len(colors)], edgecolor='black',
                linewidth=.3, bottom=y+.15
            )

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())


def plot(data, err_data, data2, err_data2, xxx=22):
    scenarios = ['static-wait', 'DQN Stage II', 'PPO']
    fig, ax = plt.subplots(figsize=(4, 3.5))
    bar_plot(
        ax,
        data,
        err_data,
        data2,
        err_data2,
        # colors=['#6497b1', '#005b96', '#9FD983', '#009440'],
        colors=['#9FD983', '#009440'],
        total_width=.6,
        single_width=.9
    )

    plt.xticks([0, 1, 2], scenarios)
    plt.xlabel('Algorithm')
    plt.ylabel('Sustainability')
    plt.xlim(-0.5, 2.5)
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    # static-wait, dqn, ppo
    data = {
        "Swarm": [4.850000, 0.0, 33.050000],
        "TurnAway": [9.740000, 6.504000, 16.220000]
    }
    err_data = {
        "Swarm": [0.540000, 0.0, 0.640000],
        "TurnAway": [0.170000, 0.299598, 0.180000]
    }
    data2 = {
        "Swarm": [0.170000, 0.0, 7.530000],
        "TurnAway": [1.150000, 2.157000, 10.880000]
    }
    err_data2 = {
        "Swarm": [0.120000, 0.0, 0.210000],
        "TurnAway": [0.070000, 0.252410, 0.060000]
    }

    # # X fish
    # data = {
    #     "Static": [2.070000],
    #     "Static-wait": [4.850000],
    #     "DQN Stage II": [0.0],
    #     "PPO": [33.050000]
    # }
    # err_data = {
    #     "Static": [0.060000],
    #     "Static-wait": [0.540000],
    #     "DQN Stage II": [0.0],
    #     "PPO": [0.640000]
    # }
    # data2 = {
    #     "Static": [0.000000],
    #     "Static-wait": [0.170000],
    #     "DQN Stage II": [0.0],
    #     "PPO": [7.530000]
    # }
    # err_data2 = {
    #     "Static": [0.000000],
    #     "Static-wait": [0.120000],
    #     "DQN Stage II": [0.0],
    #     "PPO": [0.210000]
    # }
    fig = plot(data, err_data, data2, err_data2, xxx=20)
    fig.savefig("swarmcomp.pdf", bbox_inches='tight')
