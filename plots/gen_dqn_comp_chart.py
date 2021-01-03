import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


def bar_plot(ax, data, err_data, colors=None, total_width=0.8, single_width=1, legend=True):
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
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, yerr=err[x], width=bar_width * single_width, color=colors[i % len(colors)], edgecolor='black', linewidth=.3)

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())


def plot(data, err_data):
    scenarios = ['1', '2', '5', '10', '15', '20']
    fig, ax = plt.subplots()
    # bar_plot(ax, data, err_data, colors=['#006a4e', '#024064'], total_width=.8, single_width=.9)
    # bar_plot(ax, data, err_data, colors=['#9FD983', '#3CB3C0'], total_width=.8, single_width=.9)
    bar_plot(ax, data, err_data, colors=['#9FD983', '#009440'], total_width=.6, single_width=.9)
    plt.xticks([0, 1, 2, 3, 4, 5, 6], scenarios)
    plt.xlabel('Initial Fish Population')
    plt.ylabel('Avg Episodic Reward')
    # plt.xlim(-0.5, 3.5)
    plt.xlim(-0.5, 5.5)
    plt.show()


if __name__ == "__main__":
    # fish procreation time 100
    data = {
        # "DQN Stage II": [40.816667, 78.216667, 187.966667, 200.850000, 203.900000, 212.500000],
        "DQN Stage II": [33.060000, 77.760000, 203.160000, 224.920000, 234.330000, 246.520000],
        "PPO": [142.230000, 162.550000, 194.300000, 187.320000, 205.960000, 233.310000]
    }
    err_data = {
        "DQN Stage II": [2.316720, 2.367767, 2.068570, 2.492685, 2.004670, 2.259842],
        "PPO": [1.490000, 1.780000, 2.120000, 2.610000, 2.690000, 2.380000]
    }

    plot(data, err_data)

    # fish procreation time 500
    data = {
        "DQN Stage II": [11.300000, 21.500000, 53.600000, 112.400000, 169.500000, 200.400000],
        "PPO": [0.530000, 16.690000, 50.000000, 119.440000, 173.810000, 218.670000]
    }
    err_data = {
        "DQN Stage II": [0.470113, 0.499144, 0.670984, 1.220569, 1.847904, 2.599611],
        "PPO": [0.140000, 0.330000, 0.440000, 0.570000, 0.770000, 1.070000]
    }

    plot(data, err_data)
