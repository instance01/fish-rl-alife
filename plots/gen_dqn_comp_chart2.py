import os

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data

from matplotlib.image import BboxImage
from matplotlib.transforms import TransformedBbox
from matplotlib.transforms import Bbox
from matplotlib.legend_handler import HandlerBase

import matplotlib.patches as mpatches


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


class ImageHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        # enlarge the image by these margins
        sx, sy = self.image_stretch

        if self.special:
            ydescent += 6
            xdescent += 2
        xdescent -= 2
        # create a bounding box to house the image
        bb = Bbox.from_bounds(xdescent - sx,
                              ydescent - sy,
                              width + sx,
                              height + sy)

        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)

        self.update_prop(image, orig_handle, legend)

        return [image]

    def set_image(self, image_path, image_stretch=(0, 0), special=False):
        fn = get_sample_data(image_path, asfileobj=False)
        self.image_data = plt.imread(fn, format='png')
        self.image_stretch = image_stretch
        self.special = special


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
        hatch = None
        if 'Static' in name:
            hatch = '//'
        for x, y in enumerate(values):
            bar = ax.bar(
                x + x_offset, y, yerr=err[x], width=bar_width * single_width,
                color=colors[i % len(colors)], edgecolor='black', linewidth=.3,
                hatch=hatch
            )
            bar = ax.bar(
                x + x_offset, values2[x], yerr=err2[x], width=bar_width *
                single_width, color=colors[i % len(colors)], edgecolor='black',
                linewidth=.3, bottom=y+.15, hatch=hatch
            )

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    # if legend:
    #     ax.legend(bars, data.keys())
    return bars, data.keys()


def plot(data, err_data, data2, err_data2, xxx=22):
    scenarios = [100, 200, 300]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    bars, data_keys = bar_plot(
        ax,
        data,
        err_data,
        data2,
        err_data2,
        colors=['#6497b1', '#005b96', '#9FD983', '#009440'],
        total_width=.6,
        single_width=.9
    )

    fn = get_sample_data(os.getcwd() + "/fishpop.png", asfileobj=False)
    arr_img = plt.imread(fn, format='png')
    imagebox1 = OffsetImage(arr_img, zoom=0.07)
    imagebox1.image.axes = ax
    xy = [0.3, xxx]
    ab = AnnotationBbox(imagebox1, xy,
                        xybox=(40., 0.),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.1,
                        arrowprops=dict(
                            arrowstyle="-|>,head_length=.2,head_width=.1",
                            facecolor="w"
                            # connectionstyle="angle,angleA=0,angleB=90,rad=3")
                        ),
                        bboxprops=dict(
                            linewidth=0.0
                        ))
    ax.add_artist(ab)

    fn = get_sample_data(os.getcwd() + "/deadfish.png", asfileobj=False)
    arr_img = plt.imread(fn, format='png')
    imagebox2 = OffsetImage(arr_img, zoom=0.015)
    imagebox2.image.axes = ax
    xy = [0.3, 8]
    ab = AnnotationBbox(imagebox2, xy,
                        xybox=(40., 50.),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.1,
                        arrowprops=dict(
                            arrowstyle="-|>,head_length=.2,head_width=.1",
                            facecolor="w"
                            # connectionstyle="angle,angleA=0,angleB=90,rad=3")
                        ),
                        bboxprops=dict(
                            linewidth=0.0
                        ))
    ax.add_artist(ab)

    custom_handler1 = ImageHandler()
    custom_handler1.set_image(os.getcwd() + "/deadfish.png", image_stretch=(-6, -1))
    custom_handler2 = ImageHandler()
    custom_handler2.set_image(os.getcwd() + "/fishpop.png", image_stretch=(-2, 11), special=True)
    p1 = mpatches.Patch()
    p2 = mpatches.Patch()
    ax.legend([*bars, p1, p2], [*data_keys, 'Prey caught', 'Left-over prey'], handler_map={p1: custom_handler1, p2: custom_handler2})

    plt.xticks([0, 1, 2], scenarios)
    plt.xlabel('Time until reproduction possible', fontsize=11)
    plt.ylabel(r'Sustainability $\sigma$', fontsize=11)
    plt.xlim(-0.5, 2.5)
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    # 1 fish
    data = {
        "Static": [1.000000, 1.000000, 1.000000],
        "Static-wait": [8.290000, 3.910000, 2.970000],
        "DQN Stage II": [2.393000, 1.328000, 1.146000],
        "PPO": [13.900000, 5.300000, 2.480000]
    }
    err_data = {
        "Static": [0.000000, 0.000000, 0.000000],
        "Static-wait": [0.410000, 0.100000, 0.040000],
        "DQN Stage II": [0.190701, 0.052108, 0.028226],
        "PPO": [0.200000, 0.090000, 0.050000]
    }
    data2 = {
        "Static": [0.000000, 0.000000, 0.000000],
        "Static-wait": [1.070000, 0.980000, 1.000000],
        "DQN Stage II": [0.687000, 0.037000, 0.038000],
        "PPO": [10.520000, 2.880000, 3.280000]
    }
    err_data2 = {
        "Static": [0.000000, 0.000000, 0.000000],
        "Static-wait": [0.120000, 0.030000, 0.030000],
        "DQN Stage II": [0.149947, 0.013550, 0.014765],
        "PPO": [0.130000, 0.080000, 0.070000]
    }
    fig = plot(data, err_data, data2, err_data2, xxx=20)
    fig.savefig("dqncomp1.pdf", bbox_inches='tight')

    # 2 fish
    data = {
        "Static": [2.410000, 2.000000, 2.000000],
        "Static-wait": [9.740000, 5.000000, 3.900000],
        "DQN Stage II": [6.504000, 2.734000, 2.298000],
        "PPO": [16.220000, 8.110000, 4.570000]
    }
    err_data = {
        "Static": [0.150000, 0.000000, 0.000000],
        "Static-wait": [0.170000, 0.000000, 0.080000],
        "DQN Stage II": [0.299598, 0.076727, 0.039030],
        "PPO": [0.180000, 0.090000, 0.060000]
    }
    data2 = {
        "Static": [0.000000, 0.000000, 0.000000],
        "Static-wait": [1.150000, 1.000000, 1.030000],
        "DQN Stage II": [2.157000, 0.109000, 0.083000],
        "PPO": [10.880000, 2.580000, 3.520000]
    }
    err_data2 = {
        "Static": [0.000000, 0.000000, 0.000000],
        "Static-wait": [0.070000, 0.000000, 0.030000],
        "DQN Stage II": [0.252410, 0.025981, 0.024373],
        "PPO": [0.060000, 0.110000, 0.070000]
    }
    fig = plot(data, err_data, data2, err_data2)
    fig.savefig("dqncomp2.pdf", bbox_inches='tight')
