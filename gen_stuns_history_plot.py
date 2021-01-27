import os
import sys
import glob
from collections import defaultdict

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record


def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def aggregate_multi(paths_d100, paths_d300, paths_d500, paths_d700):
    arr_d100 = aggregate(paths_d100)
    arr_d300 = aggregate(paths_d300)
    arr_d500 = aggregate(paths_d500)
    arr_d700 = aggregate(paths_d700)

    fig, ax = plt.subplots()

    def sub_plot(arr, col):
        mean = np.mean(arr, axis=0)
        x = np.arange(mean.shape[0])
        # TODO Use scipy here.. no idea if this is correct.
        ci = 1.96 * np.std(arr, axis=0) / np.mean(arr, axis=0)

        ax.plot(x, mean, color=col)
        ax.fill_between(x, (mean - ci), (mean + ci), color=col, alpha=.1)

    sub_plot(arr_d100, 'c')
    sub_plot(arr_d300, 'b')
    sub_plot(arr_d500, 'r')
    sub_plot(arr_d700, 'g')
    plt.ylim((1, 7))
    plt.show()

    return arr


def aggregate(paths):
    aggregated = []
    for path in paths:
        path = glob.glob(os.path.join(path, "events.out.tfevents.*"))
        if not path:
            continue
        path = path[0]

        data = []
        for event in my_summary_iterator(path):
            if not event.summary.value:
                continue
            tag = event.summary.value[0].tag
            if tag == 'Train/Stuns':
                val = float(tf.make_ndarray(event.summary.value[0].tensor))
                # print(val)
                data.append(val)

        if len(data) < 100:
            continue
        # data = smooth(data, 20)[10:-9]
        data = smooth(data, 60)[30:-29]
        print(len(data))
        aggregated.append(data)

    max_len = max(len(x) for x in aggregated)
    aggregated_ = []
    for i, x in enumerate(aggregated):
        aggregated_.append(
            np.pad(
                x,
                (0, max_len - len(x)),
                mode='constant',
                constant_values=(0, x[-1])
            )
        )
    arr = np.array(aggregated_)
    return arr


def run():
    use_percentile = True
    pad_to_constant = True
    pad_constant = 14000

    paths_d100 = glob.glob('runs/*d100*')
    paths_d300 = glob.glob('runs/*d300*')
    paths_d500 = glob.glob('runs/*d500*')
    paths_d700 = glob.glob('runs/*d700*')

    aggregate_multi(paths_d100, paths_d300, paths_d500, paths_d700)

    # if len(sys.argv) > 1:
    #     data = aggregate(sys.argv[1:])


if __name__ == '__main__':
    run()
