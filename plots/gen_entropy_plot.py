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
    print(arr_d100)

    fig, ax = plt.subplots()

    def sub_plot(arr, col):
        mean = np.mean(arr, axis=0)
        x = np.arange(mean.shape[0])
        ci = 1.96 * np.std(arr, axis=0) / np.mean(arr, axis=0)

        ax.plot(x, mean, color=col)
        ax.fill_between(x, (mean - ci), (mean + ci), color=col, alpha=.1)

    sub_plot(arr_d100, 'c')
    plt.ylabel("Entropy")
    plt.ylim((0, 4))
    plt.show()


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
            if tag.endswith('policy_entropy'):
                val = float(tf.make_ndarray(event.summary.value[0].tensor))
                data.append(val)

        if len(data) < 100:
            continue
        data = smooth(data, 20)[10:-9]
        # data = smooth(data, 60)[30:-29]
        print(len(data))
        aggregated.append(data)

    if not aggregated:
        return []
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
    paths_d100 = glob.glob('/big/r/ratke/runs30/*two_net_vd35_f*')
    # paths_d100 = glob.glob('/big/r/ratke/runs30/*i5_p150_r10_s035_sp200_two_net_vd35_f*')

    aggregate_multi(paths_d100, [], [], [])


if __name__ == '__main__':
    run()
