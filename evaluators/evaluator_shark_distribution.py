import os
import glob

import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record


def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def aggregate_multi(paths1, paths2, paths3):
    aggregate(paths1)
    aggregate(paths2)
    aggregate(paths3)


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
            if tag == 'Train/Shark_Speed_H':
                val = tf.make_ndarray(event.summary.value[0].tensor)
                peaks, _ = find_peaks(val[:, -1], distance=8)
                data.append((val, len(peaks) >= 2))

        if len(data) < 100:
            continue
        # data = smooth(data, 20)[10:-9]
        aggregated.append(data[-1][1])

    print(sum([1 for x in aggregated if x]))
    print(sum([1 for x in aggregated if not x]))
    return aggregated


def run():
    paths1 = glob.glob('/big/r/ratke/runs14/*ma3_obs_starve_maxsteps_t*_i2_p150*')
    paths2 = glob.glob('/big/r/ratke/runs15/*ma3_obs_starve_maxsteps_t*_i2_p150*')
    paths3 = glob.glob('/big/r/ratke/runs16/*ma3_obs_starve_maxsteps_t*_i2_p150*')

    aggregate_multi(paths1, paths2, paths3)


if __name__ == '__main__':
    run()
