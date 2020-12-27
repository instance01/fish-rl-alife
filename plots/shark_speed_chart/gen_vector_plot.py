import glob
import sys
import os
from collections import defaultdict

import numpy as np

from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

from tensorboard.plugins.distribution.compressor import compress_histogram_proto


def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def aggregate(path, run_wanted=0):
    keys = [
        #'Eval/Reward',
        # 'Eval/Length',
        # 'Eval/MCTS_Confidence',
        #'Train/AvgLoss'
        # 'Actor/Sample_length',
        # 'Train/Samples'
        'Train/Shark_Speed_H'
    ]

    agg_keys = {
        k: [] for k in keys
    }

    path = glob.glob(os.path.join(path, "events.out.tfevents.*"))
    if not path:
        return None, None
    path = path[0]

    import tensorflow as tf
    i = 0
    j = 0
    str_ = ''
    for event in my_summary_iterator(path):
        val = event.summary.value
        if not val:
            continue
        tag = event.summary.value[0].tag
        for key in keys:
            if tag.startswith(key):
                # run = tag[tag.rfind('/')+1:]
                # if int(run) != run_wanted:
                #     continue
                x = compress_histogram_proto(event.summary.value[0].histo)
                # import pdb; pdb.set_trace()
                agg_keys[key].append([y.value for y in x])
                if i in [1, 744, 988, 2288, 2971]:
                    arr = tf.make_ndarray(event.summary.value[0].tensor)
                    X = (arr[:, 0] + arr[:, 1]) / 2.
                    Y = np.vstack([X, arr[:, 2]]).T.tolist()
                    str_ += '\n\\addplot3 [area plot] coordinates {'
                    for x_ in Y:
                        str_ += str((x_[0], j, x_[1]))
                    str_ += '};\n'
                    j += 1
                i += 1
    print(str_)
    import pdb; pdb.set_trace()
    return agg_keys


def gen_tex_single_key(data, key):
    new_data = [[] for _ in range(len(data[0]))]
    for x in data:
        for i, y in enumerate(x):
            new_data[i].append(y)

    del new_data[1]
    del new_data[7]

    coords = [
        "".join(str((i, round(x, 3))) for i, x in enumerate(nd))
        for nd in new_data
    ]

    if key == 'Train/Samples':
        key = 'Training samples'
    if key == 'Actor/Sample_length':
        key = 'Actor episode rewards'

    return """
        \\begin{tikzpicture}
            \\begin{axis}[
                thick,smooth,no markers,
                grid=both,
                grid style={line width=.1pt, draw=gray!10},
                xlabel={Steps},
                ylabel={%s}]
            ]
            \\addplot+[name path=B,black,draw=none,line width=.001pt] coordinates {%s};
            \\addplot+[name path=C,black,draw=none,line width=.001pt] coordinates {%s};
            \\addplot+[name path=D,black,draw=none,line width=.001pt] coordinates {%s};
            \\addplot+[name path=A,black,draw=none,line width=.01pt] coordinates {%s};
            \\addplot+[name path=E,black,draw=none,line width=.001pt] coordinates {%s};
            \\addplot+[name path=F,black,draw=none,line width=.001pt] coordinates {%s};
            \\addplot+[name path=G,black,draw=none,line width=.001pt] coordinates {%s};

            \\addplot[blue!50,fill opacity=0.2] fill between[of=A and B];
            \\addplot[blue!50,fill opacity=0.4] fill between[of=A and C];
            \\addplot[blue!50,fill opacity=0.9] fill between[of=A and D];
            \\addplot[blue!50,fill opacity=0.9] fill between[of=A and E];
            \\addplot[blue!50,fill opacity=0.4] fill between[of=A and F];
            \\addplot[blue!50,fill opacity=0.2] fill between[of=A and G];
            \\end{axis}
        \\end{tikzpicture}
        """ % (
            key, *coords
        )


def gen_tex_single(agg_keys):
    gen = """\\documentclass{standalone}
        \\usepackage{tikz,pgfplots}

        \\pgfplotsset{compat=1.10}

        \\usepgfplotslibrary{fillbetween,external}
        \\tikzexternalize

        \\begin{document}
        """
    for key, data in agg_keys.items():
        gen += gen_tex_single_key(data, key)
    gen += "\\end{document}"
    return gen


def run():
    if len(sys.argv) < 2:
        for filename in os.listdir('runs/'):
            path = os.path.join('runs/', filename)
            if not os.path.isdir(path):
                continue
            print('Example path: ', path)
            return

    if len(sys.argv) == 2:
        path = sys.argv[1]
        print('-- Loading', path)
        agg_keys = aggregate(path, 0)
        gen = gen_tex_single(agg_keys)
        with open('testingtex/a.tex', 'w+') as f:
            f.write(gen)


if __name__ == '__main__':
    run()
