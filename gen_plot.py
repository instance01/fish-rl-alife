import tensorflow as tf
import glob
import sys
import os
from collections import defaultdict

import numpy as np

from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record


def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def aggregate(path):
    keys = [
        # 'Train/Tot_Reward',
        # 'Eval/Length',
        # 'Eval/MCTS_Confidence',
        # 'Train/AvgLoss'
        # 'Eval/MCTS_Confidence'
        'Train/Last_Fish_Population'
    ]

    agg_runs = defaultdict(lambda: defaultdict(list))
    agg_keys = {}

    path = glob.glob(os.path.join(path, "events.out.tfevents.*"))
    if not path:
        return None, None
    path = path[0]

    data  =[]
    for event in my_summary_iterator(path):
        if not event.summary.value:
            continue
        tag = event.summary.value[0].tag
        for key in keys:
            # if tag.startswith(key):
            #     run = tag[tag.rfind('/')+1:]
            #     val = tf.make_ndarray(event.summary.value[0].tensor)
            #     agg_runs[key][run].append(val)
            if key == tag:
                val = float(tf.make_ndarray(event.summary.value[0].tensor))
                # print(val)
                data.append(val)

    # data = smooth(data, 20)[10:-9]
    # import pickle
    # print(data)
    # print(len(data))
    # with open('t800_5fish_last_fish_pop.pickle', 'wb+') as f:
    #     pickle.dump(data, f)
    # import matplotlib.pyplot as plt
    # plt.plot(data)
    # plt.show()

    for key in agg_runs:
        aggregated = []
        for run in agg_runs[key]:
            aggregated.append(agg_runs[key][run])
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
        agg_keys[key] = np.array(aggregated_)

    return agg_runs, agg_keys


def gen_tex_single_key(data, key, use_percentile):
    print(data[:, -1])
    if use_percentile:
        new_data = [
            # Smoothing loses 20 data points.
            #smooth(np.percentile(data, percentile, axis=0), 2)[1:-1]
            smooth(np.percentile(data, 50, axis=0), 20)[10:-9],
            #smooth(np.percentile(data, 10, axis=0), 20)[10:-9],
            [],
            smooth(np.percentile(data, 25, axis=0), 20)[10:-9],
            smooth(np.percentile(data, 75, axis=0), 20)[10:-9],
            #smooth(np.percentile(data, 90, axis=0), 20)[10:-9],
            [],
            #smooth(np.percentile(data, 0, axis=0), 20)[10:-9],
            [],
            #smooth(np.percentile(data, 100, axis=0), 20)[10:-9],
            [],
            #for percentile in [50, 10, 25, 75, 90, 0, 100]
        ]
    else:
        stddev = np.var(data, axis=0, ddof=1) ** .5
        mean = np.average(data, axis=0)
        new_data = [
            # Smoothing loses 20 data points.
            smooth(mean, 20)[10:-9],
            smooth(mean + 2 * stddev, 20)[10:-9],
            smooth(mean + 1 * stddev, 20)[10:-9],
            smooth(mean - 1 * stddev, 20)[10:-9],
            smooth(mean - 2 * stddev, 20)[10:-9],
            [],
            []
        ]

    coords = [
        "".join(str((i, round(x, 3))) for i, x in enumerate(nd))
        for nd in new_data
    ]

    red_line = ""
    if "Train" in key or "Eval" in key:
        #red_line = "".join(["(%d, 310)" % i for i, _ in enumerate(data[0])])
        red_line = "".join(["(%d, 430)" % i for i, _ in enumerate(data[0])])

    if key == 'Eval/Reward':
        key = 'Mean evaluation reward'
    if key == 'Eval/MCTS_Confidence':
        key = 'Search confidence'
    if key == 'Train/Tot_Reward':
        key = 'Total Reward'

    return """
        \\begin{tikzpicture}
            \\begin{axis}[
                thick,smooth,no markers,
                grid=both,
                grid style={line width=.1pt, draw=gray!10},
                xlabel={Steps},
                ylabel={%s}]
            ]
            \\addplot+[name path=A,black,line width=1pt] coordinates {%s};
            %%\\addplot+[name path=B,black,line width=.1pt] coordinates {%s};
            \\addplot+[name path=C,black,line width=.1pt] coordinates {%s};
            \\addplot+[name path=D,black,line width=.1pt] coordinates {%s};
            %%\\addplot+[name path=E,black,line width=.1pt] coordinates {%s};

            %%\\addplot+[name path=F,black,line width=.1pt] coordinates {%s};
            %%\\addplot+[name path=G,black,line width=.1pt] coordinates {%s};

            \\addplot+[name path=ZZZ,red,dashed,line width=.3pt] coordinates {%s};

            %%\\addplot[blue!50,fill opacity=0.2] fill between[of=A and B];
            \\addplot[blue!50,fill opacity=0.2] fill between[of=A and C];
            \\addplot[blue!50,fill opacity=0.2] fill between[of=A and D];
            %%\\addplot[blue!50,fill opacity=0.2] fill between[of=A and E];
            \\end{axis}
        \\end{tikzpicture}
        """ % (
            key, *coords, red_line
        )


def gen_tex_single(agg_runs, agg_keys, use_percentile=True):
    gen = """\\documentclass{standalone}
        \\usepackage{tikz,pgfplots}

        \\pgfplotsset{compat=1.10}

        \\usepgfplotslibrary{fillbetween,external}
        \\tikzexternalize

        \\begin{document}
        """
    for key in agg_runs:
        data = agg_keys[key]
        gen += gen_tex_single_key(data, key, use_percentile)
    gen += "\\end{document}"
    return gen


def gen_tex_multiple(agg_runs, agg_keys):
    gen = """\\documentclass{standalone}
        \\usepackage{tikz,pgfplots}

        \\pgfplotsset{compat=1.10}

        \\usepgfplotslibrary{fillbetween,external}
        \\tikzexternalize

        \\begin{document}
        """
    data = {}
    for key in agg_runs:
        data = agg_keys[key]
        gen += gen_tex_single(data, key)
    gen += "\\end{document}"
    return gen


def run():
    use_percentile = True
    pad_to_constant = True
    pad_constant = 5000

    if len(sys.argv) < 2:
        for filename in os.listdir('runs/'):
            path = os.path.join('runs/', filename)
            if not os.path.isdir(path):
                continue
            print('Example path: ', path)
            return

    if len(sys.argv) == 2:
        path = sys.argv[1]
        print('##-- Loading', path)
        agg_runs, agg_keys = aggregate(path)
        gen = gen_tex_single(agg_runs, agg_keys, use_percentile)
        with open('plottex/a.tex', 'w+') as f:
            f.write(gen)
        # In case I want to automate pdflatex too.
        # os.system('cd testingtex && lualatex --shell-escape a.tex && cd ..')

    if len(sys.argv) > 2:
        paths = []
        data = {}
        for path in sys.argv[1:]:
            print('##-- Loading', path)
            paths.append(path)
            agg_runs, agg_keys = aggregate(path)
            if not agg_runs and not agg_keys:
                continue
            for key in agg_runs:
                if key not in data:
                    data[key] = agg_keys[key]
                else:
                    xlen = max(data[key].shape[1], agg_keys[key].shape[1])
                    if pad_to_constant:
                        xlen = pad_constant

                    X = data[key]
                    Y = agg_keys[key]
                    if xlen > X.shape[1]:
                        X = np.pad(
                            X, ((0, 0), (0, xlen - X.shape[1])), mode='edge'
                        )
                    if xlen > Y.shape[1]:
                        Y = np.pad(
                            Y, ((0, 0), (0, xlen - Y.shape[1])), mode='edge'
                        )

                    data[key] = np.concatenate((X, Y))
        print('total', data[next(iter(agg_runs.keys()))].shape)
        gen = gen_tex_single(agg_runs, data, use_percentile)
        gen += "\n% " + ", ".join(paths)
        with open('plottex/a.tex', 'w+') as f:
            f.write(gen)


if __name__ == '__main__':
    run()
