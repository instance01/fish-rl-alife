import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def sliding_mean(data: np.ndarray, window: int):
    if window > 1:
        sliding_mean = []
        min_index = 0
        max_index = len(data) - 1
        for index in range(len(data)):
            top = min(index + window // 2, max_index)
            bottom = max(index - window // 2, min_index)
            sliding_mean.append(np.mean(data[bottom:top]))
        return sliding_mean
    return data


COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'black']


def mode_plot(experiment_index: int, file_index: int, window: int):

    root = "./runs"
    folder_name = "statistics"
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    sub_folders.sort()

    # which files
    exp_folder = sub_folders[experiment_index]
    stats_folder = os.path.join(exp_folder, folder_name)
    stats_files = os.listdir(stats_folder)
    stats_files.sort()
    file = stats_files[file_index]
    path = os.path.join(stats_folder, file)

    # get data
    fig, axis = plt.subplots()
    legend = []
    data = pd.read_csv(path)
    rewards = data[[key for key in data.keys() if 'reward' in key][0]]
    episodes = data[[key for key in data.keys() if 'episode' in key][0]]
    epsilon = data[[key for key in data.keys() if 'epsilon' in key][0]]

    if window is not None and window > 1:
        rewards = sliding_mean(rewards, window)

    line = axis.plot(episodes, rewards, color='blue')
    legend.append((line[0], 'rewards'))

    axis = axis.twinx()
    axis.set_ylim(ymin=0, ymax=1)

    line = axis.plot(episodes, epsilon, color='green')
    legend.append((line[0], 'epsilon'))

    try:  # try plot fish_density
        fish_density = data[[key for key in data.keys() if 'density' in key][0]]
        if window is not None and window > 1:
            fish_density = sliding_mean(fish_density, window)
        line = axis.plot(episodes, fish_density, color='red')
        legend.append((line[0], 'fish_density'))
    except (KeyError, IndexError):
        warn('No fish density data found!!!')

    # print info
    find_information(os.path.join(exp_folder, 'info.csv'))

    # create plot
    plt.title(file)
    lines, labels = list(zip(*legend))
    plt.legend(lines, labels, loc=4)
    plt.show()


def mode_info(indices: [int]):
    root = "./runs"
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    sub_folders.sort()
    for i in indices:
        print("################", sub_folders[i], "################")
        find_information(os.path.join(sub_folders[i], 'info.csv'))


def find_information(path: str):

    if not path.endswith('.csv'):
        raise ValueError('File is not a csv file !!!')

    info = pd.read_csv(os.path.join(path), index_col=False)
    TAB = 40
    MAX_LINE_LEN = 70

    print("##########", path, "##########")
    for key in info.keys():
        line = str(info[key][0])
        white_space = TAB - len(key)

        if len(line) <= MAX_LINE_LEN:
            print(key, '--->', (" " * white_space), line)

        else:

            split = []
            for x in range(len(line) // MAX_LINE_LEN):
                split.append(line[x * MAX_LINE_LEN: (x + 1) * MAX_LINE_LEN])
            split.append(line[(x + 1) * MAX_LINE_LEN:])

            for i, txt in enumerate(split):
                if i == 0:
                    print(key, '--->', (" " * white_space), txt)
                else:
                    print('---> ', (" " * TAB), txt)


def mode_build(experiment_index:int, file_index:int, window:int, tag: [str]):
    root = "./runs"
    folder_name = "statistics"
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    sub_folders.sort()

    # which files
    exp_folder = sub_folders[experiment_index]
    stats_folder = os.path.join(exp_folder, folder_name)
    stats_files = os.listdir(stats_folder)
    stats_files.sort()
    file = stats_files[file_index]
    path = os.path.join(stats_folder, file)

    # get data
    fig, axis = plt.subplots()
    legend = []
    data = pd.read_csv(path)

    small_axis = axis.twinx()
    #small_axis.set_ylim([0, 0.05])
    #small_axis.set_ylim([0, 1.0])

    episodes = data[[k for k in data.keys() if 'episode' in k][0]]
    keys = [key for tag in tags for key in data.keys() if tag in key]

    if len(keys) == 0:
        raise ValueError('Tag {} is not part of any key!'.format(tag))

    for key in keys:
        if len(COLORS) == 0:
            R = np.random.randint(0, 256)
            G = np.random.randint(0, 256)
            B = np.random.randint(0, 256)
            color = '#' + str(hex(R))[2:] + str(hex(G))[2:] + str(hex(B))[2:]
        else:
            color = COLORS.pop()

        values = data[key]
        if 'speed' in key:
            values *= 1 / 0.04

        values = data[key]
        if window is not None and window > 1:
            values = sliding_mean(values, window)
        if max(values) <= 1:
            line = small_axis.plot(episodes, values, color=color)
        else:
            line = axis.plot(episodes, values, color=color)
        legend.append((line[0], key))

    # print info
    find_information(os.path.join(exp_folder, 'info.csv'))

    # create plot
    plt.title(file)
    lines, labels = list(zip(*legend))
    plt.legend(lines, labels, loc=4, fontsize='small')
    plt.show()


def mode_hist(experiment_index: int, file_index: int):
    root = "./runs"
    folder_name = "stistics"
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    sub_folders.sort()

    # which files
    exp_folder = sub_folders[experiment_index]
    stats_folder = os.path.join(exp_folder, folder_name)
    stats_files = os.listdir(stats_folder)
    stats_files.sort()
    file = stats_files[file_index]
    path = os.path.join(stats_folder, file)

    # get data
    data = pd.read_csv(path)
    # first key = [0] and first element [0]
    fish_win = data[[k for k in data.keys() if 'fish' in k and 'score' in k][0]][0]
    shark_win = data[[k for k in data.keys() if 'shark' in k and 'score' in k][0]][0]

    y_pos = [0, 1]
    plt.bar(y_pos, [fish_win, shark_win], align='center', alpha=0.5)
    plt.xticks(y_pos, ['fishes', 'sharks'])
    plt.ylabel('sum wins')
    plt.title('win histogram')
    plt.show()


def mode_keys(experiment_index: int, file_index: int):
    root = "./runs"
    folder_name = "statistics"
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    sub_folders.sort()

    # which files
    exp_folder = sub_folders[experiment_index]
    stats_folder = os.path.join(exp_folder, folder_name)
    stats_files = os.listdir(stats_folder)
    stats_files.sort()
    file = stats_files[file_index]
    path = os.path.join(stats_folder, file)
    data = pd.read_csv(path)
    [print(k) for k in data.keys()]


if __name__ == '__main__':
    if len(sys.argv) == 1:
        exit(0)

    mode = sys.argv[1]
    if mode == 'plot':
        which_experiment = int(sys.argv[2])
        which_file = int(sys.argv[3])
        window = int(sys.argv[4])
        mode_plot(which_experiment, which_file, window,)
        exit(0)
    elif mode == 'info':
        indices = sys.argv[2].split(',')
        indices = list(map(lambda s: int(s), indices))
        mode_info(indices)
        exit(0)
    elif mode == 'build':
        which_experiment = int(sys.argv[2])
        which_file = int(sys.argv[3])
        window = int(sys.argv[4])
        tags = [tag for tag in sys.argv[5].split(',')]
        mode_build(which_experiment, which_file, window, tags)
        exit(0)
    elif mode == 'winner':
        which_experiment = int(sys.argv[2])
        which_file = int(sys.argv[3])
        mode_hist(which_experiment, which_file)
        exit(0)
    elif mode == 'keys':
        which_experiment = int(sys.argv[2])
        which_file = int(sys.argv[3])
        mode_keys(which_experiment, which_file)
        exit(0)
