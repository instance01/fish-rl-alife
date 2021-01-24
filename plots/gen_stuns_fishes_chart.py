import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    '''truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)'''
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))

    return new_cmap


def _prep(data, prefix='t1000'):
    ret = {}
    ret_labels = {}
    for k, v in zip(data[0], data[1]):
        mean = v[0]
        ci = mean - v[1][0]
        if np.isnan(ci) or ci == 1.0:
            print('ENCOUNTERED NAN!')
            ci = 0
        ret[k] = mean
        ret_labels[k] = '%0.2f\n±%0.2f' % (round(mean, 2), round(ci, 2))
        # ret_labels[k] = '%0.2f' % mean
    print(ret_labels)

    print('##', prefix)
    print(ret)

    data = [
        [ret[prefix + '_i2_d100'], ret[prefix + '_i4_d100'], ret[prefix + '_i6_d100'], ret[prefix + '_i8_d100'], ret[prefix + '_i10_d100']],
        [ret[prefix + '_i2_d300'], ret[prefix + '_i4_d300'], ret[prefix + '_i6_d300'], ret[prefix + '_i8_d300'], ret[prefix + '_i10_d300']]
    ]
    label_data = [
        [ret_labels[prefix + '_i2_d100'], ret_labels[prefix + '_i4_d100'], ret_labels[prefix + '_i6_d100'], ret_labels[prefix + '_i8_d100'], ret_labels[prefix + '_i10_d100']],
        [ret_labels[prefix + '_i2_d300'], ret_labels[prefix + '_i4_d300'], ret_labels[prefix + '_i6_d300'], ret_labels[prefix + '_i8_d300'], ret_labels[prefix + '_i10_d300']]
    ]

    return data, label_data


base_path = '../pickles/'
with open(base_path + 't1000_stuns_fishes.pickle', 'rb') as f:
    i5_data, i5_label_data = _prep(pickle.load(f), prefix='t1000')
with open(base_path + 't3000_stuns_fishes.pickle', 'rb') as f:
    i5_data2, i5_label_data2 = _prep(pickle.load(f), prefix='t3000')


def doit(i5_data, i5_label_data, i5_data2, i5_label_data2):
    print('doing it')
    print(i5_data)
    print(i5_label_data)

    x_labels = ['2', '4', '6', '8', '10']
    y_labels = ['100', '300']

    data_max = np.max([i5_data, i5_data2])

    cmap_mod = truncate_colormap('Greens', minval=.3, maxval=.99)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.5, 1.8), constrained_layout=True)
    im = ax.imshow(i5_data, cmap=cmap_mod, vmin=0, vmax=data_max)
    im2 = ax2.imshow(i5_data2, cmap=cmap_mod, vmin=0, vmax=data_max)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=[ax, ax2], aspect=40)
    cbar.ax.set_ylabel('Avg Number of Stuns', rotation=-90, va="bottom")

    # Ticks and labels
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax2.set_xticks(np.arange(len(x_labels)))
    ax2.set_yticks(np.arange(len(y_labels)))
    ax2.set_xticklabels(x_labels)
    ax2.set_yticklabels(y_labels)
    ax.set_ylabel('Stun duration', rotation=90, va="bottom")
    ax.set_xlabel('Initial Fishes', rotation=0, va="top")
    ax2.set_xlabel('Initial Fishes', rotation=0, va="top")

    # Text annotations
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            ax.text(j, i, i5_label_data[i][j], ha="center", va="center", color="w")
            ax2.text(j, i, i5_label_data2[i][j], ha="center", va="center", color="w")

    plt.show()
    return fig


def doit_single(id_, i5_data, i5_label_data, i5_data2, i5_label_data2):
    y_labels = ['4', '6', '10']
    x_labels = ['.03', '.04', '.05']

    cmap_mod = truncate_colormap('Greens', minval=.3, maxval=.99)

    data = [i5_data, i5_data2]
    label_data = [i5_label_data, i5_label_data2]

    for m in range(2):
        d = data[m]
        ld = label_data[m]
        fig, ax = plt.subplots(1, 1, figsize=(3.0, 2.0), constrained_layout=True)
        im = ax.imshow(d, cmap=cmap_mod, vmin=0, vmax=1)

        # Colorbar
        if m > 1:
            cbar = ax.figure.colorbar(im, ax=[ax], aspect=20)
            cbar.ax.set_ylabel('Avg Cooperation Rate', rotation=-90, va="bottom")

        # Ticks and labels
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.set_ylabel('Killzone Radius', rotation=90, va="bottom")
        ax.set_xlabel('Shark speed', rotation=0, va="top")

        # Text annotations
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                ax.text(j, i, ld[i][j], ha="center", va="center", color="w")

        fig.savefig(id_ + "_stuns_fishes" + str(m) + ".pdf", bbox_inches='tight')
        plt.show()


print('DOING i5')
fig = doit(i5_data, i5_label_data, i5_data2, i5_label_data2)
fig.savefig("stuns_fishes.pdf", bbox_inches='tight')

# print('DOING i5')
# fig = doit('i5', i5_data, i5_label_data, i5_data2, i5_label_data2, i5_data3, i5_label_data3, i5_data4, i5_label_data4)
# print('DOING i10')
# fig = doit_single('i10', i10_data, i10_label_data, i10_data2, i10_label_data2, i10_data3, i10_label_data3, i10_data4, i10_label_data4)
