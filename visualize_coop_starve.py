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


def _prep(data, prefix='i10'):
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
    print(ret_labels)

    data = [
        [ret[prefix + '_r4_s03'], ret[prefix + '_r4_s04'], ret[prefix + '_r4_s05']],
        [ret[prefix + '_r6_s03'], ret[prefix + '_r6_s04'], ret[prefix + '_r6_s05']],
        [ret[prefix + '_r10_s03'], ret[prefix + '_r10_s04'], ret[prefix + '_r10_s05']]
    ]
    label_data = [
        [ret_labels[prefix + '_r4_s03'], ret_labels[prefix + '_r4_s04'], ret_labels[prefix + '_r4_s05']],
        [ret_labels[prefix + '_r6_s03'], ret_labels[prefix + '_r6_s04'], ret_labels[prefix + '_r6_s05']],
        [ret_labels[prefix + '_r10_s03'], ret_labels[prefix + '_r10_s04'], ret_labels[prefix + '_r10_s05']]
    ]

    return data, label_data


base_path = 'pickles/'
with open(base_path + 'i5_coop_starve.pickle', 'rb') as f:
    i5_data, i5_label_data = _prep(pickle.load(f), prefix='t2000_i5')
with open(base_path + 'i5_coop_starve.pickle', 'rb') as f:
    i5_data2, i5_label_data2 = _prep(pickle.load(f), prefix='t1500_i5')
with open(base_path + 'i5_coop_starve.pickle', 'rb') as f:
    i5_data3, i5_label_data3 = _prep(pickle.load(f), prefix='t1000_i5')

with open(base_path + 'i10_coop_starve.pickle', 'rb') as f:
    i10_data, i10_label_data = _prep(pickle.load(f), prefix='t2000_i10')
with open(base_path + 'i10_coop_starve.pickle', 'rb') as f:
    i10_data2, i10_label_data2 = _prep(pickle.load(f), prefix='t1500_i10')
with open(base_path + 'i10_coop_starve.pickle', 'rb') as f:
    i10_data3, i10_label_data3 = _prep(pickle.load(f), prefix='t1000_i10')


def doit(i5_data, i5_label_data, i5_data2, i5_label_data2, i5_data3, i5_label_data3):
    print(i5_data)

    y_labels = ['4', '6', '10']
    x_labels = ['.03', '.04', '.05']

    cmap_mod = truncate_colormap('Greens', minval=.3, maxval=.99)
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(6.5, 2.0), constrained_layout=True)
    im = ax.imshow(i5_data, cmap=cmap_mod)
    im2 = ax2.imshow(i5_data2, cmap=cmap_mod)
    im3 = ax3.imshow(i5_data3, cmap=cmap_mod)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=[ax, ax2, ax3], aspect=40)
    cbar.ax.set_ylabel('Avg Coop Ratio', rotation=-90, va="bottom")

    # Ticks and labels
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax2.set_xticks(np.arange(len(x_labels)))
    ax2.set_yticks(np.arange(len(y_labels)))
    ax2.set_xticklabels(x_labels)
    ax2.set_yticklabels(y_labels)
    ax3.set_xticks(np.arange(len(x_labels)))
    ax3.set_yticks(np.arange(len(y_labels)))
    ax3.set_xticklabels(x_labels)
    ax3.set_yticklabels(y_labels)
    ax.set_ylabel('Killzone Radius', rotation=90, va="bottom")
    ax.set_xlabel('Shark speed', rotation=0, va="top")
    ax2.set_xlabel('Shark speed', rotation=0, va="top")
    ax3.set_xlabel('Shark speed', rotation=0, va="top")

    # Text annotations
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            # ax.text(j, i, i10_label_data[i][j], ha="center", va="center", color="w")
            ax.text(j, i, i5_label_data[i][j], ha="center", va="center", color="w")
            ax2.text(j, i, i5_label_data2[i][j], ha="center", va="center", color="w")
            ax3.text(j, i, i5_label_data3[i][j], ha="center", va="center", color="w")

    plt.show()


doit(i5_data, i5_label_data, i5_data2, i5_label_data2, i5_data3, i5_label_data3)
doit(i10_data, i10_label_data, i10_data2, i10_label_data2, i10_data3, i10_label_data3)
