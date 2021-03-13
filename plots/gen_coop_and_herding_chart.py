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
        # mean = v[0]
        # ci = mean - v[1][0]
        # if np.isnan(ci):
        #     print('ENCOUNTERED NAN!')
        #     ci = 0
        # ret[k] = mean
        # ret_labels[k] = '%0.2f\n±%0.2f' % (round(mean, 2), round(ci, 2))
        mean = v[3]
        ret[k] = mean
        ret_labels[k] = '%0.2f' % round(mean, 2)
    print(ret_labels)

    data = [
        [ret[prefix + '_r4_s03'], ret[prefix + '_r4_s035'], ret[prefix + '_r4_s04'], ret[prefix + '_r4_s05']],
        [ret[prefix + '_r6_s03'], ret[prefix + '_r6_s035'], ret[prefix + '_r6_s04'], ret[prefix + '_r6_s05']],
        [ret[prefix + '_r10_s03'], ret[prefix + '_r10_s035'], ret[prefix + '_r10_s04'], ret[prefix + '_r10_s05']]
    ]
    label_data = [
        [ret_labels[prefix + '_r4_s03'], ret_labels[prefix + '_r4_s035'], ret_labels[prefix + '_r4_s04'], ret_labels[prefix + '_r4_s05']],
        [ret_labels[prefix + '_r6_s03'], ret_labels[prefix + '_r6_s035'], ret_labels[prefix + '_r6_s04'], ret_labels[prefix + '_r6_s05']],
        [ret_labels[prefix + '_r10_s03'], ret_labels[prefix + '_r10_s035'], ret_labels[prefix + '_r10_s04'], ret_labels[prefix + '_r10_s05']]
    ]

    return data, label_data


def plot(data, label_data, do_cbar=False):
    y_labels = ['4', '6', '10']
    x_labels = ['.03', '.035', '.04', '.05']

    cmap_mod = truncate_colormap('Greens', minval=.4, maxval=.99)
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.0), constrained_layout=True)
    im = ax.imshow(data, cmap=cmap_mod, vmin=0.0, vmax=1.0)

    # Colorbar
    if do_cbar:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r'Avg Cooperation Ratio $\kappa$', rotation=-90, va="bottom")

    # Ticks and labels
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel('Shared Catch Zone Radius', rotation=90, va="bottom", fontsize=9)
    ax.set_xlabel('Predator Speed', rotation=0, va="top", fontsize=9)

    # Text annotations
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            ax.text(j, i, label_data[i][j], ha="center", va="center", color="w")

    # ax.set_title("Fish population: 10")
    # ax2.set_title("Fish population: 5")
    plt.show()
    return fig


base_path = '../pickles/'

with open(base_path + 'i5_coop_herding.pickle', 'rb') as f:
    i5_data, i5_label_data = _prep(pickle.load(f), prefix='i5')
with open(base_path + 'i10_coop_herding.pickle', 'rb') as f:
    i10_data, i10_label_data = _prep(pickle.load(f))
with open(base_path + 'vd20_coop_herding.pickle', 'rb') as f:
    i5_vd20_data, i5_vd20_label_data = _prep(pickle.load(f), prefix='i5')

fig = plot(i5_data, i5_label_data)
fig.savefig("i5_coop_herding.pdf", bbox_inches='tight')
fig = plot(i10_data, i10_label_data)
fig.savefig("i10_coop_herding.pdf", bbox_inches='tight')
fig = plot(i5_vd20_data, i5_vd20_label_data)
fig.savefig("i5_vd20_coop_herding.pdf", bbox_inches='tight')
