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
        # ret_labels[k] = '%0.2f' % mean
    print(ret_labels)

    print('##', prefix)
    print(ret)

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


base_path = '../pickles/'
with open(base_path + 'i5_1_stuns.pickle', 'rb') as f:
    i5_data, i5_label_data = _prep(pickle.load(f), prefix='t2000_i5')
with open(base_path + 'i5_1_stuns.pickle', 'rb') as f:
    i5_data2, i5_label_data2 = _prep(pickle.load(f), prefix='t1500_i5')
with open(base_path + 'i5_2_stuns.pickle', 'rb') as f:
    i5_data3, i5_label_data3 = _prep(pickle.load(f), prefix='t1000_i5')
with open(base_path + 'i5_2_stuns.pickle', 'rb') as f:
    i5_data4, i5_label_data4 = _prep(pickle.load(f), prefix='t500_i5')

with open(base_path + 'i10_1_stuns.pickle', 'rb') as f:
    i10_data, i10_label_data = _prep(pickle.load(f), prefix='t2000_i10')
with open(base_path + 'i10_1_stuns.pickle', 'rb') as f:
    i10_data2, i10_label_data2 = _prep(pickle.load(f), prefix='t1500_i10')
with open(base_path + 'i10_2_stuns.pickle', 'rb') as f:
    i10_data3, i10_label_data3 = _prep(pickle.load(f), prefix='t1000_i10')
with open(base_path + 'i10_2_stuns.pickle', 'rb') as f:
    i10_data4, i10_label_data4 = _prep(pickle.load(f), prefix='t500_i10')


def doit(i5_data, i5_label_data, i5_data2, i5_label_data2, i5_data3, i5_label_data3, i5_data4, i5_label_data4):
    print(i5_data)

    y_labels = ['4', '6', '10']
    x_labels = ['.03', '.04', '.05']

    data_max = np.max([i5_data, i5_data2, i5_data3, i5_data4])

    cmap_mod = truncate_colormap('Greens', minval=.3, maxval=.99)
    fig, (ax, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8.5, 2.0), constrained_layout=True)
    im = ax.imshow(i5_data4, cmap=cmap_mod, vmin=0, vmax=data_max)
    im2 = ax2.imshow(i5_data, cmap=cmap_mod, vmin=0, vmax=data_max)
    im3 = ax3.imshow(i5_data2, cmap=cmap_mod, vmin=0, vmax=data_max)
    im4 = ax4.imshow(i5_data3, cmap=cmap_mod, vmin=0, vmax=data_max)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=[ax, ax2, ax3, ax4], aspect=40)
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
    ax3.set_xticks(np.arange(len(x_labels)))
    ax3.set_yticks(np.arange(len(y_labels)))
    ax3.set_xticklabels(x_labels)
    ax3.set_yticklabels(y_labels)
    ax4.set_xticks(np.arange(len(x_labels)))
    ax4.set_yticks(np.arange(len(y_labels)))
    ax4.set_xticklabels(x_labels)
    ax4.set_yticklabels(y_labels)
    ax.set_ylabel('Shared Catch Zone Radius', rotation=90, va="bottom")
    ax.set_xlabel('Predator Speed', rotation=0, va="top")
    ax2.set_xlabel('Predator Speed', rotation=0, va="top")
    ax3.set_xlabel('Predator Speed', rotation=0, va="top")
    ax4.set_xlabel('Predator Speed', rotation=0, va="top")

    # Text annotations
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            # ax.text(j, i, i10_label_data[i][j], ha="center", va="center", color="w")
            ax.text(j, i, i5_label_data4[i][j], ha="center", va="center", color="w")
            ax2.text(j, i, i5_label_data[i][j], ha="center", va="center", color="w")
            ax3.text(j, i, i5_label_data2[i][j], ha="center", va="center", color="w")
            ax4.text(j, i, i5_label_data3[i][j], ha="center", va="center", color="w")

    plt.show()
    return fig


def doit_single(id_, i5_data, i5_label_data, i5_data2, i5_label_data2, i5_data3, i5_label_data3, i5_data4, i5_label_data4):
    print(i5_data)

    y_labels = ['4', '6', '10']
    x_labels = ['.03', '.04', '.05']

    cmap_mod = truncate_colormap('Greens', minval=.5, maxval=.99)

    data = [i5_data4, i5_data, i5_data2, i5_data3]
    label_data = [i5_label_data4, i5_label_data, i5_label_data2, i5_label_data3]

    for m in range(4):
        d = data[m]
        ld = label_data[m]
        fig, ax = plt.subplots(1, 1, figsize=(3.0, 2.0), constrained_layout=True)
        im = ax.imshow(d, cmap=cmap_mod, vmin=0, vmax=1)

        # Colorbar
        if m > 1:
            cbar = ax.figure.colorbar(im, ax=[ax], aspect=20)
            cbar.ax.set_ylabel('Avg Cooperation Ratio', rotation=-90, va="bottom")

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
                ax.text(j, i, ld[i][j], ha="center", va="center", color="w")

        print(id_ + "_stuns" + str(m) + ".pdf")
        fig.savefig(id_ + "_stuns" + str(m) + ".pdf", bbox_inches='tight')
        plt.show()


print('DOING i5')
fig = doit(i5_data, i5_label_data, i5_data2, i5_label_data2, i5_data3, i5_label_data3, i5_data4, i5_label_data4)
fig.savefig("i5_stuns.pdf", bbox_inches='tight')
print('DOING i10')
fig = doit(i10_data, i10_label_data, i10_data2, i10_label_data2, i10_data3, i10_label_data3, i10_data4, i10_label_data4)
fig.savefig("i10_stuns.pdf", bbox_inches='tight')

# print('DOING i5')
# fig = doit('i5', i5_data, i5_label_data, i5_data2, i5_label_data2, i5_data3, i5_label_data3, i5_data4, i5_label_data4)
# print('DOING i10')
# fig = doit_single('i10', i10_data, i10_label_data, i10_data2, i10_label_data2, i10_data3, i10_label_data3, i10_data4, i10_label_data4)
