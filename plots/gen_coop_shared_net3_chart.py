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


def _prep(data, failure=False, prefix='i10'):
    ret = {}
    ret_labels = {}
    for k, v in zip(data[0], data[1]):
        mean = v[0]
        ci = mean - v[1][0]
        if failure:
            mean = v[2]
            ci = mean - v[3][0]
        if np.isnan(ci) or ci == 1.0:
            print('ENCOUNTERED NAN!')
            ci = 0
        ret[k] = mean
        ret_labels[k] = '%0.2f\n±%0.2f' % (round(mean, 2), round(ci, 2))
    print(ret_labels)

    print('##', prefix)
    print(ret)

    # Note: We also have data for s025.

    # data = [
    #     [ret['r4_s025'], ret['r4_s03'], ret['r4_s035'], ret['r4_s04'], ret['r4_s05']],
    #     [ret['r6_s025'], ret['r6_s03'], ret['r6_s035'], ret['r6_s04'], ret['r6_s05']],
    #     [ret['r10_s025'], ret['r10_s03'], ret['r10_s035'], ret['r10_s04'], ret['r10_s05']]
    # ]
    # label_data = [
    #     [ret_labels['r4_s025'], ret_labels['r4_s03'], ret_labels['r4_s035'], ret_labels['r4_s04'], ret_labels['r4_s05']],
    #     [ret_labels['r6_s025'], ret_labels['r6_s03'], ret_labels['r6_s035'], ret_labels['r6_s04'], ret_labels['r6_s05']],
    #     [ret_labels['r10_s025'], ret_labels['r10_s03'], ret_labels['r10_s035'], ret_labels['r10_s04'], ret_labels['r10_s05']]
    # ]

    data = [
        [ret['r4_s03'], ret['r4_s035'], ret['r4_s04'], ret['r4_s05']],
        [ret['r6_s03'], ret['r6_s035'], ret['r6_s04'], ret['r6_s05']],
        [ret['r10_s03'], ret['r10_s035'], ret['r10_s04'], ret['r10_s05']]
    ]
    label_data = [
        [ret_labels['r4_s03'], ret_labels['r4_s035'], ret_labels['r4_s04'], ret_labels['r4_s05']],
        [ret_labels['r6_s03'], ret_labels['r6_s035'], ret_labels['r6_s04'], ret_labels['r6_s05']],
        [ret_labels['r10_s03'], ret_labels['r10_s035'], ret_labels['r10_s04'], ret_labels['r10_s05']]
    ]

    return data, label_data


base_path = '../pickles/'
failure = False
with open(base_path + 'vd15_coop_net3_shared.pickle', 'rb') as f:
    i5_data, i5_label_data = _prep(pickle.load(f), failure, prefix='vd15')
# with open(base_path + 'vd20_coop_net3.pickle', 'rb') as f:
#     i5_data2, i5_label_data2 = _prep(pickle.load(f), failure, prefix='vd20')
with open(base_path + 'vd25_coop_net3_shared.pickle', 'rb') as f:
    i5_data3, i5_label_data3 = _prep(pickle.load(f), failure, prefix='vd25')
# with open(base_path + 'vd30_coop_net3.pickle', 'rb') as f:
#     i5_data4, i5_label_data4 = _prep(pickle.load(f), failure, prefix='vd30')
# with open(base_path + 'vd35_coop_net3.pickle', 'rb') as f:
#     i5_data5, i5_label_data5 = _prep(pickle.load(f), failure, prefix='vd35')


def doit(i5_data, i5_label_data, i5_data2, i5_label_data2, i5_data3, i5_label_data3, i5_data4, i5_label_data4, i5_data5, i5_label_data5,):
    print(i5_data)

    y_labels = ['4', '6', '10']
    x_labels = ['.03', '.035', '.04', '.05']

    cmap_mod = truncate_colormap('Greens', minval=.3, maxval=.99)
    # fig, (ax, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(14.5, 2.0), constrained_layout=True)
    fig, (ax, ax3) = plt.subplots(1, 2, figsize=(14.5, 2.0), constrained_layout=True)
    im = ax.imshow(i5_data, cmap=cmap_mod, vmin=0, vmax=1)
    # im2 = ax2.imshow(i5_data2, cmap=cmap_mod, vmin=0, vmax=1)
    im3 = ax3.imshow(i5_data3, cmap=cmap_mod, vmin=0, vmax=1)
    # im4 = ax4.imshow(i5_data4, cmap=cmap_mod, vmin=0, vmax=1)
    # im5 = ax5.imshow(i5_data5, cmap=cmap_mod, vmin=0, vmax=1)

    # Colorbar
    # cbar = ax.figure.colorbar(im, ax=[ax, ax2, ax3, ax4, ax5], aspect=60)
    cbar = ax.figure.colorbar(im, ax=[ax, ax3], aspect=60)
    cbar.ax.set_ylabel('Avg Cooperation Rate', rotation=-90, va="bottom")
    if failure:
        cbar.ax.set_ylabel('Avg Failure Rate', rotation=-90, va="bottom")

    # Ticks and labels
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    # ax2.set_xticks(np.arange(len(x_labels)))
    # ax2.set_yticks(np.arange(len(y_labels)))
    # ax2.set_xticklabels(x_labels)
    # ax2.set_yticklabels(y_labels)
    ax3.set_xticks(np.arange(len(x_labels)))
    ax3.set_yticks(np.arange(len(y_labels)))
    ax3.set_xticklabels(x_labels)
    ax3.set_yticklabels(y_labels)
    # ax4.set_xticks(np.arange(len(x_labels)))
    # ax4.set_yticks(np.arange(len(y_labels)))
    # ax4.set_xticklabels(x_labels)
    # ax4.set_yticklabels(y_labels)
    # ax5.set_xticks(np.arange(len(x_labels)))
    # ax5.set_yticks(np.arange(len(y_labels)))
    # ax5.set_xticklabels(x_labels)
    # ax5.set_yticklabels(y_labels)
    ax.set_ylabel('Killzone Radius', rotation=90, va="bottom")
    ax.set_xlabel('Shark speed', rotation=0, va="top")
    # ax2.set_xlabel('Shark speed', rotation=0, va="top")
    ax3.set_xlabel('Shark speed', rotation=0, va="top")
    # ax4.set_xlabel('Shark speed', rotation=0, va="top")
    # ax5.set_xlabel('Shark speed', rotation=0, va="top")

    # Text annotations
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            # ax.text(j, i, i10_label_data[i][j], ha="center", va="center", color="w")
            ax.text(j, i, i5_label_data[i][j], ha="center", va="center", color="w")
            # ax2.text(j, i, i5_label_data2[i][j], ha="center", va="center", color="w")
            ax3.text(j, i, i5_label_data3[i][j], ha="center", va="center", color="w")
            # ax4.text(j, i, i5_label_data4[i][j], ha="center", va="center", color="w")
            # ax5.text(j, i, i5_label_data5[i][j], ha="center", va="center", color="w")

    plt.show()
    return fig


def doit_single(id_, i5_data, i5_label_data, i5_data2, i5_label_data2):
    print(i5_data)

    y_labels = ['4', '6', '10']
    x_labels = ['.03', '.035', '.04', '.05']

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

        print(id_ + "_coop_net3_shared" + str(m) + ".pdf")
        fig.savefig(id_ + "_coop_net3_shared" + str(m) + ".pdf", bbox_inches='tight')
        plt.show()


print('DOING i5')
# fig = doit(i5_data, i5_label_data, None, None, i5_data3, i5_label_data3, None, None, None, None)
# if failure:
#     fig.savefig("i5_coop_net3_failure.pdf", bbox_inches='tight')
# else:
#     fig.savefig("i5_coop_net3.pdf", bbox_inches='tight')
doit_single('i5', i5_data, i5_label_data, i5_data3, i5_label_data3)
