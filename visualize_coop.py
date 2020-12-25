import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    '''truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)'''    
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))

    # arr = np.linspace(0, 50, 100).reshape((10, 10))
    # fig, ax = plt.subplots(ncols=2)
    # ax[0].imshow(arr, interpolation='nearest', cmap=cmapIn)
    # ax[1].imshow(arr, interpolation='nearest', cmap=new_cmap)
    # plt.show()

    return new_cmap


def _prep(data, prefix='i10'):
    ret = {}
    ret_labels = {}
    for k, v in zip(data[0], data[1]):
        mean = v[0]
        ci = mean - v[1][0]
        if np.isnan(ci):
            print('ENCOUNTERED NAN!')
            ci = 0
        ret[k] = mean
        ret_labels[k] = '%0.2f\n±%0.2f' % (round(mean, 2), round(ci, 2))
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


base_path = 'pickles/'

with open(base_path + 'i10_coop.pickle', 'rb') as f:
    i10_data, i10_label_data = _prep(pickle.load(f))
with open(base_path + 'i5_coop.pickle', 'rb') as f:
    i5_data, i5_label_data = _prep(pickle.load(f), prefix='i5')


print(i10_data)
print(i5_data)


# y_labels = ['r4', 'r6', 'r10']
# x_labels = ['s03', 's035', 's04', 's05']
y_labels = ['4', '6', '10']
x_labels = ['.03', '.035', '.04', '.05']

cmap_mod = truncate_colormap('Greens', minval=.3, maxval=.99)
# fig, (ax, ax2) = plt.subplots(1, 2, figsize=(5,3.3))
# fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 6.5))
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6, 2.2), constrained_layout=True)
im = ax.imshow(i10_data, cmap=cmap_mod)
im2 = ax2.imshow(i5_data, cmap=cmap_mod)

# Colorbar
cbar = ax.figure.colorbar(im, ax=[ax, ax2])
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
ax.set_ylabel('Killzone Radius', rotation=90, va="bottom")
ax.set_xlabel('Shark speed', rotation=0, va="top")
ax2.set_xlabel('Shark speed', rotation=0, va="top")

# Text annotations
for i in range(len(y_labels)):
    for j in range(len(x_labels)):
        ax.text(j, i, i10_label_data[i][j], ha="center", va="center", color="w")
        ax2.text(j, i, i5_label_data[i][j], ha="center", va="center", color="w")

ax.set_title("Fish population: 10")
ax2.set_title("Fish population: 5")
# fig.tight_layout()
plt.show()
