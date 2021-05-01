import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap


def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))

    return new_cmap


def truncate_colormap2(cmapIn, minval=0.0, maxval=1.0, n=100):
    return colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))


# cmap_mod = truncate_colormap('viridis_r', minval=.4, maxval=.99)
# cmap_mod = truncate_colormap('summer_r', minval=.4, maxval=.99)
# cmap_mod = truncate_colormap('summer_r', minval=.0, maxval=.99)
# cmap_mod = truncate_colormap('bone_r', minval=.0, maxval=.99)
# cmap_mod = sns.light_palette("mako", as_cmap=True)

# cmap_mod = sns.color_palette("mako", as_cmap=True).reversed()
# cmap_mod = truncate_colormap2(cmap_mod, minval=.0, maxval=.8)
cmap_mod = sns.color_palette("mako", as_cmap=True).reversed()
cmap_mod = truncate_colormap2(cmap_mod, minval=-.2, maxval=.9)
# cmap_mod = truncate_colormap2(cmap_mod, minval=-.0, maxval=.9)  # Used for failure chart

# n = cmap_mod.N
# cmap_mod = cmap_mod(np.arange(cmap_mod.N))
# cmap_mod[:, -1] = .95  # np.linspace(0, 1, n)
# cmap_mod = ListedColormap(cmap_mod)
print(cmap_mod(179))
cutoff = .55
col1 = cmap_mod(179)  # #37669e
col2 = cmap_mod(159)
col3 = cmap_mod(199)
