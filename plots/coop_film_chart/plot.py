from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data


# fig = plt.figure(figsize=(15, 5))
fig = plt.figure(figsize=(8, 5))
ax = fig.gca(projection='3d', proj_type='ortho')

# ax.set_zlim(-2.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


def plot(fname, z, quality=5):
    fn = get_sample_data("/Users/instance/Desktop/fword/ritz/fishDomain/plots/coop_film_chart/ff/aa/" + fname, asfileobj=True)
    arr = read_png(fn)
    # 10 is equal length of x and y axises of your surface
    stepX, stepY = 10. / arr.shape[0], 10. / arr.shape[1]

    X1 = np.arange(-5, 5, stepX)
    Y1 = np.arange(-5, 5, stepY)
    X1, Y1 = np.meshgrid(X1, Y1)
    # stride args allows to determine image quality
    # stride = 1 work slow
    Z = np.ones(shape=Y1.shape) * z
    # arr = arr.reshape(100,96,4)
    print(X1.shape, Y1.shape, arr.shape, Z.shape)
    ax.plot_surface(X1, Z, Y1, rstride=quality, cstride=quality, facecolors=arr)


"""Scaling is done from here.."""
x_scale=1
y_scale=2
z_scale=1
scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0 / scale.max())
scale[3, 3] = 1.0
def short_proj():
  return np.dot(Axes3D.get_proj(ax), scale)
ax.get_proj=short_proj
"""to here"""

quality = 1

# ax.view_init(elev=15., azim=-12)
# plot('1.png', 0., quality)
# plot('2.png', 20., quality)
# plot('4.png', 40., quality)
# plot('5.png', 60., quality)
# plot('6.png', 80., quality)
# plot('7.png', 100., quality)
# plot('8.png', 120., quality)
# plot('9.png', 140., quality)
# plot('10.png', 160., quality)

ax.view_init(elev=15., azim=-24)
plot('1.png', 0., quality)
plot('4.png', 10., quality)
plot('7.png', 20., quality)
plot('9.png', 30., quality)
plot('10.png', 40., quality)

ax.set_zticks([])
# TODO Maybe show y ticks?
# https://stackoverflow.com/questions/20416609/r
# for tic in ax.xaxis.get_major_ticks():
#     tic.tick1On = tic.tick2On = False
plt.yticks([], [])
plt.xticks([], [])
plt.tight_layout()
plt.show()
plt.savefig("coop_movie_new.pdf", bbox_inches='tight')
