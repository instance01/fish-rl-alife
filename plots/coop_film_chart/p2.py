from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data


fn = get_sample_data("/Users/instance/Desktop/fword/ritz/fishDomain/plots/coop_film_chart/placeholder3.png", asfileobj=True)
arr = read_png(fn)
plt.imshow(arr)
plt.show()
