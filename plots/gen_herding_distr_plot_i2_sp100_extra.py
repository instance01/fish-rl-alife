import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


base_path = '../pickles/'

with open(base_path + 'i2_p150_two_net_herding.pickle', 'rb') as f:
    i2_p150 = pickle.load(f)


print(i2_p150)


def func(x, a, b, c, d, e):
    return a + b * x + c * x**2 + d * x**3 + e * x**4


def plot_one(axs, idx, data, title, data_idx, ylabel, c='#B80F0A'):
    axs.set_title(title)
    axs.set_ylabel(ylabel)
    axs.set_xlabel(r'Initial predator survival time t $\times$ 100')
    x = [str(int(x[1:]) // 100) for x in data[0]]
    y = [x[data_idx] for x in data[1]]
    axs.bar(x, y, color=c)
    x = np.arange(len(y))
    fittedParameters, pcov = curve_fit(func, x, y)
    y_fit = func(x, *fittedParameters)
    axs.plot(x, y_fit, 'k:')


fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharey=True)
plot_one(axs, 2, i2_p150, '2 initial prey', 0, 'Herding ratio', '#009440')
plt.tight_layout()
plt.ylim(0., 1.0)
plt.show()
fig.savefig("i2_sp100_herding_good.pdf", bbox_inches='tight')


fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharey=True)
plot_one(axs, 2, i2_p150, '2 initial prey', 3, 'Defect (fear)')
plt.tight_layout()
plt.ylim(0., 1.0)
plt.show()
fig.savefig("i2_sp100_herding_defect.pdf", bbox_inches='tight')


fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharey=True)
plot_one(axs, 2, i2_p150, '2 initial prey', 2, 'PPO Fails')
plt.tight_layout()
plt.ylim(0., 1.0)
plt.show()
fig.savefig("i2_sp100_herding_fail.pdf", bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharey=True)
plot_one(axs, 2, i2_p150, '2 initial prey', 4, 'Defect (greed)')
plt.tight_layout()
plt.ylim(0., 1.0)
plt.show()
fig.savefig("i2_sp100_herding_greed.pdf", bbox_inches='tight')
