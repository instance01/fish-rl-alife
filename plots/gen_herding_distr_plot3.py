import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


base_path = '../pickles/'

with open(base_path + 'i10_p150_two_net_herding3.pickle', 'rb') as f:
    i10_p150 = pickle.load(f)
with open(base_path + 'i5_p150_two_net_herding3.pickle', 'rb') as f:
    i5_p150 = pickle.load(f)
with open(base_path + 'i2_p150_sp500_two_net_herding3.pickle', 'rb') as f:
    i2_p150 = pickle.load(f)


print(i10_p150)
print(i5_p150)
print(i2_p150)


def func(x, a, b, c, d, e):
    return a + b * x + c * x**2 + d * x**3 + e * x**4


def plot_one(axs, idx, data, title, data_idx, ylabel, c='#B80F0A'):
    axs[idx].set_title(title)
    axs[idx].set_ylabel(ylabel)
    axs[idx].set_xlabel(r'Initial predator survival time t $\times$ 100')
    x = [str(int(x[1:]) // 100) for x in data[0]]
    y = [x[data_idx] for x in data[1]]
    axs[idx].bar(x, y, color=c)
    x = np.arange(len(y))
    fittedParameters, pcov = curve_fit(func, x, y)
    y_fit = func(x, *fittedParameters)
    axs[idx].plot(x, y_fit, 'k:')


fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
plot_one(axs, 0, i10_p150, '10 initial prey', 0, 'Herding ratio', '#009440')
plot_one(axs, 1, i5_p150, '5 initial prey', 0, 'Herding ratio', '#009440')
plot_one(axs, 2, i2_p150, '2 initial prey', 0, 'Herding ratio', '#009440')
plt.tight_layout()
plt.ylim(0., 1.0)
plt.show()
fig.savefig("herding_good3.pdf", bbox_inches='tight')


fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
plot_one(axs, 0, i10_p150, '10 initial prey', 3, 'Defect (fear)')
plot_one(axs, 1, i5_p150, '5 initial prey', 3, 'Defect (fear)')
plot_one(axs, 2, i2_p150, '2 initial prey', 3, 'Defect (fear)')
plt.tight_layout()
plt.ylim(0., 1.0)
plt.show()
fig.savefig("herding_defect3.pdf", bbox_inches='tight')


fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
plot_one(axs, 0, i10_p150, '10 initial prey', 2, 'PPO Fails')
plot_one(axs, 1, i5_p150, '5 initial prey', 2, 'PPO Fails')
plot_one(axs, 2, i2_p150, '2 initial prey', 2, 'PPO Fails')
plt.tight_layout()
plt.ylim(0., 1.0)
plt.show()
fig.savefig("herding_fail3.pdf", bbox_inches='tight')

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
plot_one(axs, 0, i10_p150, '10 initial prey', 4, 'Defect (greed)')
plot_one(axs, 1, i5_p150, '5 initial prey', 4, 'Defect (greed)')
plot_one(axs, 2, i2_p150, '2 initial prey', 4, 'Defect (greed)')
plt.tight_layout()
plt.ylim(0., 1.0)
plt.show()
fig.savefig("herding_greed3.pdf", bbox_inches='tight')
